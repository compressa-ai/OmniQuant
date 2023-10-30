import numpy as np
import torch
import torch.nn as nn
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from quantize.int_linear import QuantLinear
import copy
import math
import utils
import os
import pdb
import shutil



class Inps:
    def __init__(
            self, name, folder,
            nsamples, seqlen, hidden_size,
            dtype, device, nsamples_in_memory=128, batch_size=1):

        if nsamples % nsamples_in_memory != 0:
            raise ValueError(
                'Please make sure `nsamples` is divisible by `nsamples_in_memory` without a remainder.'
            )

        self.name = name
        self._folder = folder
        self.folder = os.path.join(self._folder, name)

        self.nsamples_total = nsamples
        self.nsamples_buffer = nsamples_in_memory
        self.seqlen = seqlen
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.device = device

        self.inps = self._init()

        self.batch_size = batch_size
        self._buffer_count = 0

    def deepcopy(self, name, folder):
        result = Inps(
            name=name, folder=folder,
            nsamples=self.nsamples_total, seqlen=self.seqlen,
            hidden_size=self.hidden_size,
            dtype=self.dtype, device=self.device,
            nsamples_in_memory=self.nsamples_buffer, batch_size=self.batch_size
        )

        assert not os.path.isdir(result.folder)

        shutil.copytree(self.folder, result.folder)

        assert len(os.listdir(result.folder)) == len(os.listdir(self.folder))

        return result

    def __len__(self):
        return self.nsamples_total

    def __setitem__(self, key, value):
        low = self._buffer_count * self.nsamples_buffer
        high = (self._buffer_count + 1) * self.nsamples_buffer

        if low <= key < high:
            self.inps[key] = value
            return

        if key >= high:
            self._save()
            self.inps = self._load_next()
        elif key < low:
            # TODO: assuming we start from zero
            assert key == 0

            self._save()
            self._buffer_count = -1
            self.inps = self._load_next()
        else:
            assert False

        self[key] = value

    def __getitem__(self, key):
        # https://stackoverflow.com/a/9951672/8094251
        if isinstance(key, slice):
            return [self[i] for i in range(*key.indices(len(self)))]

        low = self._buffer_count * self.nsamples_buffer
        high = (self._buffer_count + 1) * self.nsamples_buffer

        if low <= key < high:
            return self.inps[key]

        if key >= high:
            self._save()
            self.inps = self._load_next()
        elif key < low:
            # TODO: assuming we start from zero
            assert key == 0

            self._save()
            self._buffer_count = -1
            self.inps = self._load_next()
        else:
            assert False

        return self[key]

    def _init(self):
        return torch.zeros(
            (self.nsamples_buffer, self.seqlen, self.hidden_size),
            dtype=self.dtype, device=self.device
        )

    def _get_file_path(self):
        return os.path.join(self.folder, f'{self._buffer_count}.npy')

    def _save(self):
        os.makedirs(self.folder, exist_ok=True)
        file_path = self._get_file_path()

        with open(file_path, 'wb') as f:
            np.save(f, self.inps)

    def _load_next(self):
        if not os.path.isdir(self.folder):
            print(f'No save folder found for inps "{self.name}"'
                  f' (count {self._buffer_count}).'
                  f' Returning zeros.')

            return self._init()

        self._buffer_count += 1
        file_path = self._get_file_path()

        if not os.path.isfile(file_path):
            print(f'No saved tensor found for inps "{self.name}"'
                  f' (count {self._buffer_count}).'
                  f' Returning zeros.')

            return self._init()

        with open(file_path, 'rb') as f:
            result = np.load(f)

        assert result.dtype == self.dtype, f'{self.dtype}, {result.dtype}'
        assert result.device == self.device, f'{self.device}, {result.device}'

        return result



def omniquant(
    lm,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger=None,
):
    logger.info("Starting ...")
    
    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    if 'llama' in args.model or 'Llama' in args.model:
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            'q_proj':'qkv',
            'o_proj':'out',
            'up_proj':'fc1'
        }
        layer_name_prefix = 'model.layers'
    elif 'opt' in args.model:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            'q_proj':'qkv',
            'out_proj':'out',
            'fc1':'fc1'
        }
        layer_name_prefix = 'model.decoder.layers'
    else:
        raise ValueError("Only support for opt/llama/Llama-2 now")

    
    layers[0] = layers[0].to(dev)
    dtype = next(iter(model.parameters())).dtype
    # inps = torch.zeros(
    #     (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    # )
    inps = Inps(
        name='inps', folder=args.samples_dir,
        nsamples=args.nsamples, seqlen=lm.seqlen,
        hidden_size=model.config.hidden_size,
        dtype=dtype, device=dev
    )
    cache = {"i": 0}

    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama

    print(f'!!! Num batches total: {len(dataloader)}.')

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if 'llama' in args.model or 'Llama' in args.model:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif 'opt' in args.model:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    else:
        raise ValueError("Only support for opt/llama/Llama-2 now")
    torch.cuda.empty_cache()

    
    # same input of first layer for fp model and quant model
    quant_inps = inps
    # fp_inps = copy.deepcopy(inps)   # take output of fp model as input
    # fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None # take output of quantization model as input
    fp_inps = inps.deepcopy(name='fp_inps', folder=args.samples_dir)  # take output of fp model as input
    fp_inps_2 = inps.deepcopy(name='fp_inps_2', folder=args.samples_dir) if args.aug_loss else None  # take output of quantization model as input

    attention_mask = cache["attention_mask"]
    attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1)
    loss_func = torch.nn.MSELoss()
    if is_llama:
        position_ids = cache['position_ids']
    else:
        position_ids = None



    if args.resume:
        omni_parameters = torch.load(args.resume)
    else:
        omni_parameters = {}

    
    
    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)
        qlayer = DecoderLayer(lm.model.config, layer, args)

        
        # obtain output of full-precision model
        qlayer.set_quant_state(weight_quant=False, act_quant=False)
        if args.epochs > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        fp_inps[j] = qlayer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
                        if args.aug_loss:
                            fp_inps_2[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]

        # init smooth parameters
        qlayer.set_quant_state(weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
        qlayer.let = args.let
        use_shift = True 
        # if is_llama and args.abits == 16:
        #     use_shift = False                   # deactivate channel-wise shifting for llama weight-
        # use_shift = True if args.abits < 16 else False   # only activate per-channel shifting when weight-activation quantization
        
        if args.let:
            # init channel-wise scaling and shift
            qlayer.register_parameter("qkt_smooth_scale",torch.nn.Parameter(torch.ones(layer.self_attn.q_proj.out_features,device=dev, dtype=dtype)))
            for name,module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    for key in pairs.keys():
                        if key in name:
                            act = act_scales[f'{layer_name_prefix}.{i}.{name}'].to(device=dev, dtype=dtype).clamp(min=1e-5)
                            weight = module.weight.max(dim=0)[0].clamp(min=1e-5)
                            scale = (act.pow(args.alpha)/weight.pow(1-args.alpha)).clamp(min=1e-5)
                            if use_shift and not is_llama:
                                shift = act_shifts[f'{layer_name_prefix}.{i}.{name}'].to(device=dev, dtype=dtype)
                            else:
                                shift = torch.zeros_like(scale)
                            qlayer.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(shift))
                            qlayer.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(scale))
                                
        if args.resume:
            qlayer.load_state_dict(omni_parameters[i], strict=False)
        
        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()      # required for AMP training
            # create optimizer

            let_params = qlayer.let_parameters(use_shift)
            lwc_params = qlayer.lwc_parameters()
            perturb_params = qlayer.perturb_parameters()

            # print(f'!!! Optimizer LET params: {list(qlayer.let_parameters(use_shift))}.')
            # print(f'!!! Optimizer LWC params: {list(qlayer.lwc_parameters())}.')

            optimizer = torch.optim.AdamW(
                [{'params':let_params,'lr':args.let_lr},
                 {'params':lwc_params,'lr':args.lwc_lr},
                 {'params':perturb_params, 'lr':args.perturb_lr}],
                weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            
            for epochs in range(args.epochs):
                loss_list = []
                norm_list = []
                for j in range(args.nsamples//args.batch_size):    
                    index = j * args.batch_size
                    # obtain output of quantization model
                    with torch.cuda.amp.autocast():
                        qlayer.smooth_and_quant_temporary()
                        quant_out = qlayer(quant_inps[index:index+args.batch_size,], attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                        loss = loss_func(fp_inps[index:index+args.batch_size,], quant_out)
                        if args.aug_loss:
                            loss += loss_func(fp_inps_2[index:index+args.batch_size,], quant_out)
                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                        
                    loss_list.append(loss.data)
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer,parameters=qlayer.omni_parameters(use_shift))
                    norm_list.append(norm.data)

                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
            qlayer.clear_temp_variable()
            del optimizer

        # real smooth and quantization
        qlayer.smooth_and_quant_inplace()
        if args.epochs>0:
            # update input of quantization model
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        quant_inps[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
            qlayer.half()
            layers[i] = qlayer.to("cpu")
            omni_parameters[i] = qlayer.omni_state_dict()
            torch.save(omni_parameters, os.path.join(args.output_dir, f'omni_parameters.pth'))
        else:
            qlayer.half()
            layers[i] = qlayer.to("cpu")

        
        del layer
        torch.cuda.empty_cache()

    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    model.config.use_cache = use_cache
    return model

