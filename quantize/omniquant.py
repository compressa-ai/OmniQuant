import torch
import torch.nn as nn
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from models.int_falcon_layer import QuantFalconDecoderLayer
from quantize.int_linear import QuantLinear
import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
import gc


from datautils import Inps



def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}


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
    if "llama" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1"
        }
        layer_name_prefix = "model.layers"
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "out_proj":"out",
            "fc1":"fc1"
        }
        layer_name_prefix = "model.decoder.layers"
    elif "falcon" in args.net.lower():
        layers = model.transformer.h
        model.transformer.word_embeddings.to(dev)
        model.transformer.ln_f.to(dev)
        model.lm_head.to(dev)
        DecoderLayer = QuantFalconDecoderLayer
        layer_name_prefix = "model.transformer.h"
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon now")
    
    
    layers[0] = layers[0].to(dev)
    if args.deactive_amp and args.epochs>0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = torch.cuda.amp.autocast

    if not args.keep_samples_on_disk:
        inps_dev = dev if not args.keep_samples_on_cpu else 'cpu'
        inps = torch.zeros(
            (args.nsamples, lm.seqlen, model.config.hidden_size),
            dtype=dtype, device=inps_dev
        )
    else:
        inps_dev = dev
        inps = Inps(
            name='inps', folder=args.samples_dir,
            nsamples=args.nsamples, seqlen=lm.seqlen,
            hidden_size=model.config.hidden_size,
            dtype=dtype, device=inps_dev
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
                cache["position_ids"] = kwargs["position_ids"]
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
    if "llama" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "opt" in args.net.lower():
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif 'falcon' in args.model:
        model.transformer.word_embeddings =  model.transformer.word_embeddings.cpu()
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon now")
    torch.cuda.empty_cache()

    
    # same input of first layer for fp model and quant model
    quant_inps = inps

    if not args.keep_samples_on_disk:
        fp_inps = copy.deepcopy(inps) if not args.no_ord_loss else None  # take output of fp model as input
        fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None  # take output of quantization model as input
    else:
        fp_inps = inps.deepcopy(
            name='fp_inps', folder=args.samples_dir
        ) if not args.no_ord_loss else None  # take output of fp model as input

        fp_inps_2 = inps.deepcopy(
            name='fp_inps_2', folder=args.samples_dir
        ) if args.aug_loss else None  # take output of quantization model as input

    attention_mask = cache["attention_mask"]
    attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1) if args.deactive_amp else attention_mask.repeat(args.batch_size,1,1,1).float()
    loss_func = torch.nn.MSELoss()
    if is_llama:
        position_ids = cache["position_ids"]
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
                        if not args.no_ord_loss:
                            fp_inps[j] = qlayer(
                                fp_inps[j].unsqueeze(0).to(dev),
                                attention_mask=attention_mask,
                                position_ids=position_ids
                            )[0].to(inps_dev)

                        if args.aug_loss:
                            fp_inps_2[j] = qlayer(
                                quant_inps[j].unsqueeze(0).to(dev),
                                attention_mask=attention_mask,
                                position_ids=position_ids
                            )[0].to(inps_dev)

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
                            act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype).clamp(min=1e-5)
                            weight = module.weight.max(dim=0)[0].clamp(min=1e-5)
                            scale = (act.pow(args.alpha)/weight.pow(1-args.alpha)).clamp(min=1e-5)
                            if use_shift and not is_llama:
                                shift = act_shifts[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype)
                            else:
                                shift = torch.zeros_like(scale)
                            qlayer.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(shift))
                            qlayer.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(scale))
                                
        if args.resume:
            print(f'!!! Loading quant params: {omni_parameters[i].keys()}.')

            qlayer.load_state_dict(omni_parameters[i], strict=False)
        

        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()      # required for AMP training

            let_params = qlayer.let_parameters(use_shift)
            lwc_params = qlayer.lwc_parameters()
            alpha_params = qlayer.alpha_parameters()
            dweight_params = qlayer.dweight_parameters()

            params = list()

            if args.let:
                params.append({"params": lwc_params, "lr": args.lwc_lr})
            if args.lwc:
                params.append({"params": lwc_params, "lr": args.lwc_lr})

            if args.adaround or args.adaqround:
                params.append({"params": alpha_params, "lr": args.alpha_lr})
            elif args.adaquant:
                params.append({"params": alpha_params, "lr": args.alpha_lr})
                params.append({"params": dweight_params, "lr": args.dweight_lr})

            # create optimizer
            optimizer = torch.optim.AdamW(params, weight_decay=args.wd)

            if args.use_lr_scheduler:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs
                )
            else:
                scheduler = torch.optim.lr_scheduler.StepLR(
                    step_size=10 ** 9, gamma=1.0
                )

            loss_scaler = utils.NativeScalerWithGradNormCount()
            
            for epochs in range(args.epochs):
                print(f'!!! LR before epoch: {scheduler.get_last_lr()}.')

                loss_list = []
                norm_list = []
                for j in range(args.nsamples//args.batch_size):    
                    index = j * args.batch_size
                    # obtain output of quantization model
                    with traincast():
                        qlayer.smooth_and_quant_temporary()
                        quant_out = qlayer(quant_inps[index:index + args.batch_size].to(dev),
                                           attention_mask=attention_mask_batch,
                                           position_ids=position_ids)[0]
                        loss1 = loss2 = 0
                        if not args.no_ord_loss:
                            loss1 = loss_func(fp_inps[index:index + args.batch_size].to(dev), quant_out)
                        if args.aug_loss:
                            loss2 = loss_func(fp_inps_2[index:index + args.batch_size].to(dev), quant_out)
                        loss = loss1 + loss2

                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                        
                    loss_list.append(loss.data)
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer,parameters=qlayer.omni_parameters(use_shift))
                    norm_list.append(norm.data)

                scheduler.step()

                print(f'!!! LR after epoch: {scheduler.get_last_lr()}.')

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
                        quant_inps[j] = qlayer(
                            quant_inps[j].unsqueeze(0).to(dev),
                            attention_mask=attention_mask,
                            position_ids=position_ids
                        )[0].to(inps_dev)

            qlayer.register_scales_and_zeros()
            qlayer.half()
            layers[i] = qlayer.to("cpu")

            if not args.no_save_params:
                print(f'Saving params: {i}.')

                omni_parameters[i] = qlayer.omni_state_dict()
                torch.save(omni_parameters, os.path.join(args.output_dir, f'omni_parameters.pth'))
        else:
            qlayer.register_scales_and_zeros()
            qlayer.half()
            layers[i] = qlayer.to("cpu")

        if args.real_quant:
            named_linears = get_named_linears(qlayer)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scales
                zeros = module.weight_quantizer.zeros
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1)
                zeros = zeros.view(dim0,-1)
                q_linear = qlinear_cuda.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                q_linear.pack(module.float().cpu(),  scales.float().cpu(), zeros.float().cpu())
                
                levels = name.split('.')
                if len(levels) > 1:
                    mod_ = qlayer
                    for l_idx in range(len(levels)-1):
                        if levels[l_idx].isdigit():
                            mod_ = mod_[int(levels[l_idx])]
                        else:
                            mod_ = getattr(mod_, levels[l_idx])
                    setattr(mod_, levels[-1], q_linear)
                else:
                    setattr(qlayer, name, q_linear)        
                del module        
        del layer
        torch.cuda.empty_cache()

    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model

