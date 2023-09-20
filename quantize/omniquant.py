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
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
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
    fp_inps = copy.deepcopy(inps)   # take output of fp model as input
    fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None # take output of quantization model as input
    
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


    if args.run_awq:
        assert args.dump_awq, "Please save the awq results with --dump_awq"

        enc = lm.tokenizer
        q_config_awq = {
            "zero_point": not args.no_zero_point,  # by default True
            "q_group_size": args.group_size,  # whether to use group quantization
        }
        w_bit = args.wbits
        n_samples = 128
        seqlen = 512
        auto_scale = True
        mse_range = True

        # layers = get_blocks(model)

        samples = get_calib_dataset(
            data="pileval", tokenizer=enc,
            n_samples=n_samples, block_size=seqlen  # TODO: parametrize?
        )
        samples = torch.cat(samples, dim=0)

        inps_awq = []
        layer_kwargs = {}

        layers[0] = layers[0].cuda()
        move_embed(model, "cuda")

        # get input and kwargs to layer 0
        # with_kwargs is only supported in PyTorch 2.0
        # use this Catcher hack for now
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps_awq.append(inp)
                layer_kwargs.update(kwargs)
                raise ValueError  # early exit to break later inference

        # patch layer 0 to catch input and kwargs
        layers[0] = Catcher(layers[0])
        try:
            model(samples.to(next(model.parameters()).device))
        except ValueError:  # work with early exit
            pass
        del samples
        layers[0] = layers[0].module  # restore
        inps_awq = inps_awq[0]

        layers[0] = layers[0].cpu()
        move_embed(model, "cpu")

        gc.collect()
        torch.cuda.empty_cache()

        awq_results = {
            "scale": [],
            "clip": [],
        }


    
    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")


        if args.run_awq:
            logger.info(f"== AWQ for layer {i} ==")

            """
            awq_results = run_awq(
                model=model, enc=enc,
                w_bit=args.wbits, q_config=q_config_awq,
                n_samples=128, seqlen=512,  # TODO: params?
            )
            """

            layer = layers[i]
            layer = layer.cuda()
            named_linears = get_named_linears(layer)

            # firstly, get input features of all linear layers
            def cache_input_hook(m, x, y, name, feat_dict):
                x = x[0]
                x = x.detach().cpu()
                feat_dict[name].append(x)

            input_feat = defaultdict(list)
            handles = []
            for name in named_linears:
                handles.append(named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name,
                                      feat_dict=input_feat)))
            inps_awq = inps_awq.to(next(layer.parameters()).device)  # in case multi-gpu
            # get output as next layer's input
            inps_awq = layer(inps_awq, **layer_kwargs)[0]
            for h in handles:
                h.remove()
            # now solve for scaling and clipping
            input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

            # Clear GPU memory
            torch.cuda.empty_cache()

            if auto_scale:  # if it applies, we should also modify the input_feat with scales
                scales_list = auto_scale_block(
                    layer, layer_kwargs,
                    w_bit=w_bit, q_config=q_config_awq,
                    input_feat=input_feat,
                )
                # apply_scale(layer, scales_list, input_feat_dict=input_feat)
                apply_scale(layers[i], scales_list, input_feat_dict=input_feat)
                # append prefix to make names global
                awq_results["scale"] += append_str_prefix(scales_list, get_op_name(model, layer) + ".")

            # Clear GPU memory
            torch.cuda.empty_cache()

            if mse_range:
                clip_list = auto_clip_block(
                    layer,
                    w_bit=w_bit, q_config=q_config_awq,
                    input_feat=input_feat, )
                apply_clip(layer, clip_list)
                # append prefix to make names global
                awq_results["clip"] += append_str_prefix(clip_list, get_op_name(model, layer) + ".")

            layer = layer.cpu()
            # Haotian: check activation replacement
            del input_feat
            gc.collect()
            torch.cuda.empty_cache()



        logger.info(f"== OmniQuant for layer {i} ==")

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
            optimizer = torch.optim.AdamW(
                [{'params':qlayer.let_parameters(use_shift),'lr':args.let_lr}, {'params':qlayer.lwc_parameters(),'lr':args.lwc_lr}],weight_decay=args.wd)
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

    if args.dump_awq:
        dirpath = os.path.dirname(args.dump_awq)
        os.makedirs(dirpath, exist_ok=True)

        torch.save(awq_results, args.dump_awq)
        print("AWQ results saved at", args.dump_awq)

    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    model.config.use_cache = use_cache
    return model




import gc
import functools
from collections import defaultdict

from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from awq.quantize.auto_scale import auto_scale_block, apply_scale
from awq.quantize.auto_clip import auto_clip_block, apply_clip
from awq.utils.calib_data import get_calib_dataset
from awq.utils.module import append_str_prefix, get_op_name


@torch.no_grad()
def run_awq(
        model, enc,
        w_bit, q_config,
        n_samples=512, seqlen=512,
        auto_scale=True, mse_range=True,
        # some configs for ablation study
        calib_data="pileval",  # TODO: parametrize?
        ):
    layers = get_blocks(model)

    samples = get_calib_dataset(
        data=calib_data, tokenizer=enc, n_samples=n_samples, block_size=seqlen)
    samples = torch.cat(samples, dim=0)

    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].cuda()
    move_embed(model, "cuda")

    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])
    try:
        model(samples.to(next(model.parameters()).device))
    except ValueError:  # work with early exit
        pass
    del samples
    layers[0] = layers[0].module  # restore
    inps = inps[0]

    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")

    gc.collect()
    torch.cuda.empty_cache()

    awq_results = {
        "scale": [],
        "clip": [],
    }

    # solve layer by layer
    for i in tqdm.tqdm(range(len(layers)), desc="Running AWQ..."):
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(named_linears[name].register_forward_hook(
                functools.partial(cache_input_hook, name=name,
                                  feat_dict=input_feat)))
        inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        inps = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        # Clear GPU memory
        torch.cuda.empty_cache()

        if auto_scale:  # if it applies, we should also modify the input_feat with scales
            scales_list = auto_scale_block(
                layer, layer_kwargs,
                w_bit=w_bit, q_config=q_config,
                input_feat=input_feat,
            )
            # apply_scale(layer, scales_list, input_feat_dict=input_feat)
            apply_scale(layers[i], scales_list, input_feat_dict=input_feat)
            # append prefix to make names global
            awq_results["scale"] += append_str_prefix(scales_list, get_op_name(model, layer) + ".")

        # Clear GPU memory
        torch.cuda.empty_cache()

        if mse_range:
            clip_list = auto_clip_block(layer,
                                        w_bit=w_bit, q_config=q_config,
                                        input_feat=input_feat, )
            apply_clip(layer, clip_list)
            # append prefix to make names global
            awq_results["clip"] += append_str_prefix(clip_list, get_op_name(model, layer) + ".")

        layer = layer.cpu()
        # Haotian: check activation replacement
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()

    return awq_results


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_blocks(model):
    if isinstance(model, LlamaForCausalLM):
        layers = model.model.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif "mpt" in str(model.__class__).lower():
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        layers = model.transformer.h
    else:
        raise NotImplementedError(type(model))
    return layers


def move_embed(model, device):
    if isinstance(model, LlamaForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    elif isinstance(model, OPTForCausalLM):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
    elif isinstance(model, BloomForCausalLM):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
        model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(device)
    elif "mpt" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.emb_drop = model.transformer.emb_drop.to(device)
    elif "falcon" in str(model.__class__).lower():
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
    else:
        raise NotImplementedError(type(model))
