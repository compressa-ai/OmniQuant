import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import tqdm
import numpy as np
import pdb
import math

CLIPMIN = 1e-5




def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x



class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        per_channel_axes=[],
        metric="minmax",
        dynamic=False,
        dynamic_method="per_cluster",
        group_size=None,
        shape=None,
        lwc=False
    ):
        """
        support cluster quantize
        dynamic_method support per_token and per_cluster
        """
        super().__init__()
        self.symmetric = symmetric
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1
        self.per_channel_axes = per_channel_axes
        self.metric = metric
        self.cluster_counts = None
        self.cluster_dim = None

        self.scale = None
        self.zero_point = None
        self.round_zero_point = None

        self.cached_xmin = None
        self.cached_xmax = None
        self.dynamic = dynamic
        self.dynamic_method = dynamic_method
        self.deficiency = 0
        self.lwc = lwc
        
        init_value = 4.             # inti value of learnable weight clipping
        if lwc:
            if group_size:
                dim1 = int(shape[0]*math.ceil(shape[1]/group_size))
                self.deficiency = shape[-1]%group_size
                if self.deficiency > 0:
                    self.deficiency = group_size - self.deficiency
                    assert self.symmetric   # support for mlc-llm quantization
            else:
                dim1 = shape[0]
            self.upbound_factor = nn.Parameter(torch.ones(dim1,1).cuda()*init_value)
            self.lowbound_factor = nn.Parameter(torch.ones(dim1,1).cuda()*init_value)
        self.sigmoid = nn.Sigmoid()

        self.enable = True
        self.group_size = group_size


        # AdaRound
        self.ch_axis = -1
        self.drop_prob = 1.0

        # self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        # self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
        self.gamma, self.zeta = -0.1, 1.1
        self.round_mode = 'learned_hard_sigmoid'

        if shape is not None:
            print('Have alpha!')
            self.alpha = torch.nn.Parameter(torch.ones(*shape).cuda())
            self.alpha_scale = torch.nn.Parameter(torch.ones(*shape).cuda())
            self.alpha_zero_point = torch.nn.Parameter(torch.ones(*shape).cuda())
        else:
            print('No alpha!')
            self.alpha = None

        self.num_iters = torch.nn.Parameter(
            torch.tensor([0]), requires_grad=False
        )

    def init_alpha(self, x: torch.Tensor, scale, zero_point):
        # if self.ch_axis != -1:
        #     new_shape = [1] * len(x.shape)
        #     new_shape[self.ch_axis] = x.shape[self.ch_axis]
        #     scale = scale.data.reshape(new_shape)
        # else:
        #     scale = scale.data
        scale = scale.data

        x_floor = torch.floor(x / scale)
        if self.round_mode == 'learned_hard_sigmoid':
            rest = (x / scale) - x_floor  # rest of rounding [0, 1)
            alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
            with torch.no_grad():
                self.alpha.data = alpha
                self.alpha_scale.data = scale
                self.alpha_zero_point.data = zero_point
        else:
            raise NotImplementedError

    def rectified_sigmoid(self):
        """generate rounding mask.
        """
        return ((self.zeta - self.gamma) * torch.sigmoid(self.alpha) + self.gamma).clamp(0, 1)

    def adaround_forward(self, X, scale, zero_point, hard_value=False):
        if self.num_iters == 0:
            # print(f'!!! INIT ALPHA')
            self.init_alpha(X.data.clone().detach(), scale, zero_point)

        # print(f'!!! ADAROUND')

        # if self.ch_axis != -1:
        #     new_shape = [1] * len(X.shape)
        #     new_shape[self.ch_axis] = X.shape[self.ch_axis]
        #     scale = scale.data.reshape(new_shape)
        #     zero_point = zero_point.data.int().reshape(new_shape)
        # else:
        #     scale = scale.item()
        #     zero_point = zero_point.item()
        scale = self.alpha_scale
        zero_point = self.alpha_zero_point

        X = torch.floor(X / scale)
        if hard_value:
            X += (self.alpha >= 0).float()
        else:
            X += self.rectified_sigmoid()
        X += zero_point
        X = torch.clamp(X, self.qmin, self.qmax)
        X = (X - zero_point) * scale
        return X

    # def get_hard_value(self, X):
    #     X = self.adaround_forward(X, hard_value=True)
    #     return X

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1

    def fake_quant(self, x, scale, round_zero_point):
        if self.deficiency > 0:
            pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
            x = torch.cat((x,pad_zeros),dim=1)
        
        if self.group_size:
            assert len(x.shape)==2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)

        if self.alpha is not None:
            x_dequant = self.adaround_forward(x, scale, round_zero_point)
        else:
            x_int = round_ste(x / scale)
            if round_zero_point is not None:
                x_int = x_int.add(round_zero_point)
            x_int = x_int.clamp(self.qmin, self.qmax)
            x_dequant = x_int
            if round_zero_point is not None:
                x_dequant = x_dequant.sub(round_zero_point)
            x_dequant = x_dequant.mul(scale)

        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        if self.deficiency > 0:
            x_dequant = x_dequant[:,:-self.deficiency]
        return x_dequant
    

    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x
        if self.metric == "fix0to1":
            return x.mul_(2**self.n_bits-1).round_().div_(2**self.n_bits-1)

        if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel":
            self.per_token_dynamic_calibration(x)
        else:
            raise NotImplementedError()   

        x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
        self.num_iters += 1

        return x_dequant

    def per_token_dynamic_calibration(self, x):
        if self.group_size:
            if self.deficiency == 0:
                x = x.reshape(-1,self.group_size)
            else:
                pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
                x = torch.cat((x,pad_zeros),dim=1)
                x = x.reshape(-1,self.group_size)
        reduce_shape = [-1]
        xmin = x.amin(reduce_shape, keepdim=True)
        xmax =  x.amax(reduce_shape, keepdim=True)

        if self.lwc:
            xmax = self.sigmoid(self.upbound_factor)*xmax
            xmin = self.sigmoid(self.lowbound_factor)*xmin

        if self.symmetric:
            abs_max = torch.max(xmax.abs(),xmin.abs())
            scale = abs_max / (2**(self.n_bits-1)-1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = (2**(self.n_bits-1)-1)*torch.ones_like(self.scale)
        else:
            range = xmax - xmin
            scale = range / (2**self.n_bits-1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = -(xmin) / (self.scale)

        self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()




def fake_quantize_per_tensor_affine(x, scale, zero_point, quant_min, quant_max):
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


def fake_quantize_per_channel_affine(x, scale, zero_point, ch_axis, quant_min, quant_max):
    new_shape = [1] * len(x.shape)
    new_shape[ch_axis] = x.shape[ch_axis]
    scale = scale.reshape(new_shape)
    zero_point = zero_point.reshape(new_shape)
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


class AdaRoundFakeQuantize(nn.Module):
    """
    self.adaround=True: turn on up or down forward
    self.adaround=False: turn on round-to-nearest forward
    based on the FixedFakeQuantize
    """

    # def __init__(self, bit=8, symmetric=False, ch_axis=-1):
    #     super().__init__()
    #
    #     self.bit = bit
    #     self.symmetric = symmetric
    #     self.ch_axis = ch_axis
    #     self.observer_enabled = 0
    #     self.fake_quant_enabled = 0
    #     self.quant_min = self.observer.quant_min
    #     self.quant_max = self.observer.quant_max
    #     self.drop_prob = 1.0
    #
    #     self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
    #     self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
    #     self.adaround = False
    #     self.gamma, self.zeta = -0.1, 1.1
    #
    #     self.init()

    # def init(self, weight_tensor: torch.Tensor):
    #     self.adaround = True
    #     self.round_mode = 'learned_hard_sigmoid'
    #     self.init_alpha(x=weight_tensor.data.clone().detach())

    # def init_alpha(self, x: torch.Tensor):
    #     if self.ch_axis != -1:
    #         new_shape = [1] * len(x.shape)
    #         new_shape[self.ch_axis] = x.shape[self.ch_axis]
    #         scale = self.scale.data.reshape(new_shape)
    #     else:
    #         scale = self.scale.data
    #     x_floor = torch.floor(x / scale)
    #     if self.round_mode == 'learned_hard_sigmoid':
    #         rest = (x / scale) - x_floor  # rest of rounding [0, 1)
    #         alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
    #         self.alpha = torch.nn.Parameter(alpha)
    #     else:
    #         raise NotImplementedError

    # def rectified_sigmoid(self):
    #     """generate rounding mask.
    #     """
    #     return ((self.zeta - self.gamma) * torch.sigmoid(self.alpha) + self.gamma).clamp(0, 1)
    #
    # def adaround_forward(self, X, hard_value=False):
    #     if self.ch_axis != -1:
    #         new_shape = [1] * len(X.shape)
    #         new_shape[self.ch_axis] = X.shape[self.ch_axis]
    #         scale = self.scale.data.reshape(new_shape)
    #         zero_point = self.zero_point.data.int().reshape(new_shape)
    #     else:
    #         scale = self.scale.item()
    #         zero_point = self.zero_point.item()
    #     X = torch.floor(X / scale)
    #     if hard_value:
    #         X += (self.alpha >= 0).float()
    #     else:
    #         X += self.rectified_sigmoid()
    #     X += zero_point
    #     X = torch.clamp(X, self.quant_min, self.quant_max)
    #     X = (X - zero_point) * scale
    #     return X
    #
    # def get_hard_value(self, X):
    #     X = self.adaround_forward(X, hard_value=True)
    #     return X

    def forward(self, X):
        if self.observer_enabled == 1:
            self.observer(X.detach())
            _scale, _zero_point = self.observer.calculate_qparams(self.observer.min_val, self.observer.max_val)
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled == 1:
            if not self.adaround:
                if self.ch_axis != -1:
                    X = fake_quantize_per_channel_affine(
                        X, self.scale.data, self.zero_point.data.int(), self.ch_axis,
                        self.quant_min, self.quant_max)
                else:
                    X = fake_quantize_per_tensor_affine(
                        X, self.scale.item(), self.zero_point.item(),
                        self.quant_min, self.quant_max)
            else:
                if not hasattr(self, 'alpha'):
                    raise NotImplementedError
                if self.round_mode == 'learned_hard_sigmoid':
                    X = self.adaround_forward(X)
                else:
                    raise NotImplementedError
        return X

