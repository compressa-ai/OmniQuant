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

        self.history_length = 20
        self.alpha = 0.4
        self.is_started_eval = nn.Parameter(
            torch.Tensor([0]), requires_grad=False
        )
        self.num_iters = nn.Parameter(
            torch.Tensor([0]), requires_grad=False
        )
        self.xmin = None
        self.xmax = None
        self.xmins = nn.Parameter(
            torch.Tensor([0] * self.history_length).cuda(), requires_grad=False
        )
        self.xmaxs = nn.Parameter(
            torch.Tensor([0] * self.history_length).cuda(), requires_grad=False
        )

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

        if self.lwc:  # Weight
            if self.num_iters % 100 == 0:
                # print(f'!!! I am alive and I do not update stats')
                print(f'Up: {float(self.upbound_factor[0])}. Low: {float(self.lowbound_factor[0])}.')

            xmin = x.amin(reduce_shape, keepdim=True)
            xmax = x.amax(reduce_shape, keepdim=True)

            self.num_iters += 1
        elif self.num_iters < self.history_length:
            # xmin = x.amin(reduce_shape, keepdim=True)
            # xmax = x.amax(reduce_shape, keepdim=True)
            xmin = x.min()
            xmax = x.max()

            self.xmins[int(self.num_iters)] = float(x.min())
            self.xmaxs[int(self.num_iters)] = float(x.max())

            # num_iters = int(self.num_iters)

            # with torch.no_grad():
            #     self.xmin = (num_iters * self.xmin + xmin) / (num_iters + 1)
            #     self.xmax = (num_iters * self.xmax + xmax) / (num_iters + 1)

            self.num_iters += 1
        elif self.num_iters == self.history_length:
            # assert self.num_iters >= self.max_num_iters

            self.xmin = torch.nn.Parameter(
                torch.mean(self.xmins)
            )
            self.xmax = torch.nn.Parameter(
                torch.mean(self.xmaxs)
            )

            xmin = self.xmin
            xmax = self.xmax

            self.num_iters += 1
        elif self.training:
            if self.num_iters % 100 == 0:
                # print(f'!!! I am alive and I do not update stats')
                print(f'X_min: {float(self.xmin)}. X_max: {float(self.xmax)}.')

            xmin = self.xmin
            xmax = self.xmax

            self.num_iters += 1
        elif not self.training:
            if self.is_started_eval == 0:
                print('!!! EVALUATING !!!')

                self.is_started_eval += 1

            xmin = self.xmin
            xmax = self.xmax
        else:
            print(f'!!! WTF !!!')

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