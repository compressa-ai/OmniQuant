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

        # self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        # self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))

        if shape is not None:
            print('Have alpha!')

            if group_size:
                dim1 = int(shape[0]*math.ceil(shape[1]/group_size))
                self.deficiency = shape[-1]%group_size
                if self.deficiency > 0:
                    self.deficiency = group_size - self.deficiency
                    assert self.symmetric   # support for mlc-llm quantization
            else:
                dim1 = shape[0]

            self.alpha = torch.nn.Parameter(
                torch.zeros(dim1, 1).cuda(),
            )
            self.shift = torch.nn.Parameter(
                torch.zeros(dim1, 1).cuda(),
            )
            self._perturb_coeff = 1e-5
            self.perturb = torch.nn.Parameter(
                torch.zeros(dim1, shape[0] * shape[1] // dim1).cuda()
            )
        else:
            print('No alpha!')
            self.alpha = None
            self.shift = None
            self._perturb_coeff = None
            self.perturb = None

        self.num_iters = torch.nn.Parameter(
            torch.tensor([0]), requires_grad=False
        )

    def fake_quant(self, x, scale, round_zero_point):
        if self.deficiency > 0:
            pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
            x = torch.cat((x,pad_zeros),dim=1)
        
        if self.group_size:
            assert len(x.shape)==2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)

        x_int = round_ste(
            (x + self.perturb) / scale
        )

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

        x_dequant = self.fake_quant(
            x, self.scale, self.round_zero_point,
        )
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
        xmax = x.amax(reduce_shape, keepdim=True)

        # if self.lwc:
        #     xmax = self.sigmoid(self.upbound_factor)*xmax
        #     xmin = self.sigmoid(self.lowbound_factor)*xmin

        if self.num_iters == 0:
            print(f'!!! Initializing alpha...')

            import scipy.optimize as opt

            def layer_err(p):
                range = xmax - xmin
                input_abs_total_value = torch.sum(torch.abs(x))

                if input_abs_total_value == 0:
                    denominator = 1
                else:
                    denominator = input_abs_total_value

                scale = p[0] * range / (2 ** self.n_bits - 1)
                scale = scale.clamp(min=CLIPMIN, max=1e4)
                zero_point = -(xmin + p[1]) / (scale)
                round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()

                quant_tensor = self.fake_quant(x, scale, round_zero_point)
                q_error = torch.sum(torch.abs(x - quant_tensor)) / denominator

                return q_error.cpu().detach().numpy()

            init = np.array([1, 0])
            res = opt.minimize(lambda p: layer_err(p), init, method='Nelder-Mead')

            print(f'!!! Best alpha: {res.x[0]}, best shift: {res.x[1]}.')

            with torch.no_grad():
                range = xmax - xmin
                self.alpha.data = res.x[0] * range
                self.shift.data = res.x[1] * self.shift.data
                self.perturb.data = (
                    self._perturb_coeff * torch.randn(*self.perturb.shape).cuda()
                )

        scale = self.alpha / (2**self.n_bits-1)
        self.scale = scale.clamp(min=CLIPMIN, max=1e4)
        zero_point = -(xmin + self.shift) / (self.scale)
        self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()

        # if self.symmetric:
        #     abs_max = torch.max(xmax.abs(),xmin.abs())
        #     scale = abs_max / (2**(self.n_bits-1)-1)
        #     self.scale = scale.clamp(min=CLIPMIN, max=1e4)
        #     zero_point = (2**(self.n_bits-1)-1)*torch.ones_like(self.scale)
        # else:
        #     range = xmax - xmin
        #     scale = range / (2**self.n_bits-1)
        #     self.scale = scale.clamp(min=CLIPMIN, max=1e4)
        #     zero_point = -(xmin) / (self.scale)
        # self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()
