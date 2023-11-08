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
        lwc=False,
        adaround=False,
        adaqround=False,
        adaquant=False,
        delta_round=0,
        hard_freq=0,
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
        self.adaround = adaround
        self.adaqround = adaqround
        self.adaquant = adaquant
        
        init_value = 4.             # inti value of learnable weight clipping
        if shape is not None:
            if group_size:
                dim1 = int(shape[0]*math.ceil(shape[1]/group_size))
                dim2 = shape[0] * shape[1] // dim1
                self.deficiency = shape[-1]%group_size
                if self.deficiency > 0:
                    self.deficiency = group_size - self.deficiency
                    assert self.symmetric   # support for mlc-llm symmetric quantization
            else:
                dim1 = shape[0]
                dim2 = shape[1]

            assert dim1 * dim2 == shape[0] * shape[1]

        if lwc:
            self.upbound_factor = nn.Parameter(torch.ones((dim1,1)).cuda()*init_value)
            self.lowbound_factor = nn.Parameter(torch.ones((dim1,1)).cuda()*init_value)

        self.sigmoid = nn.Sigmoid()

        self.enable = True
        self.group_size = group_size

        # AdaRound
        self.delta_round = delta_round
        self.delta_range = 2 * self.delta_round + 1
        self.gamma, self.zeta = -0.1, self.delta_range + 0.1
        self.hard_freq = hard_freq

        if self.adaround:
            assert self.delta_round == 0
            assert self.delta_range == 1
            assert self.hard_freq == 0

        if self.adaround or self.adaqround:
            print('Have alpha for AdaRound weight quantization!')
            self.alpha = torch.nn.Parameter(
                torch.ones(dim1, dim2).cuda()
            )
            self.dweight = None
        elif self.adaquant:
            self.alpha = torch.nn.Parameter(
                torch.zeros(dim1, 1).cuda(),
            )
            self._dw_init_coeff = 0.1
            self.dweight = torch.nn.Parameter(
                torch.zeros(dim1, dim2).cuda()
            )
        else:
            self.alpha = None
            self.dweight = None

        self.num_iters = torch.nn.Parameter(
            torch.tensor([0]), requires_grad=False
        )

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1

    def init_alpha(self, x: torch.Tensor, scale, zero_point):
        scale = scale.data
        x_floor = torch.floor(x / scale)

        rest = (x / scale) - x_floor  # rest of rounding [0, 1)
        alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest

        with torch.no_grad():
            self.alpha.data = alpha

    def rectified_sigmoid(self):
        """generate rounding mask.
        """
        return ((self.zeta - self.gamma) * torch.sigmoid(self.alpha)
                + self.gamma).clamp(0, self.delta_range)

    def round_ada(self, X, scale, zero_point, hard=False):
        if self.num_iters == 0:
            self.init_alpha(X.data.clone().detach(), scale, zero_point)

        scale = scale.data
        X = torch.floor(X / scale) - self.delta_round

        if hard and self.adaround:
            print('!!! Hard AdaRound !!!')
            X += (self.alpha >= 0).float()
        elif hard and self.adaqround:
            print('!!! Hard AdaQRound !!!')
            X += round_ste(self.rectified_sigmoid())
        elif self.adaround or np.random.randint(0, 10) < 10 - self.hard_freq:
            X += self.rectified_sigmoid()
        else:
            X += round_ste(self.rectified_sigmoid())

        return X

    def fake_quant(self, x, scale, round_zero_point, hard):
        if self.deficiency > 0:
            pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
            x = torch.cat((x,pad_zeros),dim=1)
        
        if self.group_size:
            assert len(x.shape)==2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)

        if self.adaround or self.adaqround:
            x_int = self.round_ada(
                x, scale, round_zero_point, hard=hard
            )
        elif self.adaquant:
            x_int = round_ste(
                ((1.0 - self._dw_init_coeff) * x + self.dweight) / scale
            )
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

    def forward(self, x: torch.Tensor, hard=None):
        if self.n_bits >= 16 or not self.enable:
            return x
        if self.metric == "fix0to1":
            return x.mul_(2**self.n_bits-1).round_().div_(2**self.n_bits-1)

        if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel":
            self.per_token_dynamic_calibration(x)
        else:
            raise NotImplementedError()   

        x_dequant = self.fake_quant(
            x, self.scale, self.round_zero_point, hard=hard
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
        xmax =  x.amax(reduce_shape, keepdim=True)

        if self.lwc:
            xmax = self.sigmoid(self.upbound_factor)*xmax
            xmin = self.sigmoid(self.lowbound_factor)*xmin

        if not self.adaquant:
            if self.symmetric:
                abs_max = torch.max(xmax.abs(),xmin.abs())
                scale = abs_max / (2**(self.n_bits-1)-1)
                self.scale = scale.clamp(min=CLIPMIN, max=1e4)
                zero_point = (2**(self.n_bits-1)-1)*torch.ones_like(self.scale)
            else:
                range = xmax - xmin
                scale = range / (2**self.n_bits-1)
                self.scale = scale.clamp(min=CLIPMIN, max=1e4)
                self.scale = scale
                zero_point = -(xmin) / (self.scale)
        elif self.num_iters == 0:
            print(f'!!! Initializing alpha...')

            with torch.no_grad():
                self.perturb.data = self._dw_init_coeff * x

            quantization_errors = list()
            range = xmax - xmin
            alphas = torch.linspace(0.6, 1, steps=100)
            input_abs_total_value = torch.sum(torch.abs(x))

            if input_abs_total_value == 0:
                denominator = 1
            else:
                denominator = input_abs_total_value

            for alpha in alphas:
                scale = alpha * range / (2 ** self.n_bits - 1)
                scale = scale.clamp(min=CLIPMIN, max=1e4)
                zero_point = -(xmin) / (scale)
                round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()

                quant_tensor = self.fake_quant(x, scale, round_zero_point)

                q_error = torch.sum(torch.abs(x - quant_tensor)) / denominator
                quantization_errors.append(q_error.cpu().detach().numpy())

            index_min = np.argmin(quantization_errors)
            best_alpha = alphas[index_min]

            print(f'!!! Best alpha: {best_alpha.item()}.')

            with torch.no_grad():
                self.alpha.data = best_alpha * range

            scale = self.alpha / (2 ** self.n_bits - 1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = -(xmin) / (self.scale)

        self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()
        
    def register_scales_and_zeros(self):
        self.register_buffer('scales', self.scale)
        self.register_buffer('zeros', self.round_zero_point)
        del self.scale
        del self.round_zero_point
