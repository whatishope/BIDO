import torch
import torch.nn as nn
import torch.nn.functional as F

from models.glow import AIO_GlowCouplingBlock


class SmallGlowFlow(nn.Module):


    def __init__(self, dim=512, hidden_dim=512, n_blocks=2,
                 clamp=2., act_norm=1., act_norm_type='SOFTPLUS', permute_soft=True):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks

        def subnet_constructor(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, hidden_dim, kernel_size=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, c_out, kernel_size=1, padding=0, bias=True),
            )

        self.blocks = nn.ModuleList([
            AIO_GlowCouplingBlock(
                dims_in=[(dim, 1, 1)],
                subnet_constructor=subnet_constructor,
                clamp=clamp,
                act_norm=act_norm,
                act_norm_type=act_norm_type,
                permute_soft=permute_soft
            )
            for _ in range(n_blocks)
        ])

    def forward(self, h, rev=False):
        B, D = h.shape
        assert D == self.dim, f'Expected dim={self.dim}, got {D}'

        x = h.view(B, D, 1, 1)
        log_det_total = 0.0

        if not rev:
            for block in self.blocks:
                x = block([x], rev=False)[0]
                log_det_total = log_det_total + block.last_jac  # [B]
        else:
            for block in reversed(self.blocks):
                x = block([x], rev=True)[0]
                log_det_total = log_det_total + block.last_jac

        z = x.view(B, D)
        return z, log_det_total
