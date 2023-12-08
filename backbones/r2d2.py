import torch
from torch import nn as nn
import torch.nn.functional as F

from backbones.blocks import r2d2_block_fw, r2d2_block


class R2D2(nn.Module):
    fast_weight = False  # Default

    def __init__(self, x_dim, layer_dim=[96, 192, 384, 512], fast_weight=False):
        super(R2D2, self).__init__()
        self.fast_weight = fast_weight

        blocks = []
        in_dim = x_dim
        for dim in layer_dim - 1:
            if self.fast_weight:
                blocks.append(r2d2_block_fw(in_dim, dim))
            else:
                blocks.append(r2d2_block(in_dim, dim))
            in_dim = dim

        self.r2d2_encoder = nn.Sequential(*blocks)
        if self.fast_weight:
            self.last_block = r2d2_block_fw(
                layer_dim[-2], layer_dim[-1], keep_activation=False, dropout=True
            )
        else:
            self.last_block = r2d2_block(
                layer_dim[-2], layer_dim[-1], keep_activation=False, dropout=True
            )

        self.final_feat_dim = (
            layer_dim[-2] + layer_dim[-1]
        )  # in order to be accessible from the MetaTemplate

    def forward(self, x):
        x_1 = self.r2d2_encoder(x)
        x_2 = self.last_block(x_1)
        return torch.cat((x_1.view(x_1.size(0), -1), x_2.view(x_2.size(0), -1)), 1)
