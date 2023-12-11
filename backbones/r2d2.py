    import torch
from torch import nn as nn
import torch.nn.functional as F

from backbones.blocks import r2d2_block_fw, r2d2_block

def find_factors(number):
    factors = []
    for i in range(1, int(number**0.5) + 1):
        if number % i == 0:
            factors.append((i, number // i))
    return factors[-1]
    
class R2D2_2(nn.Module):
    fast_weight = False  # Default

    def __init__(self, x_dim=1, layer_dim=[96, 192, 384, 512], fast_weight=False):
        super(R2D2_2, self).__init__()
        self.fast_weight = fast_weight

        blocks = []
        in_dim = x_dim
        for dim in range(len(layer_dim)-1):
            if self.fast_weight:
                blocks.append(r2d2_block_fw(in_dim, layer_dim[dim]))
            else:
                blocks.append(r2d2_block(in_dim, layer_dim[dim]))
            in_dim = layer_dim[dim]

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

        height , width = find_factors(x.shape[1]) # reshaping the features into a 2D image with dimensions (factor1, factor2) based on the factor pairs of the original feature count
        x = x.reshape(x.shape[0],1, height, width) #batch_size,channels,height,width

        x_1 = self.r2d2_encoder(x)
        x_2 = self.last_block(x_1)

        #output shape is (384*H_out*W_out) from the decoder + (512*H_out*W_out) from the last block
        return torch.cat((x_1.view(x_1.size(0), -1), x_2.view(x_2.size(0), -1)), 1)
