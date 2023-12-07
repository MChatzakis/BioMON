import torch
from torch import nn as nn
import torch.nn.functional as F


def r2d2_block(in_features, out_features, keep_activation=True, dropout=True):

    r2d2_layers = [ 
        nn.Conv2d(in_features, out_features,kernel_size=3,padding=1),
        nn.BatchNorm2d(out_features),
        nn.MaxPool2d(2)
    ]
    if keep_activation:
      r2d2_layers.append(nn.LeakyReLU(0.1))
    
    if dropout:
      r2d2_layers.append(nn.Dropout(0.3))

    return nn.Sequentila(*r2d2_layers)


def r2d2_block_fw(in_features, out_features, keep_activation=True, dropout=True):

    r2d2_layers = [ 
        nn.Conv2d_fw(in_features, out_features,kernel_size=3,padding=1),
        nn.BatchNorm2d_fw(out_features),
        nn.MaxPool2d(2)
    ]
    if keep_activation:
      r2d2_layers.append(nn.LeakyReLU(0.1))
    
    if dropout:
      r2d2_layers.append(nn.Dropout(0.3))

    return nn.Sequentila(*r2d2_layers)

    
class Conv2d_fw(nn.Conv2d): # used in R2D2 to forward input with fast weight
    def __init__(self, in_channels, out_channels):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.conv2d(x, self.weight.fast, self.bias.fast, padding=1)  # weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Conv2d_fw, self).forward(x)
        return out
    
class BatchNorm2d_fw(nn.BatchNorm2d):  # used in R2D2 to forward input with fast weight #same as BatchNorm1d_fw
    def __init__(self, num_features): 
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        #if you set training=True then batch_norm computes and uses the appropriate normalization statistics for the batch (this means we don't need to calculate the mean and std ourselves)
        running_mean = torch.zeros(x.data.size()[1]).cuda() 
        running_var = torch.ones(x.data.size()[1]).cuda()

        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(x, running_mean, running_var, self.weight.fast, self.bias.fast, training=True,
                               momentum=1)
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training=True, momentum=1)
        return out

class R2D2(nn.Module):
    fast_weight = False  # Default

    def __init__(self, x_dim, layer_dim=[96,192,384,512], fast_weight=False):
        
        super(R2D2, self).__init__()
        self.fast_weight = fast_weight

        blocks = []
        in_dim = x_dim
        for  dim in layer_dim-1:
            if self.fast_weight:
                blocks.append(r2d2_block_fw(in_dim, dim))
            else:
                blocks.append(r2d2_block(in_dim, dim))
            in_dim = dim

        self.r2d2_encoder = nn.Sequential(*blocks)
        if self.fast_weight:
            self.last_block = r2d2_block_fw(layer_dim[-2], layer_dim[-1],keep_activation=False,dropout=True)
        else:
            self.last_block = r2d2_block(layer_dim[-2], layer_dim[-1],keep_activation=False,dropout=True)
        
        self.final_feat_dim = layer_dim[-2]+layer_dim[-1] #in order to be accessible from the MetaTemplate

    def forward(self, x):
        x_1=self.r2d2_encoder(x)
        x_2 = self.last_block(x_1)
        return torch.cat((x_1.view(x_1.size(0), -1), x_2.view(x_2.size(0), -1)), 1)
