import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from methods.meta_template import MetaTemplate


class BioMetaOptNet(MetaTemplate):
    def __init__(self, backbone, head_model_params, n_way, n_support):
        super(BioMetaOptNet, self).__init__(backbone, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.feature = backbone
        self.n_way = n_way
        self.n_support = n_support
        self.head_model_params = head_model_params

    def initialize_model(self):
        raise NotImplementedError
    
    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)

        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)
        z_support = z_support.contiguous().view(self.n_way * self.n_support, -1)

        z_support_labels = torch.from_numpy(np.repeat(range(self.n_way), self.n_support))
        
        head = self.initialize_model()
        head.fit(z_support, z_support_labels)
        
        scores = head.get_logits(z_query)
        return scores
    
    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)

        return self.loss_fn(scores, y_query)