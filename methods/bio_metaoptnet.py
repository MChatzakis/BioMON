import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from methods.meta_template import MetaTemplate
from methods.heads import *


class BioMetaOptNet(MetaTemplate):
    """
    BioMetaOptNet is a MetaOptNet variant for Biomedical data collections.
    """

    def __init__(self, backbone, n_way, n_support, head_model_params):
        """
        Initialize BioMetaOptNet model.

        Args:
            backbone (model): Backbone model to use for feature extraction.
            n_way (int): Number of classes per task.
            n_support (int): Number of support samples per class.
            head_model_params (dict): Dictionary of parameters for the head model.

        """

        super(BioMetaOptNet, self).__init__(backbone, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.head_model_params = head_model_params

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)

        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)
        z_support = z_support.contiguous().view(self.n_way * self.n_support, -1)

        z_support_labels = torch.from_numpy(
            np.repeat(range(self.n_way), self.n_support)
        )

        head_args = self.head_model_params
        head = DISPATCHER[head_args["model"]](
            **head_args["args"], feat_dim=z_support.shape[1]
        )
        head.fit(z_support, z_support_labels)

        scores = head.get_logits(z_query)
                
        #print(f"scores shape: {scores.shape}")

        return scores

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        if torch.cuda.is_available():
            y_query = Variable(y_query.cuda())
        else:
            y_query = Variable(y_query)

        scores = self.set_forward(x)

        return self.loss_fn(scores, y_query)