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
        self.feature = backbone
        self.n_way = n_way
        self.n_support = n_support
        self.head_model_params = head_model_params

    def initialize_model(self) -> ClassificationHead:
        """
        Initialize the classification head model.
        
        It is based on the model specified in the head_model_params dictionary, with the following format:
        {
            "model": model name,
            "args": {
                ...
            },
        }
        
        Shared arguments among all models:
        -n_way,
        -feat_dim,
        -seed,
        -device

        Returns:
            ClassificationHead: A new instance of the classification head model.
        """
        args = self.head_model_params
        return DISPATCHER[args["model"]](**args["args"])

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
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)

        return self.loss_fn(scores, y_query)
