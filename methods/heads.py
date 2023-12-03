import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class ClassificationHead(nn.Module):
    def __init__(self):
        """
        Instanciate a classification head model. Classification heads are used in the Meta-training loop to calculate the logits for the query set.

        This is meant to be an abstract class, and should not be instanciated directly.
        
        Classification heads should be compatible with BioMetaOptNet, and should implement the following methods:
        - get_logits(self, query_features): Get the logits for the query set.
        - fit(self, support_features, support_labels): Fit the support set to the support labels.
        """
        super(ClassificationHead, self).__init__()
        raise NotImplementedError
    
    def get_logits(self, query_features):
        """
        Performs a pass through the head model to get logits for the query features.

        Args:
            query_features (tensor): tensor of shape (n_way * n_query, feat_dim)

        Returns:
            logits (tensor): tensor of shape (n_way * n_query, n_way)
        """
        raise NotImplementedError

    def fit(self, support_features, support_labels):
        """
        Complete training routine for the head model.
        It should fit the support features to the support labels.

        Args:
            support_features (tensor): Tensor of shape (n_way * n_support, feat_dim)
            support_labels (_type_): Tensor of shape (n_way * n_support, 1)
        """
        raise NotImplementedError
