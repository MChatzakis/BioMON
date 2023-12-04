import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn import svm


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
        pass

    def get_logits(self, query_features):
        """
        Performs a pass through the head model to get logits for the query features.

        Args:
            query_features (tensor): tensor of shape (n_way * n_query, feat_dim)

        Returns:
            logits (tensor): tensor of shape (n_way * n_query, n_way)
        """
        pass

    def fit(self, support_features, support_labels):
        """
        Complete training routine for the head model.
        It should fit the support features to the support labels.

        Args:
            support_features (tensor): Tensor of shape (n_way * n_support, feat_dim)
            support_labels (_type_): Tensor of shape (n_way * n_support, 1)
        """
        pass

class SVM_Head(ClassificationHead):
    """
    Multi-class Support Vector Machine classification head.
    """
    def __init__(self, kernel='linear', C=1, probability=True):
        """
        Instanciate a SVM classification head model.

        Args:
            kernel (str, optional): Which kernel to be used (linear, poly, sigmoid, ...). Defaults to 'linear'.
            C (int, optional): L2-Regularization parameter. Defaults to 1.
            probability (bool, optional): _description_. Defaults to True.
        """
        super().__init__()
        
        self.model = svm.SVC(kernel=kernel, C=C, probability=probability)

        
    def get_logits(self, query_features):
        y_test = query_features.detach().numpy()
        scores_raw = self.model.decision_function(y_test)
        
        # Transform to trainable tensor:
        scores = torch.from_numpy(scores_raw)
        
        return scores

    def fit(self, support_features, support_labels):
        X_train = support_features.detach().numpy()
        y_train = support_labels.detach().numpy()
        
        self.model.fit(X_train, y_train)

class NN_Head(ClassificationHead):
    """
    Multi-class Neural Network classification head.
    """
    
    def __init__(self):
        super().__init__()
        pass

    def get_logits(self, query_features):
        pass

    def fit(self, support_features, support_labels):
        pass


class RR_Head(ClassificationHead):
    """
    Ridge Regression classification head.
    """
    def __init__(self):
        super().__init__()
        pass

    def get_logits(self, query_features):
        pass

    def fit(self, support_features, support_labels):
        pass


class DecisionTree_Head(ClassificationHead):
    """
    Decision Tree classification head.
    """
    def __init__(self):
        super().__init__()
        pass

    def get_logits(self, query_features):
        pass

    def fit(self, support_features, support_labels):
        pass


class RandomForest_Head(ClassificationHead):
    def __init__(self):
        super().__init__()
        pass

    def get_logits(self, query_features):
        pass

    def fit(self, support_features, support_labels):
        pass


class LogisticRegression_Head(ClassificationHead):
    def __init__(self):
        super().__init__()
        pass

    def get_logits(self, query_features):
        pass

    def fit(self, support_features, support_labels):
        pass


class KNN_Head(ClassificationHead):
    def __init__(self):
        super().__init__()
        pass

    def get_logits(self, query_features):
        pass

    def fit(self, support_features, support_labels):
        pass

class DecisionTree_Head(ClassificationHead):
    def __init__(self):
        super().__init__()
        pass

    def get_logits(self, query_features):
        pass

    def fit(self, support_features, support_labels):
        pass
    
class NaiveBayes_Head(ClassificationHead):
    def __init__(self):
        super().__init__()
        pass
    
    def get_logits(self, query_features):
        pass
    
    def fit(self, support_features, support_labels):
        pass
    
class GMM_Head(ClassificationHead):
    def __init__(self):
        super().__init__()
        pass
    
    def get_logits(self, query_features):
        pass
    
    def fit(self, support_features, support_labels):
        pass