import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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
            support_labels (_type_): Tensor of shape (n_way * n_support)
        """
        pass
    
    def get_logit_from_probs(self, probabilities):
        """
        Get the logits from the probabilities. 
        Many sklearn models do not return logits, but probabilities. 
        This method should be used to transform the probabilities into logits.
        
        Given a probabiliti p e (0, 1), the logit is defined as:
        logit(p) = log(p / (1 - p)
        As per: https://en.wikipedia.org/wiki/Logit
        
        Args:
            probabilities (np.array): np.array of shape (n_way * size, n_way)

        Returns:
            np.array: np.array of shape (n_way * size, n_way), representing the logits.
        """
        print("Warning: get_logit_from_probs is probably not correct.")
        return np.log(probabilities / (1.0001 - probabilities))


######################################################################
#                                                                    #
# Classification heads: Classic Algorithms implemented with sklearn  #
#                                                                    #
######################################################################


class SVM_Head(ClassificationHead):
    """
    Multi-class Support Vector Machine classification head.
    """

    def __init__(self, kernel="linear", C=1, probability=True):
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
        x_test = query_features.detach().numpy()
        scores_raw = self.model.decision_function(x_test)

        # Transform to trainable tensor:
        scores = torch.from_numpy(scores_raw)

        return scores

    def fit(self, support_features, support_labels):
        X_train = support_features.detach().numpy()
        y_train = support_labels.detach().numpy()
        self.model.fit(X_train, y_train)


class RR_Head(ClassificationHead):
    """
    Ridge Regression classification head.
    """

    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def get_logits(self, query_features):
        raise NotImplementedError

    def fit(self, support_features, support_labels):
        raise NotImplementedError


class DecisionTree_Head(ClassificationHead):
    """
    Decision Tree classification head.
    """

    def __init__(self):
        super().__init__()
        self.model = DecisionTreeClassifier()

    def get_logits(self, query_features):
        x_test = query_features.detach().numpy()
        probabilities = np.array(self.model.predict_proba(x_test))

        # Generate logits from probabilities:
        scores_raw = self.get_logit_from_probs(probabilities)
        
        # Transform to trainable tensor:
        scores = torch.from_numpy(scores_raw)

        return scores

    def fit(self, support_features, support_labels):
        X_train = support_features.detach().numpy()
        y_train = support_labels.detach().numpy()

        self.model.fit(X_train, y_train)


class RandomForest_Head(ClassificationHead):
    def __init__(self):
        super().__init__()
        self.model = RandomForestClassifier()

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


class LogisticRegression_Head(ClassificationHead):
    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def get_logits(self, query_features):
        raise NotImplementedError

    def fit(self, support_features, support_labels):
        raise NotImplementedError


class KNN_Head(ClassificationHead):
    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def get_logits(self, query_features):
        raise NotImplementedError

    def fit(self, support_features, support_labels):
        raise NotImplementedError


class NaiveBayes_Head(ClassificationHead):
    """
    Naive Bayes classification head.
    """

    def __init__(self):
        super().__init__()
        self.model = GaussianNB()

    def get_logits(self, query_features):
        x_test = query_features.detach().numpy()
        probabilities = np.array(self.model.predict_proba(x_test))

        # Generate logits from probabilities:
        scores_raw = self.get_logit_from_probs(probabilities)
        
        # Transform to trainable tensor:
        scores = torch.from_numpy(scores_raw)

        return scores

    def fit(self, support_features, support_labels):
        X_train = support_features.detach().numpy()
        y_train = support_labels.detach().numpy()

        self.model.fit(X_train, y_train)


class GMM_Head(ClassificationHead):
    def __init__(self, covar_type='full'):
        super().__init__()
        self.nway = ...
        self.model = GMM(n_components=self.nway,covariance_type=covar_type, init_params='wc', n_iter=20)

    def get_logits(self, query_features):
        raise NotImplementedError

    def fit(self, support_features, support_labels):
        raise NotImplementedError


##########################################
#                                        #
# Classification heads: Neural Networks  #
#                                        #
##########################################


class MLP_Head(ClassificationHead):
    """
    Multi-class Neural Network classification head.
    """

    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def get_logits(self, query_features):
        raise NotImplementedError

    def fit(self, support_features, support_labels):
        raise NotImplementedError



if __name__ == "__main__":
    print("==== Generating random data =====")
    
    n_way = 5
    n_query = 15
    n_support = 5
    emb_dim = 512
    
    z_support = torch.rand(n_way * n_support, emb_dim)
    print(f"z_support shape: ", z_support.shape)
    
    z_query = torch.rand(n_way * n_query, emb_dim)
    print(f"z_query shape: ", z_query.shape)
    
    z_labels = torch.from_numpy(np.repeat(range(n_way), n_support))
    print(f"z_labels shape: ", z_labels.shape)
    
    z_query_labels = torch.from_numpy(np.repeat(range(n_way), n_query))
    print(f"z_query_labels shape: ", z_query_labels.shape)
    
    print("\n==== Testing heads =====")
    
    # Decision Tree Head
    head = DecisionTree_Head()
    head.fit(z_support, z_labels)
    logits = head.get_logits(z_query)
    assert logits.shape == (n_way * n_query, n_way), "Wrong shape for Decision Tree logits."
    
    # Naive Bayes Head
    head = NaiveBayes_Head()
    head.fit(z_support, z_labels)
    logits = head.get_logits(z_query)
    assert logits.shape == (n_way * n_query, n_way), "Wrong shape for Naive Bayes logits."
    print(">>Naive Bayes Ok!")
        
    # SVM Head
    head = SVM_Head()
    head.fit(z_support, z_labels)
    logits = head.get_logits(z_query)
    assert logits.shape == (n_way * n_query, n_way), "Wrong shape for SVM logits."
    print(">>SVM Ok!")


    
    
    