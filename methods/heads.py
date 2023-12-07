import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


class ClassificationHead(nn.Module):
    """
    Abstract class for classification heads.
    It is not necessary to be a nn.Module, but it is useful for the training loop when creating neural networks.
    """
    
    def __init__(
        self,
        n_way,
        feat_dim,
        seed=42,
    ):
        """
        Instanciate a classification head model. Classification heads are used in the Meta-training loop to calculate the logits for the query set.

        This is meant to be an abstract class, and should not be instanciated directly.

        Classification heads should be compatible with BioMetaOptNet, and should implement the following methods:
        - get_logits(self, query_features): Get the logits for the query set.
        - fit(self, support_features, support_labels): Fit the support set to the support labels.
        - _get_logit_from_probs(self, probabilities): Get the logits from the probabilities. Protected method.
        - test(self, X_test, y_test): Test the performance of the classifier.
        
        n_way (int): Number of classes in the classification task.
        feat_dim (int): Dimension of the feature vectors.
        seed (int, optional): Random seed. Defaults to 42.
        """
        super(ClassificationHead, self).__init__()
        
        self.n_way = n_way
        self.feat_dim = feat_dim
        self.seed = seed
        

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

    def _get_logit_from_probs(self, probabilities):
        """
        Get the logits from the probabilities.
        Many sklearn models do not return logits, but probabilities.
        This method should be used to transform the probabilities into logits.

        Given a probabiliti p e (0, 1), the logit is defined as:
        logit(p) = log(p / (1 - p))
        As per: https://en.wikipedia.org/wiki/Logit

        Args:
            probabilities (np.array): np.array of shape (n_way * size, n_way)

        Returns:
            np.array: np.array of shape (n_way * size, n_way), representing the logits.
        """
        print("Warning: get_logit_from_probs is probably not correct.")
        #return np.log(probabilities / (1.0001 - probabilities))
        return np.log(probabilities)
    
    def test(self, X_test, y_test):
        """
        Test the performance of the classifier.
        It should return a tuple (loss, accuracy, ...).
        
        Args:
            X_test (tensor): Input tensor of shape (n_way * n_query, feat_dim)
            y_test (tensor): Labels tensor of shape (n_way * n_query) 
        """
        pass
        


######################################################################
#                                                                    #
# Classification heads: Classic Algorithms implemented with sklearn  #
#                                                                    #
######################################################################


class SVM_Head(ClassificationHead):
    """
    Multi-class Support Vector Machine classification head.
    """
        
    def __init__(self, n_way, feat_dim, seed, kernel="linear", C=1, probability=True):
        """
        Instanciate a SVM classification head model.

        Args:
            kernel (str, optional): Which kernel to be used (linear, poly, sigmoid, ...). Defaults to 'linear'.
            C (int, optional): L2-Regularization parameter. Defaults to 1.
            probability (bool, optional): _description_. Defaults to True.
        """
        super().__init__(n_way=n_way, feat_dim=feat_dim, seed=seed)

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

    def __init__(self, n_way, feat_dim, seed=42):
        super().__init__(n_way=n_way, feat_dim=feat_dim, seed=42)
        self.model = GaussianNB()

    def get_logits(self, query_features):
        x_test = query_features.detach().numpy()
        probabilities = np.array(self.model.predict_proba(x_test))

        # Generate logits from probabilities:
        scores_raw = self._get_logit_from_probs(probabilities)

        # Transform to trainable tensor:
        scores = torch.from_numpy(scores_raw)

        return scores

    def fit(self, support_features, support_labels):
        X_train = support_features.detach().numpy()
        y_train = support_labels.detach().numpy()
        self.model.fit(X_train, y_train)


class GMM_Head(ClassificationHead):
    def __init__(self, covar_type="full"):
        super().__init__()
        self.nway = ...
        self.model = GMM(
            n_components=self.nway,
            covariance_type=covar_type,
            init_params="wc",
            n_iter=20,
        )

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

    def __init__(
        self,
        n_way,
        feat_dim,
        seed=42,
        hidden_dims=[],
        activations=None,
        dropouts=None,
        epochs=10,
        lr=0.001,
        weight_decay=0.00001,
        batch_size=32,
        device="cpu",
    ):
        super().__init__(n_way=n_way, feat_dim=feat_dim, seed=seed)

        self.hidden_dims = hidden_dims
        self.output_dim = n_way
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

        self.network = nn.Sequential()

        curr_size = feat_dim
        for i, hidden_dim in enumerate(hidden_dims):
            self.network.append(nn.Linear(curr_size, hidden_dim))

            if activations is not None and activations[i] is not None:
                self.network.append(activations[i])

            if dropouts is not None and dropouts[i] is not None:
                self.network.append(nn.Dropout(dropouts[i]))

            curr_size = hidden_dim

        self.network.append(nn.Linear(curr_size, self.output_dim))

        self.to(device)

    def forward(self, x):
        return self.network(x)

    def get_logits(self, query_features):
        return self.forward(query_features)

    def fit(self, support_features, support_labels):
        self.train_model(support_features, support_labels)

    def train_model(self, support_features, support_labels):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        dataset = TensorDataset(support_features, support_labels)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        for epoch in range(self.epochs):
            self.train()
            for X, y in loader:
                loss = self.train_epoch(X, y)
                #print(loss)

        t_loss, t_acc = self.test(support_features, support_labels)

        return t_loss, t_acc

    def train_epoch(self, X, y):
        self.optimizer.zero_grad()

        logits = self.forward(X)
        loss = self.criterion(logits, y)
        loss.backward()

        self.optimizer.step()

        return loss.item()

    def test(self, X_test, y_test):
        dataset = TensorDataset(X_test, y_test)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False
        )

        self.eval()

        total_loss = 0
        true_labels = []
        predicted_labels = []
        with torch.no_grad():
            for X, y in loader:
                logits = self.forward(X)
                loss = self.criterion(logits, y)
                total_loss += loss.item()

                pred_class = torch.argmax(logits, dim=1)
                predicted_labels += list(pred_class.cpu().numpy())
                true_labels += list(y.cpu().numpy())

        f_loss = total_loss / len(loader)
        f_acc = np.mean(np.array(predicted_labels) == np.array(true_labels))

        return f_loss, f_acc


if __name__ == "__main__":
    print("==== Generating random data =====")

    n_way = 5
    n_query = 10
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

    # MLP Head
    head = MLP_Head(
        n_way,
        emb_dim,
        seed=42,
        epochs=5,
        hidden_dims=[512, 256, 64, 32],
        dropouts=[0.4, 0.4, 0.4, 0.4],
        activations=[nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()],
        device="cpu",
    )
    head.fit(z_support, z_labels)
    logits = head.get_logits(z_query)
    assert logits.shape == (n_way * n_query, n_way), "Wrong shape for MLP logits."
    print(">>MLP Ok!")

    #SVM Head
    head = SVM_Head(n_way=n_way, feat_dim=emb_dim, seed=42)
    head.fit(z_support, z_labels)
    logits = head.get_logits(z_query)
    assert logits.shape == (n_way * n_query, n_way), "Wrong shape for SVM logits."
    print(">>SVM Ok!")
    
    # Decision Tree Head
    # head = DecisionTree_Head()
    # head.fit(z_support, z_labels)
    # logits = head.get_logits(z_query)
    # assert logits.shape == (n_way * n_query, n_way), "Wrong shape for DecTree logits."

    # Naive Bayes Head
    head = NaiveBayes_Head(n_way=n_way, feat_dim=emb_dim, seed=42)
    head.fit(z_support, z_labels)
    logits = head.get_logits(z_query)
    #print(logits)
    assert logits.shape == (n_way * n_query, n_way), "Wrong shape for NB logits."
    print(">>Naive Bayes Ok!")

    
