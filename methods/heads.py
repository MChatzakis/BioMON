import numpy as np
import torch
import time
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from sklearn import svm
from sklearn import mixture

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Ridge, LogisticRegression


###########################################################
#                                                         #
# Class protypes and definitions for classification heads #
#                                                         #
###########################################################


class ClassificationHead:
    """
    Abstract class for classification heads.
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
        - test(self, X_test, y_test): Test the performance of the classifier.

        n_way (int): Number of classes in the classification task.
        feat_dim (int): Dimension of the feature vectors.
        seed (int, optional): Random seed. Defaults to 42.
        """
        super(ClassificationHead, self).__init__()

        self.n_way = n_way
        self.feat_dim = feat_dim
        self.seed = seed
        
        self.fit_time = None
        self.test_time = None

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

    def test(self, X_test, y_test):
        """
        Test the performance of the classifier.
        It should return a tuple (loss, accuracy,).

        Args:
            X_test (tensor): Input tensor of shape (n_way * n_query, feat_dim)
            y_test (tensor): Labels tensor of shape (n_way * n_query)
        """
        pass


class TorchClassificationHead(ClassificationHead, nn.Module):
    """
    Torch classification head used mostly for Neural Network classifiers.
    It inherits from both ClassificationHead and nn.Module.
    """

    def __init__(self, n_way, feat_dim, seed=42, batch_size=32, epochs=5):
        """
        Instanciate a Torch classification head model.

        This is meant to be an abstract class, and should not be instanciated directly.

        Torch models should be compatible with BioMetaOptNet, and should implement the following methods:
        - forward(self, x): Forward pass through the model.
        - train_model(self, support_features, support_labels): Complete training routine for the head model.
        - test(self, X_test, y_test): Test the performance of the classifier.
        - methods inherited from ClassificationHead

        Additional args:
        - batch_size (int, optional): Batch size for training. Defaults to 32.
        - epochs (int, optional): Number of epochs for training. Defaults to 2.
        - device (str, optional): Device to use for training. Defaults to "cpu".

        """
        super().__init__(n_way=n_way, feat_dim=feat_dim, seed=seed)
        self.batch_size = batch_size
        self.epochs = epochs

        self.final_train_loss = None
        self.final_train_acc = None

        self.criterion = None
        self.optimizer = None

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (tensor): Tensor of shape (n_way * n_{query,support}, feat_dim)
        """
        pass

    def train_model(self, support_features, support_labels):
        """
        Complete training routine for the head model.

        Args:
            support_features (tensor): Tensor of shape (n_way * n_support, feat_dim)
            support_labels (tensor): Tensor of shape (n_way * n_support)
        """

        start_time = time.time()
        
        dataset = TensorDataset(support_features, support_labels)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        for epoch in range(self.epochs):
            self.train()
            for X, y in loader:
                if torch.cuda.is_available():
                    X = X.cuda()
                    y = y.cuda()

                self.optimizer.zero_grad()

                logits = self.forward(X)
                loss = self.criterion(logits, y)
                loss.backward(retain_graph=True)

                self.optimizer.step()

        t_acc = self.test(support_features, support_labels)

        self.final_train_acc = t_acc
        #self.final_train_loss = t_loss
        self.fit_time = time.time() - start_time
        
        return t_acc

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
                if torch.cuda.is_available():
                    X = X.cuda()
                    y = y.cuda()

                logits = self.forward(X)
                loss = self.criterion(logits, y)
                total_loss += loss.item()

                pred_class = torch.argmax(logits, dim=1)
                predicted_labels += list(pred_class.cpu().detach().numpy())
                true_labels += list(y.cpu().detach().numpy())

        f_loss = total_loss / len(loader)
        f_acc = np.mean(np.array(predicted_labels) == np.array(true_labels))

        return f_acc

    def get_logits(self, query_features):
        return self.forward(query_features)

    def fit(self, support_features, support_labels):
        self.train_model(support_features, support_labels)


class ClassicClassificationHead(ClassificationHead):
    """
    Classic classification head.
    """

    def __init__(self, n_way, feat_dim, seed=42):
        """
        Initialize a classic classification head model.
        This is meant to be an abstract class, and should not be instanciated directly.

        Classic models should be compatible with BioMetaOptNet, and most of them are implemented with sklearn.
        """

        super().__init__(n_way=n_way, feat_dim=feat_dim, seed=seed)
        self.model = None

    def get_logits(self, query_features):
        x_test = query_features.cpu().detach().numpy()
        probabilities = np.array(self.model.predict_proba(x_test))

        # Generate logits from probabilities:
        scores_raw = self._get_logit_from_probs(probabilities)

        # Transform to trainable tensor:
        # scores = torch.from_numpy(scores_raw)
        scores = to_torch(x=scores_raw, requires_grad=True)

        return scores

    def fit(self, support_features, support_labels):
        X_train = support_features.cpu().detach().numpy()
        y_train = support_labels.cpu().detach().numpy()
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.fit_time = time.time() - start_time

    def test(self, X_test, y_test):
        X_test = X_test.cpu().detach().numpy()
        y_test = y_test.cpu().detach().numpy()
        return self.model.score(X_test, y_test)

    def _get_logit_from_probs(self, probabilities):
        """
        Get the logits from the probabilities.
        Many sklearn models do not return logits, but probabilities.
        This method should be used to transform the probabilities into logits.

        Given a probability p e (0, 1), the logit is defined as: (Get ready for the show)
        logit(p) = log(p / (1 - p))
        (https://en.wikipedia.org/wiki/Logit)

        Args:
            probabilities (np.array): np.array of shape (n_way * size, n_way)

        Returns:
            np.array: np.array of shape (n_way * size, n_way), representing the logits.
        """
        c = 0.00000001
        nom = probabilities + c
        denom = 1 - probabilities + c
        return np.log(nom / denom)


######################################################################
#                                                                    #
# Classification heads: Classic Algorithms implemented with sklearn  #
#                                                                    #
######################################################################


class SVM_Head(ClassicClassificationHead):
    """
    Multi-class Support Vector Machine classification head.
    """

    def __init__(
        self,
        n_way,
        feat_dim,
        seed=42,
        kernel="linear",
        C=1,
        probability=True,
    ):
        """
        Instanciate a SVM classification head model.

        Args:
            kernel (str, optional): Which kernel to be used (linear, poly, sigmoid, ...). Defaults to 'linear'.
            C (int, optional): L2-Regularization parameter. Defaults to 1.
            probability (bool, optional): _description_. Defaults to True.
        """
        super().__init__(n_way=n_way, feat_dim=feat_dim, seed=seed)

        self.model = svm.SVC(
            kernel=kernel, C=C, probability=probability, random_state=seed
        )

    def get_logits(self, query_features):
        x_test = query_features.cpu().detach().numpy()
        scores_raw = self.model.decision_function(x_test)

        # Transform to trainable tensor:
        scores = to_torch(x=scores_raw, requires_grad=True)
        # print(f"Scores shape: {scores.shape}")
        # print(scores)

        return scores


class RidgeRegression_Head(ClassicClassificationHead):
    """
    Ridge-Regression classification head.
    """

    def __init__(
        self,
        n_way,
        feat_dim,
        seed=42,
        alpha=1.0,
        fit_intercept=True,
        solver="lsqr",
    ):
        """
        Instanciate a Ridge-Regression classification head.

        Args:
            alpha(float,optional): Whether to fit the intercept for this model. If False X and y are expected to be centered.
            it_intercept(bool,optional): Whether to fit the intercept for this model. If False X and y are expected to be centered.
            solver (string, optional): Constant that multiplies the L2 term, controlling regularization strength.
        """
        super().__init__(n_way=n_way, feat_dim=feat_dim, seed=seed)

        self.model = Ridge(alpha=alpha, fit_intercept=fit_intercept, solver=solver)

    def get_logits(self, query_features):
        x_test = query_features.cpu().detach().numpy()
        scores_raw = self.model.predict(x_test)

        # Transform to trainable tensor:
        scores = to_torch(x=scores_raw, requires_grad=True)
        # print(f"Scores shape: {scores.shape}") -> (samples, 1)
        # print(scores)
        return scores


class NaiveBayes_Head(ClassicClassificationHead):
    """
    Naive Bayes classification head.
    """

    def __init__(self, n_way, feat_dim, seed=42):
        super().__init__(n_way=n_way, feat_dim=feat_dim, seed=seed)
        self.model = GaussianNB()


class KNN_Head(ClassicClassificationHead):
    def __init__(self, n_way, feat_dim, seed=42, n_neighbors=3):
        super().__init__(
            n_way=n_way,
            feat_dim=feat_dim,
            seed=seed,
        )

        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)


class DecisionTree_Head(ClassicClassificationHead):
    """
    Decision Tree classification head.
    """

    def __init__(self, n_way, feat_dim, seed=42):
        super().__init__(n_way=n_way, feat_dim=feat_dim, seed=seed)
        self.model = DecisionTreeClassifier(random_state=seed)


class RandomForest_Head(ClassicClassificationHead):
    def __init__(self, n_way, feat_dim, seed=42, n_estimators=100):
        super().__init__(n_way=n_way, feat_dim=feat_dim, seed=seed)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=seed
        )


class GMM_Head(ClassicClassificationHead):
    def __init__(
        self,
        n_way,
        feat_dim,
        seed=42,
        covar_type="full",
        init_params="kmeans",
    ):
        super().__init__(n_way=n_way, feat_dim=feat_dim, seed=seed)
        self.model = mixture.GaussianMixture(
            n_components=n_way,
            covariance_type=covar_type,
            init_params=init_params,
            random_state=seed,
        )

    def test(self, X_test, y_test):
        X_test = X_test.cpu().detach().numpy()
        y_test = y_test.cpu().detach().numpy()

        y_test_pred = self.model.predict(X_test)
        test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel())

        return test_accuracy

    def fit(self, support_features, support_labels):
        
        X_train = support_features.cpu().detach().numpy()
        start_time = time.time()
        self.model.fit(X_train)
        self.fit_time = time.time() - start_time


class LogisticRegression_Head(ClassicClassificationHead):
    """
    Multi-class Logistic Regression classification head.
    """

    def __init__(
        self,
        n_way,
        feat_dim,
        seed=42,
        max_iter=100,
        C=1.0,
    ):
        super().__init__(n_way=n_way, feat_dim=feat_dim, seed=seed)

        self.model = LogisticRegression(random_state=seed, max_iter=max_iter, C=C)


##########################################
#                                        #
# Classification heads: Neural Networks  #
#                                        #
##########################################


class MLP_Head(TorchClassificationHead):
    """
    Multi-class Neural Network classification head.
    """

    def __init__(
        self,
        n_way,
        feat_dim,
        seed=42,
        batch_size=32,
        epochs=5,
        hidden_dims=[],
        activations=None,
        dropouts=None,
        lr=0.001,
        weight_decay=0.00001,
    ):
        super().__init__(
            n_way=n_way,
            feat_dim=feat_dim,
            seed=seed,
            batch_size=batch_size,
            epochs=epochs,
        )

        self.hidden_dims = hidden_dims
        self.output_dim = n_way
        self.lr = lr
        self.weight_decay = weight_decay

        self.network = nn.Sequential()

        curr_size = feat_dim
        for i, hidden_dim in enumerate(hidden_dims):
            self.network.append(nn.Linear(curr_size, hidden_dim))

            if activations is not None and activations[i] is not None:
                if type(activations[i]) == str:
                    if activations[i] == "relu":
                        self.network.append(nn.ReLU())
                    elif activations[i] == "sigmoid":
                        self.network.append(nn.Sigmoid())
                    elif activations[i] == "tanh":
                        self.network.append(nn.Tanh())
                else:
                    self.network.append(activations[i])

            if dropouts is not None and dropouts[i] is not None:
                self.network.append(nn.Dropout(dropouts[i]))

            curr_size = hidden_dim

        self.network.append(nn.Linear(curr_size, self.output_dim))

        if torch.cuda.is_available():
            self.to("cuda")

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def forward(self, x):
        return self.network(x)


class LogisticRegressionTorch_Head(TorchClassificationHead):
    def __init__(
        self,
        n_way,
        feat_dim,
        seed=42,
        batch_size=32,
        epochs=5,
        lr=0.001,
    ):
        super().__init__(
            n_way=n_way,
            feat_dim=feat_dim,
            seed=seed,
            batch_size=batch_size,
            epochs=epochs,
        )
        self.lr = lr
        self.model = nn.Linear(feat_dim, n_way)

        if torch.cuda.is_available():
            self.to("cuda")

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)

    def forward(self, x):
        return torch.sigmoid(self.model(x))


##############################
#                            #
# Dispatcher and other utils #
#                            #
##############################

# Note: When adding a new model, be sure to update the dispatcher.
DISPATCHER = {
    "MLP": MLP_Head,
    "LogisticRegression": LogisticRegression_Head,
    "SVM": SVM_Head,
    "NaiveBayes": NaiveBayes_Head,
    "KNN": KNN_Head,
    "DecisionTree": DecisionTree_Head,
    "RandomForest": RandomForest_Head,
    "GMM": GMM_Head,
    "RR": RidgeRegression_Head,
}


def to_torch(x, requires_grad=True):
    """
    Transform a numpy array to a torch tensor.

    Args:
        x (np.array): Numpy array to transform.

    Returns:
        Tensor: Pytorch tensor.
    """

    ten = torch.from_numpy(x)

    if torch.cuda.is_available():
        ten = ten.cuda()

    if requires_grad:
        ten.requires_grad = True

    # Variable(ten.cuda()) # deprecated

    return ten


##########################################
#                                        #
# Classification heads: Neural Networks  #
#                                        #
##########################################

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
        n_way=n_way,
        feat_dim=emb_dim,
        seed=42,
        batch_size=32,
        epochs=500,
        hidden_dims=[512, 256, 64, 32],
        activations=[nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()],
        dropouts=[0.4, 0.4, 0.4, 0.4],
        lr=0.001,
        weight_decay=0.00001,
    )
    head.fit(z_support, z_labels)
    logits = head.get_logits(z_query)
    assert logits.shape == (n_way * n_query, n_way), "Wrong shape for MLP logits."
    print(f"MLP test: {head.test(z_query, z_query_labels)}")
    print(">>MLP Ok!\n")

    # Logistic Regression Head
    head = LogisticRegression_Head(n_way=n_way, feat_dim=emb_dim, seed=42)
    head.fit(z_support, z_labels)
    logits = head.get_logits(z_query)
    assert logits.shape == (n_way * n_query, n_way), "Wrong shape for LR logits."
    print(f"LR test: {head.test(z_query, z_query_labels)}")
    print(">>LR Ok!\n")

    # SVM Head
    head = SVM_Head(n_way=n_way, feat_dim=emb_dim, seed=42)
    head.fit(z_support, z_labels)
    logits = head.get_logits(z_query)
    assert logits.shape == (n_way * n_query, n_way), "Wrong shape for SVM logits."
    print(f"SVM test: {head.test(z_query, z_query_labels)}")
    print(">>SVM Ok!\n")

    # Decision Tree Head
    head = DecisionTree_Head(n_way=n_way, feat_dim=emb_dim, seed=42)
    head.fit(z_support, z_labels)
    logits = head.get_logits(z_query)
    assert logits.shape == (n_way * n_query, n_way), "Wrong shape for DecTree logits."
    print(f"DecTree test: {head.test(z_query, z_query_labels)}")
    print(">>DecTree Ok!\n")

    # Naive Bayes Head
    head = NaiveBayes_Head(n_way=n_way, feat_dim=emb_dim, seed=42)
    head.fit(z_support, z_labels)
    logits = head.get_logits(z_query)
    assert logits.shape == (n_way * n_query, n_way), "Wrong shape for NB logits."
    print(f"NB test: {head.test(z_query, z_query_labels)}")
    print(">>Naive Bayes Ok!\n")

    # KNN
    head = KNN_Head(n_way=n_way, feat_dim=emb_dim, seed=42, n_neighbors=3)
    head.fit(z_support, z_labels)
    logits = head.get_logits(z_query)
    assert logits.shape == (n_way * n_query, n_way), "Wrong shape for KNN logits."
    print(f"KNN test: {head.test(z_query, z_query_labels)}")
    print(">>KNN Ok!\n")

    # Random Forest
    head = RandomForest_Head(n_way=n_way, feat_dim=emb_dim, seed=42, n_estimators=100)
    head.fit(z_support, z_labels)
    logits = head.get_logits(z_query)
    assert logits.shape == (n_way * n_query, n_way), "Wrong shape for RF logits."
    print(f"RF test: {head.test(z_query, z_query_labels)}")
    print(">>RF Ok!\n")

    # GMM
    head = GMM_Head(n_way=n_way, feat_dim=emb_dim, seed=42)
    head.fit(z_support, z_labels)
    logits = head.get_logits(z_query)
    assert logits.shape == (n_way * n_query, n_way), "Wrong shape for GMM logits."
    print(f"GMM test: {head.test(z_query, z_query_labels)}")
    print(">>GMM Ok!")
