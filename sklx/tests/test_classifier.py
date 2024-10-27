import numpy as np
from mlx import nn
from mlx.nn import losses
from sklearn.datasets import make_classification

from sklx.classifier import NeuralNetworkClassifier


def test_neural_network_classifier():
    """
    This is just a simple test to make sure the basic usage works.
    """
    X, y = make_classification(1000, 20, n_informative=10, random_state=0)
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    class MyModule(nn.Module):
        def __init__(self, num_units=10, nonlin=nn.ReLU()):
            super().__init__()
            self.layers = [
                nn.Linear(20, num_units),
                nonlin,
                nn.Dropout(0.5),
                nn.Linear(num_units, num_units),
                nn.Linear(num_units, 2),
                nn.LogSoftmax(),
            ]

        def __call__(self, X, **kwargs):
            for _, layer in enumerate(self.layers):
                X = layer(X)
            return X

    net = NeuralNetworkClassifier(
        MyModule, max_epochs=10, lr=0.1, criterion=losses.nll_loss
    )

    net.fit(X, y)
    net.predict_proba(X)
