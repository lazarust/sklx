import numpy as np
import pytest
from mlx import nn
from mlx.nn import losses
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklx.classifier import NeuralNetworkClassifier


def test_run():
    print("Test")


class TestNeuralNetworkClassifier:
    @pytest.fixture
    @staticmethod
    def nerual_network_classifier():
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

        return net

    def test_neural_network_classifier(self, nerual_network_classifier):
        """
        This is just a simple test to make sure the basic usage works.
        """
        X, y = make_classification(1000, 20, n_informative=10, random_state=0)
        X = X.astype(np.float32)
        y = y.astype(np.int64)
        nerual_network_classifier.fit(X, y)
        nerual_network_classifier.predict_proba(X)

    def test_sklearn_pipeline_support(self, nerual_network_classifier):
        """
        Test to make sure that using a model in a Sklearn pipeline works.
        """
        X, y = make_classification(1000, 20, n_informative=10, random_state=0)
        X = X.astype(np.float32)
        y = y.astype(np.int64)

        pipe = Pipeline(
            [
                ("scale", StandardScaler()),
                ("net", nerual_network_classifier),
            ]
        )

        pipe.fit(X, y)
        pipe.predict_proba(X)
