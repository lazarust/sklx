import numpy as np
import pytest
from mlx import nn
from sklearn.datasets import make_regression

from sklx.regressor import NueralNetworkRegressor


class TestNueralNetRegressor:
    @pytest.fixture
    @staticmethod
    def nerual_network_regressor():
        class RegressorModule(nn.Module):
            def __init__(
                self,
                num_units=10,
                nonlin=nn.relu,
            ):
                super().__init__()
                self.num_units = num_units
                self.layers = [
                    nn.Linear(20, num_units),
                    nn.Linear(num_units, 10),
                    nn.Linear(10, 1),
                ]
                self.dense0 = nn.Linear(20, num_units)
                self.nonlin = nonlin
                self.dense1 = nn.Linear(num_units, 10)
                self.output = nn.Linear(10, 1)

            def forward(self, X, **kwargs):
                X = self.nonlin(self.dense0(X))
                X = nn.relu(self.dense1(X))
                X = self.output(X)
                return X

            def __call__(self, X, **kwargs):
                for _, layer in enumerate(self.layers):
                    X = layer(X)
                return X

        net = NueralNetworkRegressor(RegressorModule, max_epochs=10, lr=0.1)
        return net

    def test_regressor(self, nerual_network_regressor):
        """
        Test to make sure the basic functionality of the regressor works.
        """
        X_regr, y_regr = make_regression(1000, 20, n_informative=10, random_state=0)
        X_regr = X_regr.astype(np.float32)
        y_regr = y_regr.astype(np.float32) / 100
        y_regr = y_regr.reshape(-1, 1)

        nerual_network_regressor.fit(X_regr, y_regr)
        nerual_network_regressor.predict(X_regr)
