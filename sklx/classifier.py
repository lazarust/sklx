from typing import Callable

import mlx.optimizers.optimizers as optimizers
from mlx import nn
from sklearn.base import ClassifierMixin

from sklx.net import NeuralNet


class NeuralNetworkClassifier(NeuralNet, ClassifierMixin):
    module = None
    max_epochs = 10
    lr = 0.1
    batch_size = 10
    optimizer = optimizers.SGD

    def __init__(
        self,
        module: nn.Module,
        max_epochs: float,
        lr: float,
        criterion: Callable,
    ) -> None:
        self.module = module
        self.max_epochs = max_epochs
        self.lr = lr
        self.criterion = criterion
        self.optimizer = optimizers.SGD(learning_rate=lr)

    def fit(self, raw_X, raw_y, **kwargs):
        return super().fit(raw_X, raw_y, **kwargs)
