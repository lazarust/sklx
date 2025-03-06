from typing import Callable

import mlx.optimizers.optimizers as optimizers
import numpy as np
from mlx import nn
from sklearn.base import ClassifierMixin

from sklx.net import NeuralNet


class NeuralNetworkClassifier(ClassifierMixin, NeuralNet):
    """
    Implementation of the NerualNet base class. See `sklx.net.NerualNet` for detailed documentation.
    """

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
        classes=None,
    ) -> None:
        super().__init__(
            module=module, criterion=criterion, lr=lr, max_epochs=max_epochs
        )
        self.classes = classes

    @property
    def classes_(self):
        return np.array(self.classes)
