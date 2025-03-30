from mlx.nn import losses
from sklearn.base import RegressorMixin

from sklx.net import NeuralNet


class NueralNetworkRegressor(RegressorMixin, NeuralNet):
    """
    Implmentation of the NueralNet class. See `sklx.NeuralNet` for more details.
    """

    def __init__(self, module, *args, criterion=losses.mse_loss, **kwargs):
        super().__init__(module, *args, criterion=criterion, **kwargs)
