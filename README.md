# SKLX

A scikit-learn compatible neural network library that wraps MLX.
Highly inspired by [skorch](https://github.com/skorch-dev/skorch).

> [!WARNING]
> This is still under development and non of the following examples actually work.

## Examples

```python
import numpy as np
from sklearn.datasets import make_classification
from mlx import nn
from sklx import NeuralNetClassifier

X, y = make_classification(1000, 20, n_informative=10, random_state=0)
X = X.astype(np.float32)
y = y.astype(np.int64)

class MyModule(nn.Module):
    def __init__(self, num_units=10, nonlin=nn.ReLU()):
        super().__init__()

        self.dense0 = nn.Linear(20, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(num_units, num_units)
        self.output = nn.Linear(num_units, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        X = self.softmax(self.output(X))
        return X

net = NeuralNetClassifier(
    MyModule,
    max_epochs=10,
    lr=0.1,
)

net.fit(X, y)
y_proba = net.predict_proba(X)
```
