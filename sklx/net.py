import time

import mlx.core as core
import mlx.nn as nn
import mlx.optimizers.optimizers as optimizers
import numpy as np
from sklearn.model_selection import train_test_split
from tabulate import tabulate


class NeuralNet:
    def __init__(
        self,
        module: nn.Module,
        criterion,
        optimizer=optimizers.SGD,
        lr=0.1,
        max_epochs=10,
        batch_size=128,
    ) -> None:
        self.module = module
        self.criterion = criterion
        self.optimizer = optimizer(learning_rate=lr)
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs

    def batch_iterate(self, batch_size, X, y):
        perm = core.array(np.random.permutation(y.size))
        for s in range(0, y.size, batch_size):
            ids = perm[s : s + batch_size]
            yield X[ids], y[ids]

    def loss_fn(self, model, X, y):
        return self.criterion(model(X), y, reduction="mean")

    def fit(self, raw_X, raw_y, **kwargs):
        self.module = self.module()
        self.module.train(mode=True)
        core.eval(self.module.parameters())
        loss_and_grad_fn = nn.value_and_grad(self.module, self.loss_fn)

        raw_X, X_test, raw_y, y_test = train_test_split(raw_X, raw_y)

        headers = [
            "epoch",
            "train_loss",
            "valid_acc",
            "valid_loss",
            "dur",
        ]
        metrics_list = []

        for epoch in range(1, self.max_epochs + 1):
            start_time = time.time()
            batch_losses = []
            valid_scores = []

            for X, y in self.batch_iterate(self.batch_size, raw_X, raw_y):
                loss, grads = loss_and_grad_fn(
                    self.module, core.array(X), core.array(y)
                )
                self.optimizer.update(model=self.module, gradients=grads)

                batch_losses.append(loss)

            avg_loss = np.mean(batch_losses)

            score = self.score(X_test, y_test)
            valid_scores.append(score)

            valid_loss, _ = loss_and_grad_fn(
                self.module, core.array(X_test), core.array(y_test)
            )

            total_time = time.time() - start_time

            metrics_list.append(
                [
                    f"{epoch}",
                    f"{avg_loss:.4f}",
                    f"{score:.4f}",
                    f"{float(valid_loss):.4f}",
                    f"{total_time:.4f}",
                ]
            )

            core.eval(self.module.parameters(), self.optimizer.state)

        print(tabulate(metrics_list, headers=headers))
        self.module.train(mode=False)

    def predict(self, X):
        return list(self.module(core.array(X)).argmax(axis=1))

    def predict_proba(self, X):
        pass
