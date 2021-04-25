from matplotlib.pyplot import cla
import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam
from TorchMetricLogger import TmlMetric, TMLDiceCoefficient, TMLF1, TMLMean
from TorchMetricLogger import TorchMetricLogger as TML
from TorchMetricLogger import TmlMetricFunction
import numpy as np


class CategoricalAccuracy(TmlMetricFunction):
    def __call__(self, metric):
        if metric.weights is None:
            metric.weights = np.ones(metric.predictions.shape)

        super().__call__(metric)

    def calculate(self, metric):
        tp = np.sum(
            (metric.gold_labels == metric.predictions) * metric.weights
        )
        false_predicition = np.sum(
            (metric.gold_labels != metric.predictions) * metric.weights
        )

        return {"tp": tp, "false_prediction": false_predicition}

    def reduction_function(self):
        tp = np.sum(self.partial["tp"], axis=0)
        fp = np.sum(self.partial["false_prediction"], axis=0)

        return {"mean": np.mean(tp / (tp + fp + 1e-12))}


class LitMNIST(LightningModule):

    def __init__(self):
        super().__init__()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10)
        )

        self.metric = TML()

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        x = self.model(x)
        x = F.log_softmax(x, dim=1)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)        

        self.metric(
            cat_acc=TmlMetric(
                y,
                logits.argmax(axis=-1),
                metric_class=CategoricalAccuracy,
            )
        )

        return loss


    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)
