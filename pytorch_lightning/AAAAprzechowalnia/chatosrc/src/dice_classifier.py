import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import Adam, SGD


class DiceClassifier(L.LightningModule):
    def __init__(self, lr=0.001, hidden_units=128, optimizer_type='adam', activation_function='relu'):
        super().__init__()
        self.lr = lr
        self.optimizer_type = optimizer_type
        self.activation_function = activation_function

        activation_functions = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'sigmoid': nn.Sigmoid()
        }
        self.activation = activation_functions.get(activation_function, nn.ReLU())

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            self.activation,
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3),
            self.activation,
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3),
            self.activation,
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(128 * 6 * 6, hidden_units),
            self.activation,
            nn.Linear(hidden_units, 8)  # 8 klas
        )

    def forward(self, x):
        return F.log_softmax(self.model(x), dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizers = {
            'adam': Adam(self.parameters(), lr=self.lr),
            'sgd': SGD(self.parameters(), lr=self.lr, momentum=0.9)
        }
        return optimizers.get(self.optimizer_type, Adam(self.parameters(), lr=self.lr))
