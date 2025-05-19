import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
import lightning as L

class DiceClassifier(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.lr = 0.0008945981985897909

        self.activation = nn.LeakyReLU()

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
            nn.Linear(128 * 6 * 6, 174),
            self.activation,
            nn.Linear(174, 8)  # 8 classes
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
