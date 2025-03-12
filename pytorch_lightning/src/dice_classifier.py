import torch
import torch.nn as nn
import lightning as L
from torch.optim import Adam, SGD

class DiceClassifier(L.LightningModule):
    def __init__(self, lr=0.001, hidden_units=128, optimizer_type='adam', activation_function='relu'):
        super().__init__()
        self.lr = lr
        self.optimizer_type = optimizer_type
        self.activation_function = activation_function

        if activation_function == 'relu':
            self.activation = nn.ReLU()
        elif activation_function == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation_function == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            # Default to ReLU
            self.activation = nn.ReLU()

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
            nn.Linear(hidden_units, 8)  # 8 classes for numbers 1–8
        )

        self.train_losses = []
        self.val_losses = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)

        self.train_losses.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)

        self.val_losses.append(loss.item())
        return loss

    def configure_optimizers(self):
        if self.optimizer_type == 'adam':
            optimizer = Adam(self.parameters(), lr=self.lr)
        elif self.optimizer_type == 'sgd':
            optimizer = SGD(self.parameters(), lr=self.lr, momentum=0.9)
        else:
            # Default to Adam
            optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

    """
    def on_epoch_end(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss Over Epochs')
        plt.show()
    """

