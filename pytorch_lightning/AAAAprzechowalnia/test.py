import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam, SGD

import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

import matplotlib.pyplot as plt
import optuna

mlflow_logger = MLFlowLogger(
    experiment_name='Dice_Roll_Experiment',
    tracking_uri='http://127.0.0.1:5000',
    save_dir='./mlruns'
)


class DiceDataModule(L.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=16, image_size=(64, 64)):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.train_data = None
        self.val_data = None
        # self.transform = transforms.Compose([
        #   transforms.Grayscale(),
        #   transforms.Resize(image_size),
        #   transforms.RandomRotation(90),
        #   transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.1),
        #   transforms.RandomResizedCrop(self.image_size, scale=(0.9, 1.1)),
        #   transforms.ToTensor(),
        #   transforms.Normalize((0.5,), (0.5,))
        # ])
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.setup()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)
        train_size = int(0.7 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_data, self.val_data = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=3,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=3, persistent_workers=True)


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

    def on_epoch_end(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss Over Epochs')
        plt.show()


def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    hidden_units = trial.suggest_int('hidden_units', 64, 512)
    optimizer_type = trial.suggest_categorical('optimizer_type', ['adam', 'sgd'])
    activation_function = trial.suggest_categorical('activation_function', ['relu', 'leaky_relu', 'sigmoid'])

    model = DiceClassifier(lr=lr, hidden_units=hidden_units, optimizer_type=optimizer_type,
                           activation_function=activation_function)
    trainer = L.Trainer(
        max_epochs=10,
        logger=mlflow_logger,
        enable_checkpointing=False,
        enable_model_summary=False
    )

    data_module = DiceDataModule()
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())

    return min(model.val_losses)


if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=2)

    print('Best hyperparameters found: ', study.best_params)
    print('Best validation loss: ', study.best_value)

    best_params = study.best_params
    best_model = DiceClassifier(
        lr=best_params['lr'],
        hidden_units=best_params['hidden_units'],
        optimizer_type=best_params['optimizer_type'],
        activation_function=best_params['activation_function']
    )
    trainer = L.Trainer(
        max_epochs=10,
        logger=mlflow_logger,
    )
    data_module = DiceDataModule()
    trainer.fit(best_model, data_module.train_dataloader(), data_module.val_dataloader())
