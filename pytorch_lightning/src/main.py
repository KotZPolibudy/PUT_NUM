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

import subprocess
import json

mlflow_logger = MLFlowLogger(
    experiment_name='Dice_Roll_Experiment',
    tracking_uri='http://127.0.0.1:5000',
    save_dir='./mlruns'
)


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
