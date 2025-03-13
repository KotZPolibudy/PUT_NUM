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
import json
import subprocess
import uuid
import os
from dice_classifier import DiceClassifier
from data_module import DiceDataModule

NUM_CONTAINERS = 4  # Ile eksperymentów równolegle?
N_TRIALS = 8  # Ile testów w sumie?


mlflow_logger = MLFlowLogger(
    experiment_name='Dice_Roll_Experiment',
    tracking_uri='http://127.0.0.1:5000',
    save_dir='../mlruns'
)


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
    study.optimize(objective, n_trials=N_TRIALS)

    tasks = [t.params for t in study.trials]
    running_containers = []

    while tasks or running_containers:
        running_containers = [p for p in running_containers if p.poll() is None]

        while len(running_containers) < NUM_CONTAINERS and tasks:
            # Zrzut parametrów do pliku json z odpowiednim uuid
            params = tasks.pop(0)
            param_file = os.path.abspath(f"data/params_{uuid.uuid4().hex}.json")
            with open(param_file, "w") as f:
                json.dump(params, f)

            # Uruchomienie kontenera, który włączy train.py i zaczyta odpowiednie parametry z pliku json
            try:
                process = subprocess.Popen(
                    ["docker", "run", "--rm", "-v", f"{param_file}:/app/params.json", "dice-ocr"],
                    stderr=subprocess.PIPE
                )
                running_containers.append(process)
            except Exception as e:
                print(f"Błąd podczas uruchamiania kontenera: {e}")

    # Oczekiwanie na ostatnie kontenery
    for p in running_containers:
        p.wait()

    # Print best found params
    best_params = study.best_params
    print('Best hyperparameters found: ', best_params)
    print('Best validation loss: ', study.best_value)

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
