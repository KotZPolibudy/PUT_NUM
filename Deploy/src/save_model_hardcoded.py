from pathlib import Path
import mlflow.sklearn
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam, SGD
import lightning as L
from datetime import datetime
from save_model import DiceClassifier, DiceDataModule

def get_unique_model_uri(base_uri):
    if base_uri.exists():
        if any(base_uri.iterdir()):  # Jeśli folder nie jest pusty
            counter = 1
            while True:
                new_uri = base_uri.with_name(f"{base_uri.stem}_{counter}{base_uri.suffix}")
                if not new_uri.exists() or not any(new_uri.iterdir()):
                    return new_uri
                counter += 1
        else:
            print(f"Folder {base_uri} już istnieje, ale jest pusty. Można zapisać model.")
            return base_uri
    return base_uri

if __name__ == "__main__":
    best_params = {'lr': 0.0008945981985897909,
                   'hidden_units': 174,
                   'optimizer_type': 'adam',
                   'activation_function': 'leaky_relu'}

    best_model = DiceClassifier(
        lr=best_params['lr'],
        hidden_units=best_params['hidden_units'],
        optimizer_type=best_params['optimizer_type'],
        activation_function=best_params['activation_function']
    )

    trainer = L.Trainer(max_epochs=10)
    data_module = DiceDataModule()
    trainer.fit(best_model, data_module.train_dataloader(), data_module.val_dataloader())

    current_date = datetime.now().strftime("%Y-%m-%d")
    model_uri = Path("models", f"KotestPath_{current_date}")
    unique_model_uri = get_unique_model_uri(model_uri)
    unique_model_uri.mkdir(parents=True, exist_ok=True)

    mlflow.sklearn.save_model(best_model, str(unique_model_uri.resolve()))
