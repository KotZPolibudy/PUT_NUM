import os
import optuna
from dice_classifier import DiceClassifier
from data_module import DiceDataModule
import lightning as L

# Pobranie zmiennych środowiskowych
STUDY_NAME = os.getenv("STUDY_NAME", "dice_optimization")
DB_PATH = os.getenv("DB_PATH", "sqlite:///optuna.db")
data_dir = "/app/data"

# Funkcja optymalizacyjna
def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    hidden_units = trial.suggest_int("hidden_units", 64, 512)
    optimizer_type = trial.suggest_categorical("optimizer_type", ["adam", "sgd"])
    activation_function = trial.suggest_categorical("activation_function", ["relu", "leaky_relu", "sigmoid"])

    model = DiceClassifier(lr=lr, hidden_units=hidden_units, optimizer_type=optimizer_type,
                           activation_function=activation_function)

    trainer = L.Trainer(max_epochs=10, enable_checkpointing=False, enable_model_summary=False)
    data_module = DiceDataModule(data_dir=data_dir)
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())

    return min(model.val_losses)


# Połączenie z bazą Optuny i uruchomienie optymalizacji
study = optuna.create_study(study_name=STUDY_NAME, direction="minimize", storage=DB_PATH, load_if_exists=True)
# study = optuna.load_study(study_name=STUDY_NAME, storage=DB_PATH)
study.optimize(objective, n_trials=5)  # Każdy kontener wykonuje kilka prób
