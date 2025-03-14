#train.py
import optuna
import lightning as L
from dice_classifier import DiceClassifier
from data_module import DiceDataModule

STUDY_NAME = "dice_hyperopt"
STORAGE_URL = "sqlite:///optuna_study.db"

# Funkcja optymalizacyjna
def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    hidden_units = trial.suggest_int('hidden_units', 64, 512)
    optimizer_type = trial.suggest_categorical('optimizer_type', ['adam', 'sgd'])
    activation_function = trial.suggest_categorical('activation_function', ['relu', 'leaky_relu', 'sigmoid'])

    model = DiceClassifier(lr=lr, hidden_units=hidden_units, optimizer_type=optimizer_type,
                           activation_function=activation_function)
    trainer = L.Trainer(max_epochs=10)

    data_module = DiceDataModule()
    trainer.fit(model, data_module)

    return min(model.val_losses)

# Każdy kontener dołącza do wspólnego study i optymalizuje równolegle
if __name__ == "__main__":
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_URL)
    study.optimize(objective, n_trials=5)  # Każdy kontener wykona 5 prób
