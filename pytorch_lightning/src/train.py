import os
import mlflow
import ray
from ray import tune
from ray.tune.integration.mlflow import mlflow_mixin
from dice_classifier import DiceClassifier
from data_module import DiceDataModule
import lightning as L

# Ustawienia MLflow
MLFLOW_TRACKING_URI = "http://mlflow:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

@mlflow_mixin
def train_model(config):
    lr = config["lr"]
    hidden_units = config["hidden_units"]
    optimizer_type = config["optimizer_type"]
    activation_function = config["activation_function"]

    model = DiceClassifier(lr=lr, hidden_units=hidden_units,
                           optimizer_type=optimizer_type, activation_function=activation_function)

    trainer = L.Trainer(max_epochs=10)
    data_module = DiceDataModule()

    trainer.fit(model, data_module)

    # Logowanie wyniku do Ray Tune
    tune.report(loss=min(model.val_losses))

# Definicja zakresu poszukiwań
search_space = {
    "lr": tune.loguniform(1e-5, 1e-1),
    "hidden_units": tune.randint(64, 512),
    "optimizer_type": tune.choice(["adam", "sgd"]),
    "activation_function": tune.choice(["relu", "leaky_relu", "sigmoid"])
}

# Uruchamianie optymalizacji
tuner = tune.Tuner(
    tune.with_resources(train_model, resources={"cpu": 2, "gpu": 0}),
    tune_config=tune.TuneConfig(
        metric="loss",
        mode="min",
        num_samples=10  # Liczba prób
    ),
    param_space=search_space
)

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    tuner.fit()
