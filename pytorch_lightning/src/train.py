import os
import mlflow
import ray
from ray import tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from dice_classifier import DiceClassifier
from data_module import DiceDataModule
import lightning as L

# Pobranie adresu klastra Ray
ray_address = os.getenv("RAY_ADDRESS", "auto")  # Pobiera RAY_ADDRESS z env

# Ustawienia MLflow
MLFLOW_TRACKING_URI = "http://mlflow:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

num_samples = 5  # ilość prób (NA KAŻDEJ MASZYNIE)

def train_model(config):
    """ Funkcja trenowania modelu dla Ray Tune """
    lr = config["lr"]
    hidden_units = config["hidden_units"]
    optimizer_type = config["optimizer_type"]
    activation_function = config["activation_function"]

    model = DiceClassifier(lr=lr, hidden_units=hidden_units,
                           optimizer_type=optimizer_type, activation_function=activation_function)

    trainer = L.Trainer(max_epochs=10)
    data_module = DiceDataModule()

    trainer.fit(model, data_module)

    # Logowanie do Ray Tune (MLflow zrobi to automatycznie)
    tune.report(loss=min(model.val_losses))

# Definicja przestrzeni hiperparametrów
search_space = {
    "lr": tune.loguniform(1e-5, 1e-1),
    "hidden_units": tune.randint(64, 512),
    "optimizer_type": tune.choice(["adam", "sgd"]),
    "activation_function": tune.choice(["relu", "leaky_relu", "sigmoid"])
}

if __name__ == "__main__":
    # Połącz się z istniejącym klastrem Ray
    ray.init(address=ray_address, ignore_reinit_error=True)

    # Tworzenie MLflow Callback dla Ray Tune
    mlflow_callback = MLflowLoggerCallback(
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name="ray_tune_experiment"
    )

    # Konfiguracja i uruchomienie optymalizacji
    tuner = tune.Tuner(
        tune.with_resources(train_model, resources={"cpu": 2, "gpu": 0}),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            num_samples=num_samples
        ),
        param_space=search_space,
        run_config=ray.air.RunConfig(callbacks=[mlflow_callback])
    )

    tuner.fit()
