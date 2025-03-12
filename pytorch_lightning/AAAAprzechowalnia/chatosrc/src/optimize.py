import json
import subprocess
import optuna
import uuid
import os

NUM_CONTAINERS = 4  # Ile eksperymentów równolegle?
N_TRIALS = 20  # Ile testów w sumie?

def objective(trial):
    return {
        'lr': trial.suggest_float('lr', 1e-5, 1e-1, log=True),
        'hidden_units': trial.suggest_int('hidden_units', 64, 512),
        'optimizer_type': trial.suggest_categorical('optimizer_type', ['adam', 'sgd']),
        'activation_function': trial.suggest_categorical('activation_function', ['relu', 'leaky_relu'])
    }

# Generowanie prób optymalizacji
study = optuna.create_study()
study.optimize(objective, n_trials=N_TRIALS)

tasks = [t.params for t in study.trials]
running_containers = []

while tasks or running_containers:
    # Usuwamy zakończone kontenery
    running_containers = [p for p in running_containers if p.poll() is None]

    while len(running_containers) < NUM_CONTAINERS and tasks:
        params = tasks.pop(0)
        param_file = os.path.abspath(f"data/params_{uuid.uuid4().hex}.json")

        with open(param_file, "w") as f:
            json.dump(params, f)

        try:
            process = subprocess.Popen(
                ["docker", "run", "--rm", "-v", f"{param_file}:/app/params.json", "dice-ocr"],
                stderr=subprocess.PIPE
            )
            running_containers.append(process)
        except Exception as e:
            print(f"Błąd podczas uruchamiania kontenera: {e}")

# Oczekiwanie na wszystkie kontenery
for p in running_containers:
    p.wait()
