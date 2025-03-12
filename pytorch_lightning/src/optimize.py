import json
import subprocess
import optuna
import uuid

NUM_CONTAINERS = 4  # Ile eksperymentów równolegle?
N_TRIALS = 20  # Ile testów w sumie?


def objective(trial):
    return {
        'lr': trial.suggest_loguniform('lr', 1e-5, 1e-1),
        'hidden_units': trial.suggest_int('hidden_units', 64, 512),
        'optimizer_type': trial.suggest_categorical('optimizer_type', ['adam', 'sgd']),
        'activation_function': trial.suggest_categorical('activation_function', ['relu', 'leaky_relu'])
    }


tasks = [objective(optuna.trial.FixedTrial({})) for _ in range(N_TRIALS)]
running_containers = []

while tasks:
    running_containers = [p for p in running_containers if p.poll() is None]

    while len(running_containers) < NUM_CONTAINERS and tasks:
        params = tasks.pop(0)
        param_file = f"data/params_{uuid.uuid4().hex}.json"
        with open(param_file, "w") as f:
            json.dump(params, f)

        process = subprocess.Popen([
            "docker", "run", "--rm",
            "-v", f"{param_file}:/app/params.json",
            "dice-ocr"
        ])
        running_containers.append(process)
