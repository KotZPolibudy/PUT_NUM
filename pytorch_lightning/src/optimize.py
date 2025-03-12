import json
import subprocess
import optuna

NUM_CONTAINERS = 4  # Ile eksperymentów może działać równocześnie?
N_TRIALS = 20  # Ile testów w sumie?

def objective(trial):
    params = {
        'lr': trial.suggest_loguniform('lr', 1e-5, 1e-1),
        'hidden_units': trial.suggest_int('hidden_units', 64, 512),
        'optimizer_type': trial.suggest_categorical('optimizer_type', ['adam', 'sgd']),
        'activation_function': trial.suggest_categorical('activation_function', ['relu', 'leaky_relu']),
    }
    return params

tasks = [objective(optuna.trial.FixedTrial({})) for _ in range(N_TRIALS)]
with open("tasks.json", "w") as f:
    json.dump(tasks, f)

running_containers = []
while tasks:
    running_containers = [p for p in running_containers if p.poll() is None]
    while len(running_containers) < NUM_CONTAINERS and tasks:
        params = json.dumps(tasks.pop(0))
        process = subprocess.Popen([
            "docker", "run", "--rm", "-e", f"HYPERPARAMS={params}", "dice-ocr"
        ])
        running_containers.append(process)
