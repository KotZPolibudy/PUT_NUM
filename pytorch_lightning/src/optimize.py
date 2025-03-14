# optimize.py
import optuna
import subprocess
import time

STUDY_NAME = "dice_hyperopt"
STORAGE_URL = "sqlite:///optuna_study.db"
NUM_CONTAINERS = 4  # Ile eksperymentów równolegle?
N_TRIALS = 20  # Ile testów w sumie?

# Tworzymy lub wczytujemy istniejące study
study = optuna.create_study(
    study_name=STUDY_NAME,
    storage=STORAGE_URL,
    direction="minimize",
    load_if_exists=True
)

# Uruchamiamy kontenery równolegle
running_containers = []
for _ in range(NUM_CONTAINERS):
    process = subprocess.Popen(
        ["docker", "run", "--rm", "--network=host", "dice-ocr"],
        stderr=subprocess.PIPE
    )
    running_containers.append(process)

# Czekamy na zakończenie kontenerów
for p in running_containers:
    p.wait()

print("Wszystkie eksperymenty zakończone!")
print("Najlepsze parametry:", study.best_params)
print("Najlepsza wartość:", study.best_value)
