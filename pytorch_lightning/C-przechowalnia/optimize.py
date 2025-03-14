import optuna
import subprocess

STUDY_NAME = "dice_optimization"
DB_PATH = "sqlite:///optuna.db"
NUM_CONTAINERS = 4  # Ilość równoległych kontenerów
N_TRIALS = 20  # Ilość prób

# Tworzenie/wczytanie Optuna Study
study = optuna.create_study(study_name=STUDY_NAME, direction="minimize", storage=DB_PATH, load_if_exists=True)

print(f"Uruchamiam {NUM_CONTAINERS} kontenerów...")
print("TESTER")
running_containers = []

for _ in range(NUM_CONTAINERS):
    process = subprocess.Popen(["docker", "run", "--rm",
                                "-e", f"STUDY_NAME={STUDY_NAME}",
                                "-e", f"DB_PATH={DB_PATH}",
                                "dice-ocr"])
    running_containers.append(process)

# Czekamy na zakończenie
for process in running_containers:
    process.wait()

print("DONE")
# Wyniki
print(f"Najlepsze parametry: {study.best_params}")
print(f"Najlepsza wartość straty: {study.best_value}")
