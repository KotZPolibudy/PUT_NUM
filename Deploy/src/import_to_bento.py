import mlflow
import bentoml
from pathlib import Path

current_date = "2025-04-01"  # Zmień datę na datę modelu który chcesz wczytać
model_uri = Path("models", f"KotestPath_{current_date}")

# Importowanie modelu do BentoML
bentoml.mlflow.import_model("kotest", model_uri)
