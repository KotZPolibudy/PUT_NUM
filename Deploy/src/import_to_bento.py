import mlflow
import bentoml
from pathlib import Path

current_date = "2025-05-06"
model_uri = Path("models", f"KotestPath_{current_date}")

# Importowanie modelu MLflow do BentoML
bentoml.mlflow.import_model("kotest", str(model_uri))