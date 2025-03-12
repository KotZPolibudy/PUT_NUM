import os
import json
import sys
from dice_model import DiceClassifier, DiceDataModule

import lightning as L
from lightning.pytorch.loggers import MLFlowLogger

params = json.loads(os.environ["HYPERPARAMS"])

mlflow_logger = MLFlowLogger(
    experiment_name="Dice_Roll_Experiment",
    tracking_uri="http://127.0.0.1:5000",
    save_dir="./mlruns"
)

model = DiceClassifier(
    lr=params["lr"],
    hidden_units=params["hidden_units"],
    optimizer_type=params["optimizer_type"],
    activation_function=params["activation_function"]
)

trainer = L.Trainer(
    max_epochs=10,
    logger=mlflow_logger,
    enable_checkpointing=False,
    enable_model_summary=False
)

data_module = DiceDataModule()
trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())

# Zwracamy minimalną stratę walidacyjną
print(min(model.val_losses))
sys.exit(0)  # Kończymy kontener
