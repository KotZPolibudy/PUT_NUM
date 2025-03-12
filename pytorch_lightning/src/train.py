import os
import json
from pytorch_lightning.src.dice_classifier import DiceClassifier
from data_module import DiceDataModule
import lightning as L

# Odczyt hiperparametrów z ENV
hyperparams = json.loads(os.getenv("HYPERPARAMS", "{}"))

model = DiceClassifier(
    lr=hyperparams.get("lr", 0.001),
    hidden_units=hyperparams.get("hidden_units", 128),
    optimizer_type=hyperparams.get("optimizer_type", "adam"),
    activation_function=hyperparams.get("activation_function", "relu"),
)

data_module = DiceDataModule()

trainer = L.Trainer(max_epochs=10)
trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())
