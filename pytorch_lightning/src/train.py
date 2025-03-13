import os
import json
from dice_classifier import DiceClassifier
from data_module import DiceDataModule
import lightning as L

param_file = "params.json"
with open(param_file, "r") as f:
    hyperparams = json.load(f)

print(hyperparams)  # Debug

model = DiceClassifier(**hyperparams)
data_module = DiceDataModule()

trainer = L.Trainer(max_epochs=10)
trainer.fit(model, data_module)
