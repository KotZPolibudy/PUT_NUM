import bentoml
import base64
import io
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from bentoml.models import BentoModel

# Definicja obrazu runtime
image_runtime = bentoml.images.PythonImage(python_version="3.11") \
    .python_packages("mlflow", "torch", "torchvision", "Pillow", "numpy")

# Transformacja obrazu
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@bentoml.service(
    image=image_runtime,
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class Kotest:
    # Deklaracja modelu jako atrybut klasy
    bento_model = BentoModel("kotest:latest")

    def __init__(self):
        self.model = bentoml.mlflow.load_model(self.bento_model)

    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        image = Image.open(io.BytesIO(image_data)).convert("L")
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor.numpy()

    @bentoml.api
    def predict(self, input_data: list) -> list:
        """
        Oczekuje wejścia jako lista base64-encoded stringów.
        """
        try:
            processed_images = [self.preprocess_image(base64.b64decode(img)) for img in input_data]
            preds = [self.model.predict(img) for img in processed_images]
            return [pred.tolist() for pred in preds]
        except Exception as e:
            return {"error": str(e)}
