import bentoml
import io
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from bentoml.models import BentoModel
from bentoml.io import Image as BentoImage

# Transformacja obrazu
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@bentoml.service()
class Kotest:
    bento_model = BentoModel("kotest:latest")

    def __init__(self):
        self.model = bentoml.mlflow.load_model(self.bento_model)

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        image = transform(image).unsqueeze(0)
        return image

    @bentoml.api(input=BentoImage(), output=bentoml.io.JSON())
    def predict(self, image: Image.Image):
        """
        Oczekuje obrazu jako pliku przesłanego w multipart/form-data.
        """
        try:
            processed_image = self.preprocess_image(image)
            pred = self.model.predict(processed_image)
            return {"prediction": pred.tolist()}
        except Exception as e:
            return {"error": str(e)}
