import bentoml
import io
from PIL import Image
import torch
import torchvision.transforms as transforms
from bentoml.io import Image as BentoImage, JSON

# Transformacja obrazu
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Definicja serwisu w BentoML
@bentoml.service()
class Kotest:
    def __init__(self):
        self.model = bentoml.mlflow.load_model("kotest:latest")

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        image = transform(image).unsqueeze(0)
        return image

    @bentoml.api(input_spec=BentoImage, output_spec=JSON)
    async def predict(self, image: Image.Image):
        """
        Oczekuje obrazu jako pliku przesłanego w multipart/form-data.
        """
        try:
            processed_image = self.preprocess_image(image)
            pred = self.model(processed_image)
            return {"prediction": pred.tolist()}
        except Exception as e:
            return {"error": str(e)}
