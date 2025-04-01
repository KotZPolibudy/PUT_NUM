import bentoml
import torch
import torchvision.transforms as transforms
from PIL import Image
from bentoml.io import Image as BentoImage, JSON
import numpy as np

# Transformacja obrazu
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

model_ref = bentoml.mlflow.get("kotest:latest")
runner = model_ref.to_runner()

svc = bentoml.Service("kotest_service", runners=[runner])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Konwertuje obraz na tensor zgodnie z wymaganiami modelu."""
    image = transform(image).unsqueeze(0)  # Dodanie wymiaru batcha
    return image


@svc.api(input=BentoImage(), output=JSON())
async def predict(image: Image.Image):
    """
    Oczekuje obrazu jako pliku przesłanego w multipart/form-data.
    """
    try:
        processed_image = preprocess_image(image)
        pred = await runner.async_run(processed_image)
        pred_numpy = pred.detach().cpu().numpy()
        # return {"prediction": pred_numpy}
        return {"prediction": int(np.argmax(pred_numpy)) + 1}
    except Exception as e:
        return {"error": str(e)}
