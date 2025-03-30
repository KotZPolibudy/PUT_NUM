import bentoml
import requests
import io
import base64
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

def load_and_preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0)  # Dodanie wymiaru batch
    return image.numpy().tolist()

def predict_image(image_path):
    image_data = load_and_preprocess_image(image_path)
    with bentoml.SyncHTTPClient("http://localhost:3000") as client:
        response = client.predict(input_data=image_data)
        print("Prediction:", response)

if __name__ == "__main__":
    image_path = "test_image.png"
    predict_image(image_path)
