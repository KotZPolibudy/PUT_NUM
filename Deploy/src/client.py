import bentoml
import base64
import io
from PIL import Image
import requests


def load_and_encode_image(image_path):
    """Ładuje obraz i koduje go do base64."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


def predict_image(image_path):
    """Wysyła obraz jako base64-encoded string do serwera BentoML."""
    image_data = load_and_encode_image(image_path)

    with bentoml.SyncHTTPClient("http://localhost:3000") as client:
        response = client.predict(input_data=image_data)
        print("Prediction:", response)


if __name__ == "__main__":
    image_path = "test_image.jpg"
    predict_image(image_path)