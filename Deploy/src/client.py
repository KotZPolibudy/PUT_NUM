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

def predict_images(image_paths):
    """Wysyła listę obrazów jako base64-encoded stringi do serwera BentoML."""
    image_data_list = [load_and_encode_image(image_path) for image_path in image_paths]

    with bentoml.SyncHTTPClient("http://localhost:3000") as client:
        response = client.predict(input_data=image_data_list)
        print("Prediction:", response)

if __name__ == "__main__":
    image_paths = ["test_image.jpg", "test_image2.jpg"]  # Można podać więcej obrazów
    predict_images(image_paths)