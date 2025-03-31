import requests
import os
import io
from PIL import Image


def image_request(service_url, image):
    image_byte = io.BytesIO()
    image.save(image_byte, format="JPEG")
    image_byte.seek(0)
    response = requests.post(service_url, files={"file":("image.jpg", image_byte, "image/jpeg")})
    return response.text


def main():
    service_url = "http://localhost:3000/classify"
    image = Image.open(os.path.join("test_image.jpg"))
    prediction = image_request(service_url=service_url, image=image)
    print(f"Predicted number: {prediction}")


if __name__ == "__main__":
    main()