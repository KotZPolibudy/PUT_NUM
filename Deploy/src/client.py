import requests
import io
from PIL import Image


def image_request(service_url, image):
    image_byte = io.BytesIO()
    image.save(image_byte, format="JPEG")
    image_byte.seek(0)

    response = requests.post(service_url, files={"image": ("image.jpg", image_byte, "image/jpeg")})
    return response.json()


def main():
    service_url = "http://localhost:3000/predict"

    image = Image.open("test_image1.jpg")
    prediction = image_request(service_url=service_url, image=image)

    print(f"Predicted number: {prediction}")


if __name__ == "__main__":
    main()
