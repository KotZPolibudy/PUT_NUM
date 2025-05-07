import bentoml
import boto3
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io

# Inicjalizacja modelu
model_ref = bentoml.mlflow.get("kotest:latest")
runner = model_ref.to_runner()
runner.init_local()  # W Lambda nadal wymagane do uruchomienia lokalnego runnera

# Przetwarzanie obrazu jak wcześniej
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def preprocess_image(image: Image.Image) -> torch.Tensor:
    image = transform(image).unsqueeze(0)
    return image

def lambda_handler(event, context):
    try:
        print("Lambda handler started")
        # Dane z eventu S3
        s3_event = event['Records'][0]['s3']
        bucket_name = s3_event['bucket']['name']
        object_key = s3_event['object']['key']

        # Pobierz obraz z S3
        s3 = boto3.client('s3')
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        image_data = response['Body'].read()

        # Otwórz jako obraz
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        processed_image = preprocess_image(image)

        # Predykcja
        pred = runner.predict.run(processed_image)
        prediction = int(np.argmax(pred.detach().cpu().numpy())) + 1

        # Zapisz wynik z powrotem do S3 w odpowiednim bucketcie (oskar-mlops-results)
        result_bucket = "oskar-mlops-results"  # Bucket wynikowy
        result_key = object_key.rsplit(".", 1)[0] + "_prediction.txt"
        s3.put_object(
            Bucket=result_bucket,
            Key=result_key,
            Body=str(prediction).encode("utf-8")
        )

        return {
            "statusCode": 200,
            "body": f"Prediction: {prediction} saved to {result_key}"
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": f"Error: {str(e)}"
        }