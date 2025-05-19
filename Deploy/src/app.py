import torch
import torchvision.transforms as transforms
from PIL import Image
import boto3
import logging
from save_model import DiceClassifier

import io
MODEL_CHECKPOINT = "/opt/final_model.ckpt"  # Ścieżka do checkpointa

# Logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Globalny model
model = None

def load_model():
    global model
    if model is None:
        logger.info("Loading Lightning model from checkpoint...")
        model = DiceClassifier.load_from_checkpoint(MODEL_CHECKPOINT)
        model.eval()
    return model

# Transformacja obrazu (jak w Twoim kodzie)
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def preprocess_image(image: Image.Image) -> torch.Tensor:
    return transform(image).unsqueeze(0)  # [1, 1, 64, 64]

def lambda_handler(event, context):
    try:
        logger.info("Lambda handler started")

        # Dane S3 z eventu
        s3_event = event['Records'][0]['s3']
        bucket_name = s3_event['bucket']['name']
        object_key = s3_event['object']['key']
        logger.info(f"Received file: {bucket_name}/{object_key}")

        # Pobierz obraz z S3
        s3 = boto3.client('s3')
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        image_data = response['Body'].read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Preprocessing
        input_tensor = preprocess_image(image)

        # Inference
        model = load_model()
        with torch.no_grad():
            logits = model(input_tensor)
            prediction = int(torch.argmax(logits, dim=1).item()) + 1

        logger.info(f"Prediction: {prediction}")

        # Zapisz do results bucket
        result_bucket = "oskar-mlops-results"
        result_key = object_key.rsplit(".", 1)[0] + "_prediction.txt"
        s3.put_object(
            Bucket=result_bucket,
            Key=result_key,
            Body=str(prediction).encode("utf-8")
        )
        logger.info(f"Saved to {result_bucket}/{result_key}")

        return {
            "statusCode": 200,
            "body": f"Prediction: {prediction} saved to {result_key}"
        }

    except Exception as e:
        logger.error(f"Error: {e}")
        return {
            "statusCode": 500,
            "body": f"Error: {str(e)}"
        }
