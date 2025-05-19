import torch
import torchvision.transforms as transforms
import boto3
import logging
from model import DiceClassifier  # Upewnij się, że masz odpowiednią ścieżkę do modelu
import tempfile
from PIL import Image

import io
MODEL_CHECKPOINT = "/opt/final_model.ckpt"  # Ścieżka do checkpointa

s3 = boto3.client('s3')
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

def preprocess_image(image_path) -> torch.Tensor:
    logger.info(f"Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image).unsqueeze(0)  # [1, 1, 64, 64]

# Funkcja do pobrania obrazu z S3
def download_from_s3(bucket, key, tmp_path):
    logger.info(f"Downloading image from S3: {bucket}/{key}")
    s3.download_file(bucket, key, tmp_path)
    logger.info(f"Image downloaded to: {tmp_path}")

def lambda_handler(event, context):
    try:
        logger.info("Lambda handler started")

        # Dane S3 z eventu
        s3_event = event['Records'][0]['s3']
        bucket_name = s3_event['bucket']['name']
        object_key = s3_event['object']['key']
        logger.info(f"Received file: {bucket_name}/{object_key}")

        # Inference
        model = load_model()
        with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_file:
            download_from_s3(bucket_name, object_key, temp_file.name)
            input_tensor = preprocess_image(temp_file.name)
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
            Body=str(prediction)
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
