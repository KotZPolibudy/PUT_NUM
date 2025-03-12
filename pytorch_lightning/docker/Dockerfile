# Dockerfile
FROM pytorchlightning/pytorch_lightning:latest

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

COPY . .

CMD ["python", "train.py"]
