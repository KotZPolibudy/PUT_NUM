FROM pytorchlightning/pytorch_lightning:latest

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "train.py"]
