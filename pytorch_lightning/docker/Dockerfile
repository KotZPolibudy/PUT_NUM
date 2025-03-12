FROM pytorchlightning/pytorch_lightning:latest

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

RUN mkdir -p /app/data

EXPOSE 8080

CMD ["python", "test.py"]
