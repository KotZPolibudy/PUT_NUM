# Dockerfile
ARG PYTHON_VERSION=3.10
ARG PYTORCH_VERSION=2.0
ARG CUDA_VERSION=11.8.0
FROM pytorchlightning/pytorch_lightning:base-cuda-py${PYTHON_VERSION}-torch${PYTORCH_VERSION}-cuda${CUDA_VERSION}

# Ustawienie katalogu roboczego
WORKDIR /app

# Kopiowanie pliku zależności
COPY requirements.txt requirements.txt

# Instalacja zależności Pythona
RUN pip install --no-cache-dir -r requirements.txt

# Kopiowanie kodu aplikacji
COPY . .

# Eksponowanie portu dla ewentualnych usług
EXPOSE 8080

# Domyślna komenda do uruchomienia kontenera
CMD ["python", "main.py"]