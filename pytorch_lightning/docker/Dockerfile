# 1. Bazowy obraz z Pythonem i niezbędnymi narzędziami
FROM python:3.10

# 2. Ustawienie katalogu roboczego
WORKDIR /app

# 3. Instalacja zależności systemowych (dla PostgreSQL i PyTorch)
RUN apt-get update && apt-get install -y \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. Skopiowanie plików projektu
COPY requirements.txt requirements.txt

# 5. Instalacja bibliotek Pythona
RUN pip install --no-cache-dir -r requirements.txt

# 6. Skopiowanie całego kodu aplikacji
COPY . .

# 7. Ustawienie zmiennych środowiskowych dla MLflow
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# 8. Domyślna komenda uruchamiania kontenera (zmienisz w `docker-compose.yml`)
CMD ["python", "train.py"]
