# 1. Bazowy obraz z Pythonem i niezbędnymi narzędziami
FROM python:3.10

# 2. Ustawienie katalogu roboczego
WORKDIR /app

# 3. Instalacja zależności systemowych (dla PostgreSQL i PyTorch)
RUN apt-get update && apt-get install -y \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. Skopiowanie pliku z zależnościami (teraz ze ścieżką do katalogu głównego projektu!)
COPY ../requirements.txt /app/requirements.txt

# 5. Instalacja bibliotek Pythona
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# 7. Skopiowanie całego kodu aplikacji
COPY ../src/ /app/

# 8. Ustawienie zmiennych środowiskowych dla MLflow
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# 9. Domyślna komenda uruchamiania kontenera (zmienisz w `docker-compose.yml`)
CMD ["python", "train.py"]
