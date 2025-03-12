pip install lightning
pip install mlflow
pip install optuna


Zakładając, że jesteś w folderze ./python_lightning w celu ułatwienia do loggowania mlflow:
mlflow ui
python .\src\test.py





Dobra, to teraz instrukcja wersji dockerowej, równoległej i ogólnie magicznej.

### Krok 1

docker-compose -f docker/docker-compose.yml build

### Krok 2
(optymalizacja hiperparametrów)

python src/optimize.py


Potencjalnie, zamiast tego:

docker run --rm -v $(pwd)/data:/app/data dice-ocr 

ale to dla trenowania jednego konkretnego modelu. (NIE TESTOWANE!!!)