pip install lightning
pip install mlflow
pip install optuna


Zakładając, że jesteś w folderze ./python_lightning w celu ułatwienia do loggowania mlflow:
mlflow ui
python .\src\test.py





Dobra, to teraz instrukcja wersji dockerowej

docker-compose up --scale trainer=0