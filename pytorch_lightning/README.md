pip install lightning
pip install mlflow
pip install optuna


Zakładając, że jesteś w folderze ./python_lightning w celu ułatwienia do loggowania mlflow:
mlflow ui
python .\src\test.py





Dobra, to teraz instrukcja wersji dockerowej, równoległej i ogólnie magicznej.

### Krok 1 - build kontenera

docker-compose -f docker/docker-compose.yml build


### Krok 2 - Włącz mlflow

mlflow ui


### Krok 3 - optymalizacja hiperparametrów

python src/optimize.py





Ok, mam pytania, idę dręczyć Janka dzisiaj.
Czy optuna ma być tylko na głównym skrypcie, i odpalać trening pytorch lightning na każdym dockerze, czy optuna ma być na każdym dockerze i mieć rozproszone środowisko do wykonania eksperymentu (zamiast puszczać eksperymenty równolegle z centralizowanego środowiska)