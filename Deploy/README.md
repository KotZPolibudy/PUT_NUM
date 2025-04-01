Uruchomienie mlflow:
```
mlflow server --host 127.0.0.1 --port 8080
```

Odpalasz save_model.py

następnie przyda się sprawdzenie czy model się poprawnie wgrał.
```
bentoml models list
```

Jeśli będziesz trenował dużo razy, możesz chcieć używać:
```
bentoml models delete <name>
``` 

Obecnie wgrany jest przykład na iris, więc powinno się udać z:
```
bentoml serve demoservice.py:Kotest
```
Upewnij się że jesteś w PUT_NUM/Deploy/demo!!!

Potem uruchamiając client.py jesteś w stanie wrzucać zapytania.

Teraz, zmieniając miejsce na:
PUT_NUM/Deploy/src

i tutaj już nie demoservice.py tylko service.py
```
bentoml serve service.py:svc
```

teraz już wystarczy uruchomić klienta albo jakkolwiek inaczej wysłać zapytanie do serwisu bento i mamy to ;)


Ważny szczegół!
Model powinien być tworzony i zapisywany na tej samej wersji pythona, na której potem będziemy tworzyć z niego serwis, bo jak się okazuje model zapisany za pomocą python 3.10 nie był kompatybilny z serwisem BentoML na pythonie 3.12 na innej maszynie.
(chodziło również o niezgodności w używanych libkach, jak inna wersja numpy etc.)

Ważne linki:
``` 
https://github.com/bentoml/BentoMLflow/tree/main
https://www.bentoml.com/blog/building-ml-pipelines-with-mlflow-and-bentoml
```
Honorable mentions: 
 - Hindusi z youtube'a
 - dokumentacja BentoML
