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
bentoml serve service.py:Kotest
```

Notatka na teraz:
trzeba sprawdzić service.py czy tam jest dobra struktura danych input/output czy nie trzeba zmienić, bo obecnie jest praktycznie to samo co w demo
no i to czy dobrze wrzytuję obrazek, to też trzeba by sprawdzić bo może tam się walnąłem

Ważne linki:
``` 
https://github.com/bentoml/BentoMLflow/tree/main

https://www.bentoml.com/blog/building-ml-pipelines-with-mlflow-and-bentoml
```
