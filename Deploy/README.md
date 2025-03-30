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
known issue:
service oczekuje (wg. kodu) base64 encoded string jako input (obrazek)
client wysyła dokładnie to

error: bentoml oczekuje list_type na wejściu

known issue part 2:

service przerobiony tak aby oczekiwał listy,
klient tak, aby wysyłał odpowiednią listę
ten sam error: Input should be a valid list

Ważne linki:
``` 
https://github.com/bentoml/BentoMLflow/tree/main

https://www.bentoml.com/blog/building-ml-pipelines-with-mlflow-and-bentoml
```
