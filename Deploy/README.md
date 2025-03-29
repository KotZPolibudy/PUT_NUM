Uruchomienie mlflow:

mlflow server --host 127.0.0.1 --port 8080


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
bentoml serve service.py:IrisClassifier
```

Dlaczego nie działa? 
Jeszcze nie wiem, ale to pierwszy commit itd więc rzucam żebyś coś widział.


Ważne linki:
``` 
https://github.com/bentoml/BentoMLflow/tree/main

https://www.bentoml.com/blog/building-ml-pipelines-with-mlflow-and-bentoml
```
