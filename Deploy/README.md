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

# AWS
Najważniejsza sprawa `--provenance=false` bez tego docker pushuje na repo 3 obrazy, z których ŻADEN nie jest kompatybilny ze strukturą AWSa. Zastosowanie tej flagi powoduje push 1 obrazu, który jest dokładnie taki jak spodziewa się AWS (oczywiście Amazon nic o tym nie mówi).

Po zalogowaniu się credentialami odpowiednimi dla konta (id oraz region może się zmieniać w zależności, ale dla naszego będzie to zawsze tym)
docker buildx build --platform linux/amd64 --provenance=false --push -t 694676321750.dkr.ecr.eu-central-1.amazonaws.com/kotest-repo:latest .

Aby spushować na podłączony bucket plik:
aws s3 cp .\test_image1.jpg s3://oskar-mlops-2025/


Logowanie + push + update funkcji:
aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin 694676321750.dkr.ecr.eu-central-1.amazonaws.com

docker buildx build --platform linux/amd64 --provenance=false --push -t 694676321750.dkr.ecr.eu-central-1.amazonaws.com/kotest-repo:latest .

aws lambda update-function-code --function-name kotest-lambda-predict --image-uri 694676321750.dkr.ecr.eu-central-1.amazonaws.com/kotest-repo:latest --region eu-central-1
