Obecna instrukcja obsługi to:


Build i odpalenie dwóch:  
docker-compose up -d --build 

Odpalenie jednego:  
docker-compose up -d --scale trainer=1

Odpalenie trzech...  
docker-compose up -d --scale trainer=3



INFO:

Dashboard mlflow i eksperymentów:
http://localhost:5000

Dashboard raytune:
http://localhost:8265

Obecny problem: 
Trainer containers nie łączą się z ray_head który powinien zbierać wyniki optymalizacji,
mimo że ma wystawione porty i wszystko powinno śmigać.

No nic, pozostaje uruchamiać wiele razy wersję "A" bez dockera, bo to łatwiejsze zrównoleglenie niż dockeryzacja.
