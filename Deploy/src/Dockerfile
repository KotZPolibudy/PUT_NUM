FROM public.ecr.aws/lambda/python:3.11

# Zainstaluj wymagania
COPY requirements.txt .
RUN pip install -r requirements.txt

# Skopiuj kod
COPY . .

# Wskazanie handlera
CMD ["app.lambda_handler"]
