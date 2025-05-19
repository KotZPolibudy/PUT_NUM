FROM public.ecr.aws/lambda/python:3.11

# Dodaj zależności systemowe potrzebne przez torch/Pillow (to ważne!)
RUN yum install -y \
    libjpeg-turbo-devel \
    zlib-devel \
    freetype-devel \
    && yum clean all

# Zainstaluj wymagania
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Skopiuj model i kod
COPY checkpoints/best_model.ckpt /opt/final_model.ckpt
COPY . .

# Wskazanie handlera (plik app.py, funkcja lambda_handler)
CMD ["app.lambda_handler"]