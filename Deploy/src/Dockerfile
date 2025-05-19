# FROM public.ecr.aws/lambda/python:3.10
FROM kotest-repo:latest

# Skopiuj wymagania i zainstaluj tylko potrzebne biblioteki
# COPY requirements.txt ${LAMBDA_TASK_ROOT}
# RUN pip install --upgrade pip
# RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

RUN pip install --no-cache-dir torch torchvision lightning boto3


# Skopiuj wytrenowany model i aplikację
COPY checkpoints/best_model.ckpt /opt/final_model.ckpt
COPY app.py ${LAMBDA_TASK_ROOT}
COPY model.py ${LAMBDA_TASK_ROOT}

# Określ punkt wejścia funkcji Lambda
CMD ["app.lambda_handler"]