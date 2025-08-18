
FROM python:3.12-slim


RUN apt-get update && apt-get install -y gcc libpq-dev iputils-ping && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV TORCH_HOME=/root/.cache/torch
COPY inception_resnet_v1_vggface2.pt /root/.cache/torch/checkpoints/



COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt



CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]

