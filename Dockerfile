FROM python:3.12

RUN apt-get update && apt-get install -y gcc libpq-dev iputils-ping && rm -rf /var/lib/apt/lists/*

WORKDIR /code

ENV TORCH_HOME=/root/.cache/torch
COPY inception_resnet_v1_vggface2.pt /root/.cache/torch/checkpoints/



COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app
COPY .env /code/.env


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
