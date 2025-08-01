import os
from dotenv import load_dotenv
import torch

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

# Device
DEVICE = torch.device("cuda:0" if os.getenv("DEVICE") == "cuda" and torch.cuda.is_available() else "cpu")

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# RTSP URLs
RTSP_RAW = os.getenv("RTSP_URLS", "0")
RTSP_URLS = [int(RTSP_RAW)] if RTSP_RAW.isdigit() else RTSP_RAW.split(",")

# Milvus / Face recognition settings
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "face_vectors")
MIN_LOG_INTERVAL = int(os.getenv("MIN_LOG_INTERVAL", "60"))
MIN_FACE_AREA = int(os.getenv("MIN_FACE_AREA", "8000"))

# Milvus Server
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

# Log API
LOG_EVENT_URL = os.getenv("LOG_EVENT_URL", "")

# gRPC settings
GRPC_HOST = os.getenv("GRPC_HOST", "localhost")
GRPC_PORT = int(os.getenv("GRPC_PORT", "50051"))