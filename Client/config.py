import os
import torch
from dotenv import load_dotenv

load_dotenv()

# Device
DEVICE = torch.device("cuda:0" if os.getenv("DEVICE") == "cuda" and torch.cuda.is_available() else "cpu")

# กล้อง
RTSP_RAW = os.getenv("RTSP_URLS", "0")
RTSP_URLS = [int(RTSP_RAW)] if RTSP_RAW.isdigit() else RTSP_RAW.split(",")

# Milvus / Face recognition settings
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "face_vectors")
MIN_LOG_INTERVAL = int(os.getenv("MIN_LOG_INTERVAL", "60"))
MIN_FACE_AREA = int(os.getenv("MIN_FACE_AREA", "8000"))

# Milvus Server
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))

# Log API
LOG_EVENT_URL = os.getenv("LOG_EVENT_URL", "")
