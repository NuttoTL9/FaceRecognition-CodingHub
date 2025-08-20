import os
from dotenv import load_dotenv
import torch

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

DEVICE = torch.device("cuda:0" if os.getenv("DEVICE") == "cuda" and torch.cuda.is_available() else "cpu")

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

RTSP_RAW = os.getenv("RTSP_URLS", "0")

def _parse_rtsp_urls(raw):
	parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
	urls = []
	for p in parts:
		if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
			p = p[1:-1].strip()
		if p.isdigit():
			urls.append(int(p))
		else:
			urls.append(p)
	return urls

RTSP_URLS = _parse_rtsp_urls(RTSP_RAW)

MIN_LOG_INTERVAL = int(os.getenv("MIN_LOG_INTERVAL", "60"))
MIN_FACE_AREA = int(os.getenv("MIN_FACE_AREA", "8000"))

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "face_vectors")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
LOG_EVENT_URL = os.getenv("LOG_EVENT_URL", "")

GRPC_HOST = os.getenv("GRPC_HOST", "localhost")
GRPC_PORT = int(os.getenv("GRPC_PORT", "50051"))