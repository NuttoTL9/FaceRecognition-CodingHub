from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import uuid
import numpy as np
from starlette.middleware.cors import CORSMiddleware
app = FastAPI()
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
# เชื่อมต่อกับ Milvus
connections.connect(alias="default", host="192.168.1.27", port="19530")

# สร้าง Schema Milvus หากยังไม่มี
collection_name = "face_vectors"

fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36),
    FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=100, is_indexed=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512)
]

schema = CollectionSchema(fields, description="Face Recognition Vectors")

# ลบ collection เดิม (ถ้ามี)
# if collection_name in utility.list_collections():
#     utility.drop_collection(collection_name)

# สร้าง Collection ใหม่
collection = Collection(name=collection_name, schema=schema)
collection.create_index(
    field_name="embedding",
    index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
)
collection.load()

# Model รับข้อมูลจาก Client
class VectorData(BaseModel):
    vector: list[float]

@app.post("/add_face_vector/")
async def add_face_vector(request: Request):
    data = await request.json()
    id_ = data.get("id") or str(uuid.uuid4())
    name = data.get("name")
    embedding = data.get("embedding")
    if not name or not isinstance(embedding, list) or len(embedding) != 512:
        return {"status": "error", "message": "ข้อมูลไม่ถูกต้อง"}
    collection.insert(id=[id_], name=[name], embedding=[embedding])
    return {"status": "success", "id": id_}
