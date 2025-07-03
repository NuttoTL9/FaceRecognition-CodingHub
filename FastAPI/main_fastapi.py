from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from starlette.middleware.cors import CORSMiddleware
from databases import Database
import datetime
import asyncpg

# ------------------ Configurations ------------------ #
POSTGRES_ADMIN_URL = "postgresql://Nutto:099768180Nn_@192.168.1.27:5432/postgres"
DATABASE_URL = POSTGRES_ADMIN_URL

# ------------------ FastAPI Init ------------------ #
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ------------------ Milvus Init ------------------ #
connections.connect(alias="default", host="192.168.1.27", port="19530")

collection_name = "face_vectors"
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512)
]
schema = CollectionSchema(fields, description="Face Recognition Vectors")

if collection_name not in utility.list_collections():
    collection = Collection(name=collection_name, schema=schema)
    collection.create_index(
        field_name="embedding",
        index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
    )
else:
    collection = Collection(name=collection_name)
collection.load()

# ------------------ Database Init ------------------ #
database = Database(DATABASE_URL)

# ------------------ Helper: สร้าง DB ถ้าไม่มี ------------------ #
async def ensure_table_exists():
    query = """
    CREATE TABLE IF NOT EXISTS face_recog_log (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        event VARCHAR(20) NOT NULL
    )
    """
    conn = await asyncpg.connect(dsn=DATABASE_URL)
    await conn.execute(query)
    await conn.close()
    print("Table face_recog_log is ensured")

# ------------------ Startup & Shutdown Event ------------------ #
@app.on_event("startup")
async def startup():
    await ensure_table_exists()
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

# ------------------ Models ------------------ #
class VectorData(BaseModel):
    name: str
    embedding: list[float]

class LogData(BaseModel):
    name: str
    event: str

# ------------------ API ------------------ #
@app.post("/add_face_vector/")
async def add_face_vector(data: VectorData):
    if not data.name or len(data.embedding) != 512:
        raise HTTPException(status_code=400, detail="ข้อมูลไม่ถูกต้อง")

    try:
        collection.insert([[data.name], [data.embedding]], fields=["name", "embedding"])
        collection.flush()
        print("Inserted:", data.name)
        return {"status": "success"}
    except Exception as e:
        print("Insert failed:", e)
        raise HTTPException(status_code=500, detail="Insert failed")


@app.post("/log_event/")
async def log_event(data: LogData):
    try:
        query = "INSERT INTO face_recog_log (name, timestamp, event) VALUES (:name, :timestamp, :event)"
        values = {
            "name": data.name,
            "timestamp": datetime.datetime.now(),
            "event": data.event
        }
        await database.execute(query=query, values=values)
        print(f"Log saved: {data.name} [{data.event}]")
        return {"status": "logged"}
    except Exception as e:
        print("Log failed:", e)
        raise HTTPException(status_code=500, detail="Log insert failed")
