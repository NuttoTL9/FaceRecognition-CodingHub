from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from starlette.middleware.cors import CORSMiddleware
import datetime
import asyncpg
from databases import Database
from dotenv import load_dotenv
import os
import pytz
import requests

load_dotenv()

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB")
FRAPPE_URL = os.getenv("FRAPPE_URL")
FRAPPE_API_KEY = os.getenv("FRAPPE_API_KEY")
FRAPPE_API_SECRET = os.getenv("FRAPPE_API_SECRET")

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

database = Database(DATABASE_URL)

COLLECTION_NAME = "face_vectors"
COLLECTION_SCHEMA = CollectionSchema([
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="employee_id", dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512)
], description="Face Recognition Vectors with employee_id and name")


async def ensure_table_exists():
    query = """
    CREATE TABLE IF NOT EXISTS face_recog_log (
        id SERIAL PRIMARY KEY,
        employee_id VARCHAR(50) NOT NULL,
        name VARCHAR(100) NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        event VARCHAR(20) NOT NULL
    )
    """
    conn = await asyncpg.connect(dsn=DATABASE_URL)
    await conn.execute(query)
    await conn.close()
    print("Table face_recog_log is ensured")


@app.on_event("startup")
async def startup():
    await ensure_table_exists()
    await database.connect()


@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()


class VectorData(BaseModel):
    employee_id: str
    name: str
    embedding: list[float]


class LogData(BaseModel):
    name: str  # Here 'name' holds employee_id (key to log event)
    event: str


def get_milvus_connection():
    try:
        if not connections.has_connection(alias="default"):
            connections.connect(alias="default", host="192.168.1.27", port=19530)
            print("Connected to Milvus")
    except Exception as e:
        print(f"Failed to connect Milvus: {e}")
        raise HTTPException(status_code=500, detail="Cannot connect to Milvus")


def get_or_create_collection():
    if COLLECTION_NAME not in utility.list_collections():
        collection = Collection(name=COLLECTION_NAME, schema=COLLECTION_SCHEMA)
        collection.create_index(
            field_name="embedding",
            index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}},
        )
    else:
        collection = Collection(name=COLLECTION_NAME)
    collection.load()
    return collection


@app.post("/add_face_vector/")
def add_face_vector(data: VectorData, _=Depends(get_milvus_connection)):
    if not data.employee_id.strip():
        raise HTTPException(status_code=400, detail="Invalid input: employee_id is empty")
    if not data.name.strip():
        raise HTTPException(status_code=400, detail="Invalid input: name is empty")
    if len(data.embedding) != 512:
        raise HTTPException(status_code=400, detail="Invalid input: embedding length must be 512")

    try:
        embedding = list(map(float, data.embedding))
    except ValueError as ve:
        print(f"❌ Insert failed: {ve}")
        raise HTTPException(status_code=400, detail="Invalid input: embedding contains invalid value")

    try:
        collection = get_or_create_collection()
        collection.insert([[data.employee_id], [data.name], [embedding]], fields=["employee_id", "name", "embedding"])
        collection.flush()
        print(f"✅ Inserted employee_id={data.employee_id}, name={data.name} to Milvus")
        return {"status": "success"}
    except Exception as e:
        print(f"❌ Insert failed: {e}")
        raise HTTPException(status_code=500, detail="Insert failed")


async def check_employee_exists(employee_id: str) -> str | None:
    url = f"{FRAPPE_URL}/api/resource/Employee"
    params = {
        "filters": f'[["name", "=", "{employee_id}"]]',  # Filter by employee_id key (ERPNext Employee docname)
        "fields": '["name"]'
    }
    auth = (FRAPPE_API_KEY, FRAPPE_API_SECRET)

    try:
        res = requests.get(url, params=params, auth=auth, timeout=5)
        if res.ok:
            data = res.json()
            if data["data"]:
                return data["data"][0]["name"]
        return None

    except Exception as e:
        print("ERPNext API error:", e)
        return None


def create_employee_checkin(employee_id: str, log_type: str):
    url = f"{FRAPPE_URL}/api/resource/Employee Checkin"
    headers = {"Content-Type": "application/json"}
    auth = (FRAPPE_API_KEY, FRAPPE_API_SECRET)

    bangkok = pytz.timezone("Asia/Bangkok")
    now_local = datetime.datetime.now(bangkok)
    now_local_naive = now_local.replace(tzinfo=None)
    time_str = now_local_naive.isoformat()

    data = {
        "employee": employee_id,
        "time": time_str,
        "log_type": log_type.upper(),
        "device_id": "FaceRecogSystem"
    }

    try:
        res = requests.post(url, json=data, auth=auth, headers=headers, timeout=5)
        if res.ok:
            print(f"Employee Checkin saved: {res.json()}")
            return True
        else:
            print("Failed to save Employee Checkin:", res.status_code, res.text)
            return False
    except Exception as e:
        print("ERP request error (Employee Checkin):", e)
        return False


@app.post("/log_event/")
async def log_event(data: LogData, _=Depends(get_milvus_connection)):  # <-- เพิ่ม Depends ตรงนี้
    try:
        collection = get_or_create_collection()
        expr = f'employee_id == "{data.name}"'
        collection.load()
        result = collection.query(expr=expr, output_fields=["employee_id", "name"])

        real_name = result[0]["name"] if result else data.name

        query = """
        INSERT INTO face_recog_log (employee_id, name, timestamp, event)
        VALUES (:employee_id, :name, :timestamp, :event)
        """
        values = {
            "employee_id": data.name,
            "name": real_name,
            "timestamp": datetime.datetime.now(),
            "event": data.event
        }
        await database.execute(query=query, values=values)

        employee_id = await check_employee_exists(data.name)
        if employee_id:
            create_employee_checkin(employee_id, data.event)

        print(f"✅ Log saved: {data.name} [{data.event}] ชื่อจริง: {real_name}")
        return {"status": "logged"}

    except Exception as e:
        print("❌ Log failed:", e)
        raise HTTPException(status_code=500, detail="Log insert failed")
