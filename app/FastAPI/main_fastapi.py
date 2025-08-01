from typing import Optional
from fastapi import FastAPI, Form, HTTPException, Depends, File, UploadFile
from pydantic import BaseModel
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from starlette.middleware.cors import CORSMiddleware
from facenet_pytorch import MTCNN, InceptionResnetV1
from databases import Database
from dotenv import load_dotenv
from PIL import Image

import datetime
import asyncpg
import os
import pytz
import requests
import torch
import io
import base64

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
last_checked_files = set()
COLLECTION_NAME = "face_vectors"
COLLECTION_SCHEMA = CollectionSchema([
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="employee_id", dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512)
], description="Face Recognition Vectors with employee_id and name")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

async def ensure_table_exists():
    # อัพเดตตารางเพื่อรองรับการเก็บภาพ
    query = """
    CREATE TABLE IF NOT EXISTS face_recog_log (
        id SERIAL PRIMARY KEY,
        employee_id VARCHAR(50) NOT NULL,
        name VARCHAR(100) NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        event VARCHAR(20) NOT NULL,
        snap_image TEXT,
        image_filename VARCHAR(255)
    )
    """
    conn = await asyncpg.connect(dsn=DATABASE_URL)
    await conn.execute(query)

    # เพิ่ม columns ใหม่ถ้ายังไม่มี (สำหรับ backward compatibility)
    alter_queries = [
        "ALTER TABLE face_recog_log ADD COLUMN IF NOT EXISTS snap_image TEXT",
        "ALTER TABLE face_recog_log ADD COLUMN IF NOT EXISTS image_filename VARCHAR(255)"
    ]

    for alter_query in alter_queries:
        try:
            await conn.execute(alter_query)
        except Exception as e:
            print(f"Column might already exist: {e}")

    await conn.close()
    print("Table face_recog_log is ensured with image support")


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
    snap_image: str = None  # Base64 encoded image
    image_filename: str = None


class LogDataWithImage(BaseModel):
    name: str
    event: str


class FaceImagePayload(BaseModel):
    employee_id: str
    name: str
    image_base64: str


class DeleteRequest(BaseModel):
    employee_id: str

class EmployeeCreate(BaseModel):
    firstname: str
    lastname: str
    gender: str
    date_of_joining: str
    date_of_birth: str
    company: str

def get_milvus_connection():
    try:
        if not connections.has_connection(alias="default"):
            connections.connect(alias="default", host="milvus-standalone", port=19530)
            print("Connected to Milvus")
    except Exception as e:
        print(f"Failed to connect Milvus: {e}")
        raise HTTPException(status_code=500, detail="Cannot connect to Milvus")


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


def process_image_to_base64(image_data: bytes, max_size: tuple = (400, 400)) -> str:
    """
    ประมวลผลภาพและแปลงเป็น base64 พร้อมปรับขนาด
    """
    try:
        img = Image.open(io.BytesIO(image_data)).convert("RGB")

        # ปรับขนาดภาพเพื่อประหยัดพื้นที่
        img.thumbnail(max_size, Image.Resampling.LANCZOS)

        # แปลงเป็น base64
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        return img_base64
    except Exception as e:
        print(f"Error processing image: {e}")
        return None
    
def insert_face_vector(employee_id: str, name: str, embedding: list[float]):
    if len(embedding) != 512:
        raise ValueError("embedding length must be 512")

    embedding_float = [float(x) for x in embedding]

    collection = get_or_create_collection()
    collection.insert(
        [[employee_id], [name], [embedding_float]],
        fields=["employee_id", "name", "embedding"]
    )
    collection.flush()
    print(f"✅ Inserted employee_id={employee_id}, name={name} to Milvus")


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


@app.post("/log_event/")
async def log_event(data: LogData, _=Depends(get_milvus_connection)):
    bangkok = pytz.timezone("Asia/Bangkok")
    now_local = datetime.datetime.now(bangkok)
    now_local_naive = now_local.replace(tzinfo=None)

    try:
        collection = get_or_create_collection()
        expr = f'employee_id == "{data.name}"'
        collection.load()
        result = collection.query(expr=expr, output_fields=["employee_id", "name"])

        real_name = result[0]["name"] if result else data.name

        query = """
        INSERT INTO face_recog_log (employee_id, name, timestamp, event, snap_image, image_filename)
        VALUES (:employee_id, :name, :timestamp, :event, :snap_image, :image_filename)
        """
        values = {
            "employee_id": data.name,
            "name": real_name,
            "timestamp": now_local_naive,
            "event": data.event,
            "snap_image": data.snap_image,
            "image_filename": data.image_filename
        }
        await database.execute(query=query, values=values)

        employee_id = await check_employee_exists(data.name)
        if employee_id:
            create_employee_checkin(employee_id, data.event)

        print(f"✅ Log saved: {data.name} [{data.event}] ชื่อจริง: {real_name} {'พร้อมภาพ' if data.snap_image else 'ไม่มีภาพ'}")
        return {"status": "logged"}

    except Exception as e:
        print("❌ Log failed:", e)
        raise HTTPException(status_code=500, detail="Log insert failed")


@app.post("/log_event_with_snap/")
async def log_event_with_snap(
    name: str = Form(...),
    event: str = Form(...),
    snap_file: UploadFile = File(None),
    _=Depends(get_milvus_connection)
):
    """
    บันทึก log event พร้อมรูปภาพ snap
    """
    bangkok = pytz.timezone("Asia/Bangkok")
    now_local = datetime.datetime.now(bangkok)
    now_local_naive = now_local.replace(tzinfo=None)

    snap_image_b64 = None
    image_filename = None

    # ประมวลผลภาพถ้ามี
    if snap_file and snap_file.filename:
        try:
            image_data = await snap_file.read()
            if image_data:
                snap_image_b64 = process_image_to_base64(image_data)
                image_filename = snap_file.filename
                print(f"📸 ประมวลผลภาพ {snap_file.filename} สำเร็จ")
        except Exception as e:
            print(f"❌ ประมวลผลภาพล้มเหลว: {e}")

    try:
        collection = get_or_create_collection()
        expr = f'employee_id == "{name}"'
        collection.load()
        result = collection.query(expr=expr, output_fields=["employee_id", "name"])

        real_name = result[0]["name"] if result else name

        query = """
        INSERT INTO face_recog_log (employee_id, name, timestamp, event, snap_image, image_filename)
        VALUES (:employee_id, :name, :timestamp, :event, :snap_image, :image_filename)
        """
        values = {
            "employee_id": name,
            "name": real_name,
            "timestamp": now_local_naive,
            "event": event,
            "snap_image": snap_image_b64,
            "image_filename": image_filename
        }
        await database.execute(query=query, values=values)

        employee_id = await check_employee_exists(name)
        if employee_id:
            create_employee_checkin(employee_id, event)

        print(f"✅ Log พร้อมภาพบันทึกแล้ว: {name} [{event}] ชื่อจริง: {real_name}")
        return {
            "status": "logged",
            "has_image": snap_image_b64 is not None,
            "image_filename": image_filename
        }

    except Exception as e:
        print("❌ Log with snap failed:", e)
        raise HTTPException(status_code=500, detail="Log with snap insert failed")


@app.get("/get_log_image/{log_id}")
async def get_log_image(log_id: int):
    """
    ดึงภาพจาก log โดยใช้ log_id
    """
    try:
        query = "SELECT snap_image, image_filename FROM face_recog_log WHERE id = :log_id"
        result = await database.fetch_one(query=query, values={"log_id": log_id})

        if not result:
            raise HTTPException(status_code=404, detail="Log not found")

        if not result["snap_image"]:
            raise HTTPException(status_code=404, detail="No image found for this log")

        return {
            "image_base64": result["snap_image"],
            "filename": result["image_filename"]
        }

    except Exception as e:
        print(f"❌ Error retrieving image: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve image")


@app.get("/get_recent_logs/")
async def get_recent_logs(limit: int = 10):
    """
    ดึง log ล่าสุดพร้อมข้อมูลว่ามีภาพหรือไม่
    """
    try:
        query = """
        SELECT id, employee_id, name, timestamp, event,
            CASE WHEN snap_image IS NOT NULL THEN true ELSE false END as has_image,
            image_filename
        FROM face_recog_log
        ORDER BY timestamp DESC
        LIMIT :limit
        """
        results = await database.fetch_all(query=query, values={"limit": limit})

        return {"logs": [dict(row) for row in results]}

    except Exception as e:
        print(f"❌ Error retrieving logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve logs")


@app.post("/add_face_image/")
async def add_face_image(
    employee_id: str = Form(...),
    name: str = Form(""),
    fullname: str = Form(""),
    file: UploadFile = File(...),
    _=Depends(get_milvus_connection)
):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file content")

        img = Image.open(io.BytesIO(contents)).convert("RGB")
        print(f"โหลดภาพจากไฟล์ {file.filename} สำเร็จ")

        face = mtcnn(img)
        if face is None:
            raise HTTPException(status_code=400, detail="No face detected in image")

        face = face.to(device)
        with torch.no_grad():
            embedding = resnet(face.unsqueeze(0)).squeeze().cpu().tolist()

        if len(embedding) != 512:
            raise HTTPException(status_code=500, detail="Invalid embedding length")

        collection = get_or_create_collection()
        collection.insert(
            [
                [employee_id],
                [fullname or name],
                [embedding]
            ],
            fields=["employee_id", "name", "embedding"]
        )
        collection.flush()

        return {"status": "success", "employee_id": employee_id, "fullname": fullname}

    except Exception as e:
        print("❌ ล้มเหลวในการประมวลผลภาพใบหน้า:", e)
        raise HTTPException(status_code=500, detail="Failed to process image")


@app.post("/delete_employee_embeddings/")
def delete_employee_embeddings(data: DeleteRequest):
    try:
        if not connections.has_connection(alias="default"):
            connections.connect(alias="default", host="milvus", port=19530)

        collection_name = "face_vectors"
        if collection_name not in utility.list_collections():
            return {"status": "not_found", "detail": "Collection not found"}

        collection = Collection(name=collection_name)

        # คำสั่งลบข้อมูลโดยใช้ expr filter
        expr = f'employee_id == "{data.employee_id}"'
        res = collection.delete(expr=expr)
        collection.flush()

        deleted_count = getattr(res, "delete_count", 0)

        return {"status": "success", "deleted_count": deleted_count}

    except Exception as e:
        print(f"Error deleting embeddings: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete embeddings")
    
@app.post("/api/resource/Employee")
def create_employee(data: EmployeeCreate, _=Depends(get_milvus_connection)):
    payload = {
        "first_name": data.firstname,
        "last_name": data.lastname,
        "gender": data.gender,
        "date_of_joining": data.date_of_joining,
        "date_of_birth": data.date_of_birth,
        "company": data.company
    }

    try:
        res = requests.post(
            f"{FRAPPE_URL}/api/resource/Employee",
            json=payload,
            auth=(FRAPPE_API_KEY, FRAPPE_API_SECRET),
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        if not res.ok:
            print(f"ERPNext API Response: {res.text}")
            raise HTTPException(status_code=res.status_code, detail=f"ERPNext error: {res.text}")

        employee_id = res.json()["data"]["name"]
        fullname = f"{data.firstname} {data.lastname}"
        print(f"✅ Employee Created: {employee_id}")
        return {
            "status": "success",
            "employee_id": employee_id,
            "fullname": fullname
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดในการสร้าง Employee: {str(e)}")