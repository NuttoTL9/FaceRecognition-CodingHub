from typing import Optional
from fastapi import FastAPI, Form, HTTPException, Depends, File, UploadFile
from pydantic import BaseModel
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from starlette.middleware.cors import CORSMiddleware
from facenet_pytorch import MTCNN, InceptionResnetV1
from databases import Database
from dotenv import load_dotenv
from PIL import Image
from starlette.responses import StreamingResponse

import datetime
import asyncpg
import os
import pytz
import requests
import torch
import io
import base64
import mimetypes

load_dotenv()

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB")
FRAPPE_URL = os.getenv("FRAPPE_URL")
FRAPPE_API_KEY = os.getenv("FRAPPE_API_KEY")
FRAPPE_API_SECRET = os.getenv("FRAPPE_API_SECRET")
MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")


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
            connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
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
        res = requests.post(url, json=data, auth=auth, headers=headers, timeout=30)
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
        res = requests.get(url, params=params, auth=auth, timeout=30)
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

@app.get("/list_employees/")
def list_employees(_=Depends(get_milvus_connection)):
    """
    ดึงรายชื่อ employee_id + name จาก Milvus
    """
    try:
        collection = get_or_create_collection()
        collection.load()

        print("📡 กำลัง query Milvus ...")
        results = collection.query(
            expr="",  
            output_fields=["employee_id", "name"],
            limit=1000
        )
        print("📋 Milvus raw result:", results)

        employees = {}
        for r in results:
            emp_id = r["employee_id"]
            if emp_id not in employees:
                employees[emp_id] = r["name"]

        employee_list = [{"employee_id": k, "name": v} for k, v in employees.items()]
        print(f"✅ Found {len(employee_list)} employees in Milvus")

        return {"status": "success", "employees": employee_list}

    except Exception as e:
        import traceback
        print("❌ Failed to list employees:", e)
        print(traceback.format_exc())   # <<< เพิ่มบรรทัดนี้เพื่อดู stack trace
        raise HTTPException(status_code=500, detail="Failed to list employees")

@app.post("/add_face_to_existing/")
async def add_face_to_existing(
    employee_id: str = Form(...),
    file: UploadFile = File(...),
    _=Depends(get_milvus_connection)
):
    """
    เพิ่ม embedding ใหม่เข้าไปใน employee ที่มีอยู่แล้ว โดยไม่สร้าง employee ใหม่
    """
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file content")

        img = Image.open(io.BytesIO(contents)).convert("RGB")

        face = mtcnn(img)
        if face is None:
            raise HTTPException(status_code=400, detail="No face detected in image")

        face = face.to(device)
        with torch.no_grad():
            embedding = resnet(face.unsqueeze(0)).squeeze().cpu().tolist()

        if len(embedding) != 512:
            raise HTTPException(status_code=500, detail="Invalid embedding length")

        collection = get_or_create_collection()
        collection.load()

        # ดึงชื่อจริงจาก employee_id ที่มีอยู่แล้ว
        expr = f'employee_id == "{employee_id}"'
        result = collection.query(expr=expr, output_fields=["employee_id", "name"])
        if not result:
            raise HTTPException(status_code=404, detail="Employee not found in Milvus")

        name = result[0]["name"]

        # เพิ่ม embedding เข้าไป (ไม่สร้าง employee_id ใหม่)
        collection.insert(
            [
                [employee_id],
                [name],
                [embedding]
            ],
            fields=["employee_id", "name", "embedding"]
        )
        collection.flush()

        return {"status": "success", "employee_id": employee_id, "name": name}

    except Exception as e:
        print("❌ Failed to add face to existing employee:", e)
        raise HTTPException(status_code=500, detail="Failed to add face to existing employee")



@app.post("/log_event_with_snap/")
async def log_event_with_snap(
    employee_id: str = Form(...),
    name: str = Form(...),
    event: str = Form(...),
    snap_file: UploadFile = File(None),
    _=Depends(get_milvus_connection)
):
    bangkok = pytz.timezone("Asia/Bangkok")
    now_local = datetime.datetime.now(bangkok)
    now_local_naive = now_local.replace(tzinfo=None)

    snap_image_b64 = None
    image_filename = None

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
        expr = f'employee_id == "{employee_id}"'
        collection.load()
        result = collection.query(expr=expr, output_fields=["employee_id", "name"])

        real_name = result[0]["name"] if result else name

        query = """
        INSERT INTO face_recog_log (employee_id, name, timestamp, event, snap_image, image_filename)
        VALUES (:employee_id, :name, :timestamp, :event, :snap_image, :image_filename)
        """
        values = {
            "employee_id": employee_id,
            "name": real_name,
            "timestamp": now_local_naive,
            "event": event,
            "snap_image": snap_image_b64,
            "image_filename": image_filename
        }
        await database.execute(query=query, values=values)

        employee_id_check = await check_employee_exists(employee_id)
        if employee_id_check:
            create_employee_checkin(employee_id, event)

        print(f"✅ Log พร้อมภาพบันทึกแล้ว: {employee_id} [{event}] ชื่อจริง: {real_name}")
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
    ดึงภาพจาก log และส่งเป็นรูปภาพจริง (รองรับ image/jpeg, image/png)
    """
    try:
        query = "SELECT snap_image, image_filename FROM face_recog_log WHERE id = :log_id"
        result = await database.fetch_one(query=query, values={"log_id": log_id})

        if not result:
            raise HTTPException(status_code=404, detail="Log not found")

        snap_image_base64 = result["snap_image"]
        filename = result["image_filename"] or "image.jpg"

        if not snap_image_base64:
            raise HTTPException(status_code=404, detail="No image found for this log")

        # แปลง base64 เป็น binary
        image_bytes = base64.b64decode(snap_image_base64)

        # ตรวจสอบ MIME type จากชื่อไฟล์
        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type not in ("image/jpeg", "image/png"):
            mime_type = "application/octet-stream"  # fallback

        return StreamingResponse(io.BytesIO(image_bytes), media_type=mime_type, headers={
            "Content-Disposition": f"inline; filename={filename}"
        })

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

def get_gender_options():
    """ดึง Gender options จาก ERPNext"""
    try:
        res = requests.get(
            f"{FRAPPE_URL}/api/resource/Gender",
            auth=(FRAPPE_API_KEY, FRAPPE_API_SECRET),
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        if res.ok:
            return res.json().get("data", [])
        return []
    except:
        return []

def get_company_options():
    """ดึงรายชื่อบริษัทจาก ERPNext"""
    try:
        res = requests.get(
            f"{FRAPPE_URL}/api/resource/Company",
            auth=(FRAPPE_API_KEY, FRAPPE_API_SECRET),
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        if res.ok:
            companies = res.json().get("data", [])
            # ดึงเฉพาะชื่อบริษัท
            company_names = [company.get("name", "") for company in companies if company.get("name")]
            return company_names
        return []
    except Exception as e:
        print(f"Error fetching companies: {e}")
        return []

def update_employee_image(employee_id: str, image_base64: str):
    """อัพเดตรูปภาพของ Employee ใน ERPNext"""
    try:
        print(f"🔄 กำลังอัพเดตรูปภาพ Employee {employee_id}...")
        print(f"📏 ขนาดรูปภาพ (base64): {len(image_base64)} characters")
        
        # สร้าง payload สำหรับอัพเดตรูปภาพ
        update_payload = {
            "image": image_base64
        }
        
        print(f"📡 ส่งคำขอ PUT ไปยัง: {FRAPPE_URL}/api/resource/Employee/{employee_id}")
        
        update_res = requests.put(
            f"{FRAPPE_URL}/api/resource/Employee/{employee_id}",
            json=update_payload,
            auth=(FRAPPE_API_KEY, FRAPPE_API_SECRET),
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"📡 ERPNext Response Status: {update_res.status_code}")
        
        if update_res.ok:
            print(f"✅ อัพเดตรูปภาพ Employee {employee_id} สำเร็จ")
            print(f"📋 Response: {update_res.text}")
            return True
        else:
            print(f"❌ อัพเดตรูปภาพ Employee {employee_id} ไม่สำเร็จ: {update_res.status_code} {update_res.text}")
            return False
            
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการอัพเดตรูปภาพ Employee: {str(e)}")
        return False

@app.post("/api/resource/Employee")
def create_employee(data: EmployeeCreate, _=Depends(get_milvus_connection)):
    # แปลง gender จากภาษาไทยเป็นภาษาอังกฤษ
    gender_mapping = {
        "ชาย": "Male",
        "หญิง": "Female"
    }

    mapped_gender = gender_mapping.get(data.gender, "Male")

    payload = {
        "first_name": data.firstname,
        "last_name": data.lastname,
        "gender": mapped_gender,
        "date_of_joining": data.date_of_joining,
        "date_of_birth": data.date_of_birth,
        "company": data.company
    }

    try:
        print(f"🔄 ส่งคำขอสร้าง Employee ไปยัง ERPNext: {FRAPPE_URL}")
        print(f"📋 Payload: {payload}")

        res = requests.post(
            f"{FRAPPE_URL}/api/resource/Employee",
            json=payload,
            auth=(FRAPPE_API_KEY, FRAPPE_API_SECRET),
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        print(f"📡 ERPNext Response Status: {res.status_code}")

        if not res.ok:
            print(f"❌ ERPNext API Response: {res.text}")
            raise HTTPException(status_code=res.status_code, detail=f"ERPNext error: {res.text}")

        response_data = res.json()
        employee_id = response_data["data"]["name"]
        fullname = f"{data.firstname} {data.lastname}"
        print(f"✅ Employee Created: {employee_id}")

        return {
            "status": "success",
            "employee_id": employee_id,
            "fullname": fullname
        }

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=500, detail="ERPNext timeout: ไม่สามารถเชื่อมต่อได้ภายใน 30 วินาที")
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=500, detail="ERPNext connection error: ไม่สามารถเชื่อมต่อกับ ERPNext ได้")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดในการสร้าง Employee: {str(e)}")


@app.get("/api/gender-options/")
def get_available_gender_options():
    """ดึง Gender options ที่มีอยู่ใน ERPNext"""
    try:
        gender_options = get_gender_options()
        return {
            "status": "success",
            "gender_options": gender_options,
            "available_options": ["Male", "Female"]  # fallback options
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "available_options": ["Male", "Female"]
        }

@app.get("/api/company-options/")
def get_available_company_options():
    """ดึงรายชื่อบริษัทที่มีอยู่ใน ERPNext"""
    try:
        company_options = get_company_options()
        return {
            "status": "success",
            "company_options": company_options,
            "available_options": company_options if company_options else ["Default Company"]
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "available_options": ["Default Company"] 
        }

