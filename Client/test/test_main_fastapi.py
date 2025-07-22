import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

# import app จากโมดูลหลักของคุณ
from FastAPI.main_fastapi import app, check_employee_exists, create_employee_checkin, get_milvus_connection, get_or_create_collection

client = TestClient(app)

# ตัวอย่างข้อมูล embedding 512 มิติ (เต็ม)
example_embedding = [0.1] * 512

# --- Mock Milvus connection and collection ---
class DummyCollection:
    def insert(self, data, fields=None):
        # Method intentionally left empty because this is a dummy collection for testing.
        pass
    def flush(self):
        # Method intentionally left empty because this is a dummy collection for testing.
        pass
    def load(self):
        # Method intentionally left empty because this is a dummy collection for testing.
        pass
    def query(self, expr=None, output_fields=None):
        # ตัวอย่างผลลัพธ์ query
        return [{"employee_id": "emp123", "name": "John Doe"}]

# Mock get_or_create_collection ให้คืน DummyCollection
@pytest.fixture(autouse=True)
def mock_milvus(monkeypatch):
    monkeypatch.setattr("FastAPI.main_fastapi.get_or_create_collection", lambda: DummyCollection())
    monkeypatch.setattr("FastAPI.main_fastapi.get_milvus_connection", lambda: None)

@pytest.fixture(autouse=True)
def mock_database(monkeypatch):
    mock_db = AsyncMock()
    monkeypatch.setattr("FastAPI.main_fastapi.database.execute", mock_db)
    monkeypatch.setattr("FastAPI.main_fastapi.database.connect", AsyncMock())
    monkeypatch.setattr("FastAPI.main_fastapi.database.disconnect", AsyncMock())
    return mock_db

@pytest.fixture(autouse=True)
def mock_check_employee(monkeypatch):
    monkeypatch.setattr("FastAPI.main_fastapi.check_employee_exists", AsyncMock(return_value="emp123"))
    monkeypatch.setattr("FastAPI.main_fastapi.create_employee_checkin", lambda eid, event: True)

def test_add_face_vector_success():
    payload = {
        "employee_id": "emp123",
        "name": "John Doe",
        "embedding": example_embedding
    }
    response = client.post("/add_face_vector/", json=payload)
    assert response.status_code == 200 or response.status_code == 201 or response.status_code == 204 or response.status_code == 200
    # ไม่มี response body ในโค้ดจริง ถ้าอยากเพิ่ม return {"status": "success"} ใน add_face_vector จะช่วยให้ทดสอบได้ง่ายขึ้น

@pytest.mark.asyncio
def test_log_event_success(mock_database):
    payload = {
        "name": "emp123",
        "event": "IN"
    }
    response = client.post("/log_event/", json=payload)
    assert response.status_code == 200
    assert response.json() == {"status": "logged"}
    mock_database.assert_awaited()

def test_add_face_vector_invalid_embedding():
    payload = {
        "employee_id": "emp123",
        "name": "John Doe",
        "embedding": [0.1] * 10  # ผิดขนาด
    }
    response = client.post("/add_face_vector/", json=payload)
    assert response.status_code == 400
    assert "embedding length must be 512" in response.json()["detail"]

def test_add_face_vector_empty_name():
    payload = {
        "employee_id": "emp123",
        "name": "",
        "embedding": example_embedding
    }
    response = client.post("/add_face_vector/", json=payload)
    assert response.status_code == 400
    assert "name is empty" in response.json()["detail"]

def test_add_face_vector_empty_employee_id():
    payload = {
        "employee_id": " ",
        "name": "John Doe",
        "embedding": example_embedding
    }
    response = client.post("/add_face_vector/", json=payload)
    assert response.status_code == 400
    assert "employee_id is empty" in response.json()["detail"]

    
def test_get_or_create_collection_creates(monkeypatch):
    # Fake no collections so creation branch triggers
    monkeypatch.setattr("FastAPI.main_fastapi.utility.list_collections", lambda: [])

    # Create a dummy Collection class that mocks constructor & methods
    class DummyCollection:
        def __init__(self, *args, **kwargs):
            pass
        def create_index(self, **kwargs):
            return None
        def load(self):
            return None

    # Patch Collection with DummyCollection
    monkeypatch.setattr("FastAPI.main_fastapi.Collection", DummyCollection)

    # Call the function (uses the DummyCollection)
    collection = get_or_create_collection()
    assert collection is not None


def test_get_or_create_collection_existing(monkeypatch):
    monkeypatch.setattr("FastAPI.main_fastapi.utility.list_collections", lambda: ["face_vectors"])

    class DummyCollection:
        def __init__(self, *args, **kwargs):
            
            pass
        def load(self):
            return None

    monkeypatch.setattr("FastAPI.main_fastapi.Collection", DummyCollection)
    collection = get_or_create_collection()
    assert collection is not None

def test_get_milvus_connection_success(monkeypatch):
    monkeypatch.setattr("FastAPI.main_fastapi.connections.has_connection", lambda alias: False)
    monkeypatch.setattr("FastAPI.main_fastapi.connections.connect", lambda **kwargs: None)
    get_milvus_connection()  # ไม่ควร error

def test_get_milvus_connection_failure(monkeypatch):
    def raise_exc(*args, **kwargs):
        raise ValueError("fail")
    monkeypatch.setattr("FastAPI.main_fastapi.connections.has_connection", lambda alias: False)
    monkeypatch.setattr("FastAPI.main_fastapi.connections.connect", raise_exc)
    with pytest.raises(Exception):
        get_milvus_connection()

@pytest.mark.asyncio
async def test_check_employee_exists_success(monkeypatch):
    class DummyResponse:
        ok = True
        def json(self): 
            return {"data": [{"name": "emp123"}]}
    monkeypatch.setattr("FastAPI.main_fastapi.requests.get", lambda *args, **kwargs: DummyResponse())
    emp = await check_employee_exists("emp123")
    assert emp == "emp123"

@pytest.mark.asyncio
async def test_check_employee_exists_fail(monkeypatch):
    class DummyResponse:
        ok = False
    monkeypatch.setattr("FastAPI.main_fastapi.requests.get", lambda *args, **kwargs: DummyResponse())
    emp = await check_employee_exists("emp123")
    assert emp is None

def test_create_employee_checkin_success(monkeypatch):
    class DummyResponse:
        ok = True
        def json(self): return {"result": "ok"}
    monkeypatch.setattr("FastAPI.main_fastapi.requests.post", lambda *args, **kwargs: DummyResponse())
    res = create_employee_checkin("emp123", "IN")
    assert res is True

def test_create_employee_checkin_fail(monkeypatch):
    class DummyResponse:
        ok = False
        status_code = 400
        text = "error"
    monkeypatch.setattr("FastAPI.main_fastapi.requests.post", lambda *args, **kwargs: DummyResponse())
    res = create_employee_checkin("emp123", "IN")
    assert res is False

def test_create_employee_checkin_exception(monkeypatch):
    def raise_exc(*args, **kwargs):
        raise ValueError("fail")
    monkeypatch.setattr("FastAPI.main_fastapi.requests.post", raise_exc)
    res = create_employee_checkin("emp123", "IN")
    assert res is False
