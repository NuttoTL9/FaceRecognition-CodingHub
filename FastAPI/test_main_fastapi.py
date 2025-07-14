import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from httpx import AsyncClient, ASGITransport
from main_fastapi import app


@pytest.mark.asyncio
@patch("main_fastapi.utility.list_collections", return_value=[])
@patch("main_fastapi.Collection")
@patch("main_fastapi.connections.connect", new=MagicMock())  # mock connect จริง
@patch("main_fastapi.get_milvus_connection", new=lambda: None)
async def test_add_face_vector_success(mock_collection_class, mock_list_collections):
    mock_collection = MagicMock()
    mock_collection.create_index.return_value = None
    mock_collection.load.return_value = None
    mock_collection.insert.return_value = None
    mock_collection.flush.return_value = None

    mock_collection_class.return_value = mock_collection

    payload = {"name": "Alice", "embedding": [0.1] * 512}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/add_face_vector/", json=payload)

    assert response.status_code == 200
    assert response.json() == {"status": "success"}
    mock_collection.insert.assert_called_once()
    mock_collection.flush.assert_called_once()


@pytest.mark.asyncio
@patch("main_fastapi.connections.connect", new=MagicMock())
@patch("main_fastapi.get_milvus_connection", new=lambda: None)
async def test_add_face_vector_invalid_embedding():
    payload = {"name": "Alice", "embedding": [0.1] * 500}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/add_face_vector/", json=payload)

    assert response.status_code == 400
    assert "Invalid input" in response.text


@pytest.mark.asyncio
@patch("main_fastapi.connections.connect", new=MagicMock())
@patch("main_fastapi.get_milvus_connection", new=lambda: None)
async def test_add_face_vector_empty_name():
    payload = {"name": "", "embedding": [0.1] * 512}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/add_face_vector/", json=payload)

    assert response.status_code == 400
    assert "Invalid input" in response.text


@pytest.mark.asyncio
@patch("main_fastapi.connections.connect", new=MagicMock())
@patch("main_fastapi.get_milvus_connection", new=lambda: None)
async def test_add_face_vector_invalid_embedding_type():
    payload = {"name": "Bob", "embedding": [None] * 512}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/add_face_vector/", json=payload)

    assert response.status_code in (400, 422)


@pytest.mark.asyncio
@patch("main_fastapi.utility.list_collections", return_value=[])
@patch("main_fastapi.Collection")
@patch("main_fastapi.connections.connect", new=MagicMock())
@patch("main_fastapi.get_milvus_connection", new=lambda: None)
async def test_add_face_vector_milvus_insert_fail(mock_collection_class, mock_list_collections):
    mock_collection = MagicMock()
    mock_collection.create_index.return_value = None
    mock_collection.load.return_value = None
    mock_collection.insert.side_effect = Exception("Milvus insert error")
    mock_collection.flush.return_value = None
    mock_collection_class.return_value = mock_collection

    payload = {"name": "Charlie", "embedding": [0.1] * 512}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/add_face_vector/", json=payload)

    assert response.status_code == 500
    assert "Insert failed" in response.text


@pytest.mark.asyncio
@patch("main_fastapi.database.connect", new_callable=AsyncMock)
@patch("main_fastapi.database.disconnect", new_callable=AsyncMock)
@patch("main_fastapi.database.execute", new_callable=AsyncMock)
@patch("main_fastapi.check_employee_exists", new_callable=AsyncMock)
@patch("main_fastapi.create_employee_checkin")
async def test_log_event_success(mock_checkin, mock_check_emp, mock_exec, mock_disconnect, mock_connect):
    mock_check_emp.return_value = "EMP001"
    mock_exec.return_value = None
    mock_checkin.return_value = True

    payload = {"name": "Alice", "event": "in"}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/log_event/", json=payload)

    assert response.status_code == 200
    assert response.json() == {"status": "logged"}


@pytest.mark.asyncio
@patch("main_fastapi.database.execute", new_callable=AsyncMock)
@patch("main_fastapi.check_employee_exists", new_callable=AsyncMock)
@patch("main_fastapi.create_employee_checkin")
async def test_log_event_erpnext_api_fail(mock_checkin, mock_check_emp, mock_exec):
    mock_exec.return_value = None
    mock_check_emp.side_effect = Exception("ERP API timeout")
    mock_checkin.return_value = True

    payload = {"name": "Dave", "event": "in"}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/log_event/", json=payload)

    assert response.status_code == 200
    assert response.json() == {"status": "logged"}


@pytest.mark.asyncio
@patch("main_fastapi.database.execute", new_callable=AsyncMock)
async def test_log_event_invalid_event(mock_exec):
    payload = {"name": "Eve", "event": ""}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/log_event/", json=payload)

    assert response.status_code == 200 or response.status_code == 422


@pytest.mark.asyncio
@patch("main_fastapi.utility.list_collections", return_value=[])
@patch("main_fastapi.Collection")
@patch("main_fastapi.connections.connect", new=MagicMock())
@patch("main_fastapi.get_milvus_connection", new=lambda: None)
async def test_concurrent_add_face_vector(mock_collection_class, mock_list_collections):
    mock_collection = MagicMock()
    mock_collection.insert.return_value = None
    mock_collection.flush.return_value = None
    mock_collection_class.return_value = mock_collection

    payload = {"name": "ConcurrentUser", "embedding": [0.1] * 512}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        tasks = [ac.post("/add_face_vector/", json=payload) for _ in range(5)]
        responses = await asyncio.gather(*tasks)

    for response in responses:
        assert response.status_code == 200


@pytest.mark.asyncio
@patch("main_fastapi.database.connect", new_callable=AsyncMock)
@patch("main_fastapi.database.disconnect", new_callable=AsyncMock)
@patch("main_fastapi.database.execute", new_callable=AsyncMock)
@patch("main_fastapi.check_employee_exists", new_callable=AsyncMock)
async def test_log_event_no_employee(mock_check_emp, mock_exec, mock_disconnect, mock_connect):
    mock_check_emp.return_value = None
    mock_exec.return_value = None

    payload = {"name": "Ghost", "event": "out"}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/log_event/", json=payload)

    assert response.status_code == 200
    assert response.json() == {"status": "logged"}


@pytest.mark.asyncio
@patch("main_fastapi.database.connect", new_callable=AsyncMock)
@patch("main_fastapi.database.disconnect", new_callable=AsyncMock)
@patch("main_fastapi.database.execute", new_callable=AsyncMock)
async def test_log_event_db_fail(mock_exec, mock_disconnect, mock_connect):
    mock_exec.side_effect = Exception("DB failed")

    payload = {"name": "Alice", "event": "in"}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/log_event/", json=payload)

    assert response.status_code == 500
    assert "Log insert failed" in response.text