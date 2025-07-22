import pytest
from unittest.mock import patch, MagicMock, mock_open

import grpc_client.milvus_grpc_utils as milvus_utils

@pytest.fixture
def mock_collection():
    mock = MagicMock()
    field_mock = MagicMock()
    field_mock.name = 'embedding'
    field_mock.dtype = 'FLOAT_VECTOR'
    mock.schema.fields = [field_mock]
    mock.num_entities = 2
    mock.query.return_value = [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]
    return mock


@patch('grpc_client.milvus_grpc_utils.connections.connect')
@patch('grpc_client.milvus_grpc_utils.list_collections')
def test_show_collections(mock_list_collections, mock_connect):
    mock_list_collections.return_value = ['test_collection1', 'test_collection2']
    result = milvus_utils.show_collections()
    mock_connect.assert_called_once_with("default", host=milvus_utils.MILVUS_HOST, port=milvus_utils.MILVUS_PORT)
    assert result == ['test_collection1', 'test_collection2']

@patch('grpc_client.milvus_grpc_utils.Collection')
def test_show_collection_schema(mock_collection_class, mock_collection, capsys):
    mock_collection_class.return_value = mock_collection
    milvus_utils.show_collection_schema("dummy_collection")
    captured = capsys.readouterr()
    assert "Schema for 'dummy_collection':" in captured.out
    # ตอนนี้ field.name เป็น string เลยเช็คได้เลย
    for field in mock_collection.schema.fields:
        assert field.name in captured.out

@patch('grpc_client.milvus_grpc_utils.Collection')
def test_get_vector_from_milvus_found(mock_collection_class, mock_collection):
    mock_collection_class.return_value = mock_collection
    mock_collection.query.return_value = [{"embedding": [0.5, 0.6]}]
    result = milvus_utils.get_vector_from_milvus("dummy_collection", "123")
    assert result == [0.5, 0.6]

@patch('grpc_client.milvus_grpc_utils.Collection')
def test_get_vector_from_milvus_not_found(mock_collection_class, mock_collection):
    mock_collection_class.return_value = mock_collection
    mock_collection.query.return_value = []
    with pytest.raises(ValueError):
        milvus_utils.get_vector_from_milvus("dummy_collection", "123")

@patch('grpc_client.milvus_grpc_utils.Collection')
def test_show_all_ids_with_entities(mock_collection_class, mock_collection, capsys):
    mock_collection_class.return_value = mock_collection
    mock_collection.query.return_value = [{"id": 1}, {"id": 2}]
    milvus_utils.show_all_ids("dummy_collection")
    captured = capsys.readouterr()
    assert "Total entities: 2" in captured.out
    assert "All IDs in collection:" in captured.out

@patch('grpc_client.milvus_grpc_utils.Collection')
def test_show_all_ids_no_entities(mock_collection_class, capsys):
    mock_collection = MagicMock()
    mock_collection.num_entities = 0
    mock_collection_class.return_value = mock_collection
    milvus_utils.show_all_ids("dummy_collection")
    captured = capsys.readouterr()
    assert "No entities found in the collection." in captured.out

@patch('grpc_client.milvus_grpc_utils.grpc.secure_channel')
@patch('grpc_client.milvus_grpc_utils.grpc.ssl_channel_credentials')
@patch('grpc_client.milvus_grpc_utils.image_transform_pb2_grpc.EncodeServiceStub')
@patch('builtins.open', new_callable=mock_open, read_data=b'certdata')
@patch('grpc_client.milvus_grpc_utils.image_transform_pb2.VectorRequest')
def test_encode_vector_with_grpc(
    mock_VectorRequest,
    mock_open,
    mock_Stub,
    mock_ssl_cred,
    mock_secure_channel,
):
    # mock instance stub
    mock_stub_instance = MagicMock()
    mock_stub_instance.EncodeVector.return_value.vector = [0.1] * 512
    mock_Stub.return_value = mock_stub_instance

    vector = [0.0] * 512
    employee_id = "EMP001"
    name = "Test User"

    result = milvus_utils.encode_vector_with_grpc(vector, employee_id, name)

    # Assertions ว่า mock ถูกเรียก
    mock_open.assert_called_once_with('key/server.crt', 'rb')
    mock_ssl_cred.assert_called_once()
    mock_secure_channel.assert_called_once()
    mock_Stub.assert_called_once()
    mock_VectorRequest.assert_called_once_with(name=name, employee_id=employee_id, vector=vector)

    # เช็คค่าผลลัพธ์
    assert isinstance(result, list)
    assert len(result) == 512

