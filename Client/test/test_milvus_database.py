import pytest
from unittest.mock import patch, MagicMock
import torch
from database.milvus_database import create_milvus_collection, load_face_database
from config import DEVICE

@patch("database.milvus_database.utility.list_collections", return_value=[])
@patch("database.milvus_database.Collection")
def test_create_milvus_collection_creates_new_collection(mock_collection_cls, mock_list_collections):
    mock_collection_instance = MagicMock()
    mock_collection_cls.return_value = mock_collection_instance

    collection = create_milvus_collection()

    mock_list_collections.assert_called_once()
    mock_collection_cls.assert_called_once()
    mock_collection_instance.create_index.assert_called_once()
    mock_collection_instance.load.assert_called_once()
    assert collection == mock_collection_instance

@patch("database.milvus_database.utility.list_collections", return_value=["face_vectors"])
@patch("database.milvus_database.Collection")
def test_create_milvus_collection_existing_collection_without_index(mock_collection_cls, mock_list_collections):
    mock_collection_instance = MagicMock()
    mock_collection_instance.has_index.return_value = False
    mock_collection_cls.return_value = mock_collection_instance

    collection = create_milvus_collection()

    mock_list_collections.assert_called_once()
    mock_collection_cls.assert_called_once_with("face_vectors")
    mock_collection_instance.has_index.assert_called_once()
    mock_collection_instance.create_index.assert_called_once()
    mock_collection_instance.load.assert_called_once()
    assert collection == mock_collection_instance

@patch("database.milvus_database.utility.list_collections", return_value=["face_vectors"])
@patch("database.milvus_database.Collection")
def test_create_milvus_collection_existing_collection_with_index(mock_collection_cls, mock_list_collections):
    mock_collection_instance = MagicMock()
    mock_collection_instance.has_index.return_value = True
    mock_collection_cls.return_value = mock_collection_instance

    collection = create_milvus_collection()

    mock_list_collections.assert_called_once()
    mock_collection_cls.assert_called_once_with("face_vectors")
    mock_collection_instance.has_index.assert_called_once()
    mock_collection_instance.create_index.assert_not_called()
    mock_collection_instance.load.assert_called_once()
    assert collection == mock_collection_instance

@patch("database.milvus_database.milvus_collection")
def test_load_face_database_with_data(mock_milvus_collection):
    mock_milvus_collection.load = MagicMock()
    mock_milvus_collection.query = MagicMock(return_value=[
        {"employee_id": "E1", "name": "Alice", "embedding": [0.1]*512},
        {"employee_id": "E2", "name": "Bob", "embedding": [0.2]*512},
    ])

    embeddings, names, employee_ids = load_face_database()

    mock_milvus_collection.load.assert_called_once()
    mock_milvus_collection.query.assert_called_once()
    assert embeddings.shape == (2, 512)
    assert embeddings.device == DEVICE
    assert names == ["Alice", "Bob"]
    assert employee_ids == ["E1", "E2"]

@patch("database.milvus_database.milvus_collection")
def test_load_face_database_no_data(mock_milvus_collection):
    mock_milvus_collection.load = MagicMock()
    mock_milvus_collection.query = MagicMock(return_value=[])

    embeddings, names, employee_ids = load_face_database()

    assert embeddings.shape == (0, 512)
    assert names == []
    assert employee_ids == []

@patch("database.milvus_database.milvus_collection")
def test_load_face_database_exception(mock_milvus_collection):
    mock_milvus_collection.load = MagicMock()
    mock_milvus_collection.query = MagicMock(side_effect=Exception("some error"))

    embeddings, names, employee_ids = load_face_database()

    assert embeddings.shape == (0, 512)
    assert names == []
    assert employee_ids == []
