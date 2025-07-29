import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pymilvus import Collection, connections, CollectionSchema, FieldSchema, DataType, utility

from config import MILVUS_HOST, MILVUS_PORT

connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
COLLECTION_NAME = "face_vectors"

# เช็คว่ามี collection ชื่อ face_vectors หรือยัง
if "face_vectors" in utility.list_collections():
    collection = Collection("face_vectors")
    collection.drop()
    print("Dropped existing collection face_vectors")

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="employee_id", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
]

schema = CollectionSchema(fields, description="Face Embeddings Collection with employee_id")

collection = Collection(name="face_vectors", schema=schema)
print("Created new collection face_vectors with updated schema")

def create_milvus_collection():
    if COLLECTION_NAME in utility.list_collections():
        collection = Collection(COLLECTION_NAME)
        if not collection.has_index():
            collection.create_index(
                field_name="embedding",
                index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}},
                sync=True  # ถ้ามีพารามิเตอร์นี้ในไลบรารีของคุณ
            )
        collection.load()
        return collection

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="employee_id", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512)
    ]

    schema = CollectionSchema(fields, description="Face Embeddings Collection")
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    collection.create_index(
        field_name="embedding",
        index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}},
        sync=True
    )

    collection.load()
    return collection



milvus_collection = create_milvus_collection()
