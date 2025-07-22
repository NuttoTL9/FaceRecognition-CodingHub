import torch
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from config import DEVICE, COLLECTION_NAME, MILVUS_HOST, MILVUS_PORT

connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
milvus_collection = None

def create_milvus_collection():
    global milvus_collection
    if COLLECTION_NAME in utility.list_collections():
        milvus_collection = Collection(COLLECTION_NAME)
        if not milvus_collection.has_index():
            milvus_collection.create_index(
                field_name="embedding",
                index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}},
                sync=True
            )
        milvus_collection.load()
        return milvus_collection

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="employee_id", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512)
    ]
    schema = CollectionSchema(fields, description="Face Embeddings Collection")
    milvus_collection = Collection(name=COLLECTION_NAME, schema=schema)
    milvus_collection.create_index(
        field_name="embedding",
        index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}},
        sync=True
    )
    milvus_collection.load()
    return milvus_collection

def load_face_database():
    vectors, names, employee_ids = [], [], []
    try:
        milvus_collection.load()
        results = milvus_collection.query(expr="", output_fields=["employee_id", "name", "embedding"], limit=10000)
        for item in results:
            vectors.append(torch.tensor(item['embedding']).to(DEVICE))
            names.append(item['name'])
            employee_ids.append(item['employee_id'])
        if not vectors:
            return torch.empty(0, 512).to(DEVICE), names, employee_ids
        return torch.stack(vectors), names, employee_ids
    except Exception as e:
        print("Failed to load from Milvus:", e)
        return torch.empty(0, 512).to(DEVICE), [], []
