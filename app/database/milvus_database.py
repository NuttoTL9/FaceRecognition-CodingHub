# milvus_database.py
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
    global milvus_collection
    if milvus_collection is None:
        create_milvus_collection()

    vectors, names, employee_ids = [], [], []
    try:
        milvus_collection.load()
        results = milvus_collection.query(
            expr="id >= 0",
            output_fields=["employee_id", "name", "embedding"],
            limit=10000
        )
        if not results:
            return torch.empty(0, 512).to(DEVICE), [], []

        # นับจำนวน embedding ต่อ employee_id
        employee_counts = {}
        
        for item in results:
            emb = item.get('embedding')
            if emb is None:
                continue
            vectors.append(torch.tensor(emb, dtype=torch.float32).to(DEVICE))
            names.append(item.get('name', 'Unknown'))
            employee_ids.append(item.get('employee_id', 'Unknown'))
            
            emp_id = item.get('employee_id', 'Unknown')
            employee_counts[emp_id] = employee_counts.get(emp_id, 0) + 1

        return torch.stack(vectors), names, employee_ids
    except Exception as e:
        print("Failed to load from Milvus:", e)
        return torch.empty(0, 512).to(DEVICE), [], []

def add_embedding_to_milvus(employee_id: str, name: str, embedding):
    """
    เพิ่ม embedding ใหม่สำหรับ employee_id โดยไม่ลบของเก่า
    รองรับทั้ง torch.Tensor และ list
    """
    global milvus_collection
    if milvus_collection is None:
        create_milvus_collection()
    
    # แปลง embedding เป็น list ของ float
    if isinstance(embedding, torch.Tensor):
        if embedding.ndim == 2 and embedding.shape[0] == 1:
            embedding = embedding.squeeze(0)  # shape (512,)
        if embedding.shape[0] != 512:
            raise ValueError(f"Embedding dimension mismatch: {embedding.shape[0]} != 512")
        vector = embedding.cpu().detach().numpy().astype(float).tolist()
    elif isinstance(embedding, list):
        if len(embedding) != 512:
            raise ValueError(f"Embedding dimension mismatch: {len(embedding)} != 512")
        vector = [float(x) for x in embedding]
    else:
        raise ValueError(f"Unsupported embedding type: {type(embedding)}")

    vectors = [vector]  # Milvus ต้องการ list of vectors
    employee_ids = [employee_id]
    names = [name]

    milvus_collection.insert([employee_ids, names, vectors])
    milvus_collection.flush()
    return f"Added new embedding for {employee_id}"

def search_face(embedding: torch.Tensor, top_k: int = 3):
    global milvus_collection
    if milvus_collection is None:
        create_milvus_collection()

    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = milvus_collection.search(
        data=[embedding.cpu().numpy().tolist()],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        expr=None
    )

    if results and len(results[0]) > 0:
        return [(hit.entity.get("employee_id"), hit.distance) for hit in results[0]]
    return []
