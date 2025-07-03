from pymilvus import connections, utility, CollectionSchema, FieldSchema, DataType, Collection

connections.connect("default", host="192.168.1.27", port="19530")

if "face_vectors" in utility.list_collections():
    utility.drop_collection("face_vectors")

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512)
]

schema = CollectionSchema(fields, description="Face Embeddings Collection")
collection = Collection(name="face_vectors", schema=schema)
collection.create_index(
    field_name="embedding",
    index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
)
collection.load()

print("Milvus collection 'face_vectors' created with dim=2048")
