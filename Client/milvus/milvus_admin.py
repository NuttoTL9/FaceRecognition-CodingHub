from pymilvus import connections, utility, CollectionSchema, FieldSchema, DataType, Collection
import torch

connections.connect("default", host="192.168.1.27", port="19530")
collection = Collection("face_vectors")

def list_all(limit=100):
    print("ข้อมูลใน Milvus:")
    collection.load()
    results = collection.query(expr="", output_fields=["name"], limit=limit)
    print(results)
    print
    for i, item in enumerate(results, start=1):
        print(f"{i}. {item['name']}")

def delete_by_name(name: str):
    expr = f'name == "{name}"'
    print(f"ลบข้อมูลที่มีชื่อ: {name}")
    res = collection.delete(expr)
    print("ลบแล้ว:", res)

def drop_all():
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

    print("Drop database Suscess ")

def count_entities():
    print(f"จำนวนข้อมูลใน Milvus: {collection.num_entities}")

def list_all(limit=100):
    print("ข้อมูลใน Milvus:")
    collection.load()
    results = collection.query(expr="", output_fields=["name", "embedding"], limit=limit)
    for i, item in enumerate(results, start=1):
        name = item['name']
        emb_sample = item['embedding'][:5]
        print(f"{i}. Name: {name}")
        print(f"   Embedding: {emb_sample}")


if __name__ == "__main__":
    print("Milvus Admin Tool")
    print("1. แสดงข้อมูลทั้งหมด")
    print("2. ลบข้อมูลตามชื่อ")
    print("3. ลบข้อมูลทั้งหมด")
    print("4. ตรวจสอบจำนวนข้อมูล")
    print("5. แสดงข้อมูลแบบ Embedding")

    choice = input("เลือกตัวเลขคำสั่ง: ").strip()

    if choice == "1":
        list_all()
    elif choice == "2":
        name = input("กรอกชื่อที่ต้องการลบ: ").strip()
        delete_by_name(name)
    elif choice == "3":
        drop_all()
    elif choice == "4":
        count_entities()
    elif choice == "5":
        list_all()
    else:
        print("คำสั่งไม่ถูกต้อง")
