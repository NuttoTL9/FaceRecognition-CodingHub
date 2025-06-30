from pymilvus import connections, Collection
import torch

# ⚙️ กำหนดค่าการเชื่อมต่อ
connections.connect("default", host="127.0.0.1", port="19530")
collection = Collection("face_vectors")

def list_all(limit=100):
    print("📦 ข้อมูลใน Milvus:")
    collection.load()
    results = collection.query(expr="", output_fields=["name"], limit=limit)
    for i, item in enumerate(results, start=1):
        print(f"{i}. {item['name']}")

def delete_by_name(name: str):
    expr = f'name == "{name}"'
    print(f"🗑️ ลบข้อมูลที่มีชื่อ: {name}")
    res = collection.delete(expr)
    print("✅ ลบแล้ว:", res)

def drop_all():
    confirm = input("⚠️ ต้องการลบทุกข้อมูลใน Milvus ใช่ไหม? พิมพ์ 'yes' เพื่อยืนยัน: ")
    if confirm.lower() == "yes":
        collection.drop()
        print("❌ ลบ collection แล้ว")
    else:
        print("❎ ยกเลิกการลบ")

def count_entities():
    print(f"📊 จำนวนข้อมูลใน Milvus: {collection.num_entities}")

def list_all(limit=100):
    print("📦 ข้อมูลใน Milvus:")
    collection.load()
    results = collection.query(expr="", output_fields=["name", "embedding"], limit=limit)
    for i, item in enumerate(results, start=1):
        name = item['name']
        emb_sample = item['embedding'][:5]  # แสดงแค่ 5 ตัวแรก
        print(f"{i}. Name: {name}")
        print(f"   Embedding: {emb_sample}...")


if __name__ == "__main__":
    print("🔧 Milvus Admin Tool")
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
        print("❌ คำสั่งไม่ถูกต้อง")
