import os
import time
import datetime
import requests
import threading
import cv2
import torch
import numpy as np

from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

from VideoStreamThread import VideoStreamThread
from glasses_overlay import overlay_glasses_with_eyes
from mask_overlay import overlay_mask_with_eyes
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

# ---------------- Configurations ---------------- #
RTSP_URLS = [
    "rtsp://admin:Codinghub22@192.168.1.101:554/Streaming/Channels/102",
    "rtsp://admin:Codinghub22@192.168.1.102:554/Streaming/Channels/102",
    "rtsp://admin:johny2121@192.168.1.30:554/Streaming/Channels/201/"
]

SHEETDB_API_URL = "https://sheetdb.io/api/v1/vq3gqcx2oz3kt"
BASE_DIR = os.path.dirname(__file__)
MIN_LOG_INTERVAL = 60  # seconds

# ---------------- Initialize Models ---------------- #
device = torch.device("cuda:0")
mtcnn = MTCNN(keep_all=True, device=device, post_process=False)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
connections.connect("default", host="192.168.1.27", port="19530")

# ---------------- Global State ---------------- #
camera_streams = {}
camera_frames = {}
embedding_lock = threading.Lock()
shared_embeddings = torch.empty(0, 512).to(device)
shared_names = []
should_exit = False

# ---------------- Milvus ---------------- #
def create_milvus_collection():
    if "face_vectors" in utility.list_collections():
        return Collection("face_vectors")

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
    return collection

milvus_collection = create_milvus_collection()

def reload_face_database():
    global shared_embeddings, shared_names
    embeddings, names = load_face_database_from_milvus()
    print("Loaded names from Milvus:", names)  # Debug เพิ่มเติม
    with embedding_lock:
        shared_embeddings = embeddings
        shared_names = names

def load_face_database_from_milvus():
    vectors, names = [], []
    try:
        milvus_collection.load()
        results = milvus_collection.query(expr="", output_fields=["name", "embedding"], limit=10000)
        for item in results:
            vectors.append(torch.tensor(item['embedding']).to(device))
            names.append(item['name'])
        if not vectors:
            return torch.empty(0, 512).to(device), []
        return torch.stack(vectors), names
    except Exception as e:
        print("Failed to load from Milvus:", e)
        return torch.empty(0, 512).to(device), []


def find_closest_match(face_embedding, db_embeddings, db_names, threshold=1.0):
    if db_embeddings.shape[0] == 0:
        return "Unknown", float('inf')
    distances = torch.norm(db_embeddings - face_embedding, dim=1)
    min_dist, idx = torch.min(distances, dim=0)
    return (db_names[idx], min_dist.item()) if min_dist.item() <= threshold else ("Unknown", min_dist.item())


def preprocess_face(img_np, box):
    x1, y1, x2, y2 = [int(coord) for coord in box]
    h, w, _ = img_np.shape

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    face_img = img_np[y1:y2, x1:x2]

    if face_img.size == 0:
        return None

    face_img = cv2.resize(face_img, (160, 160))
    face_tensor = torch.from_numpy(face_img).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    return face_tensor


# ---------------- Camera Processor ---------------- #
def process_camera(rtsp_url, window_name):
    stream = VideoStreamThread(rtsp_url)
    camera_streams[window_name] = stream
    print(f"Started: {window_name}")

    while not should_exit:
        ret, frame = stream.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs, landmarks = mtcnn.detect(frame_rgb, landmarks=True)

        if boxes is not None:
            face_tensors = []
            for box in boxes:
                face_tensor = preprocess_face(frame_rgb, box)
                if face_tensor is None:
                    continue
                face_tensors.append(face_tensor)

            if face_tensors:
                faces_batch = torch.cat(face_tensors)
                with torch.no_grad():
                    embeddings = resnet(faces_batch)

                with embedding_lock:
                    local_embeddings = shared_embeddings.clone()
                    local_names = shared_names.copy()

                for embedding, box in zip(embeddings, boxes):
                    name, distance = find_closest_match(embedding.unsqueeze(0), local_embeddings, local_names)
                    x1, y1, x2, y2 = map(int, box)

                    label = f"{name} ({distance:.2f})"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


        camera_frames[window_name] = frame.copy()



# ---------------- Main ---------------- #
def main():
    print("Loading database from Milvus...")
    reload_face_database()
    db_embeddings, db_names = load_face_database_from_milvus()
    print(device)
    threads = []
    for idx, rtsp in enumerate(RTSP_URLS):
        cam_name = f"Camera-{idx+1}"
        t = threading.Thread(target=process_camera, args=(rtsp, cam_name))
        t.start()
        threads.append(t)

    while True:
        for name, frame in camera_frames.copy().items():
            cv2.imshow(name, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            def capture_and_save():
                try:
                    cam_choice = input(f"เลือกกล้องที่จะใช้ (1 - {len(RTSP_URLS)}): ").strip()
                    if not cam_choice.isdigit() or int(cam_choice) < 1 or int(cam_choice) > len(RTSP_URLS):
                        print("เลือกกล้องไม่ถูกต้อง")
                        return

                    cam_name = f"Camera-{cam_choice}"
                    stream = camera_streams.get(cam_name)
                    frame = camera_frames.get(cam_name)
                    if not stream or frame is None:
                        print("ไม่พบกล้องหรือภาพ")
                        return

                    input_name = input("ชื่อของบุคคล: ").strip()
                    if not input_name:
                        print("ชื่อไม่ถูกต้อง")
                        return

                    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    boxes, _ = mtcnn.detect(img_pil)
                    if boxes is None:
                        print("ไม่พบใบหน้าในภาพ")
                        return

                    for i, box in enumerate(boxes):
                        face_tensor = preprocess_face(frame, box)
                        if face_tensor is None:
                            print("ไม่สามารถสร้าง face tensor ได้")
                            continue

                        with torch.no_grad():
                            embedding = resnet(face_tensor)

                        embedding_np = embedding.squeeze(0).cpu().numpy().astype('float32')
                        print(f"Embedding shape: {embedding.shape}, numpy length: {len(embedding_np)}")
                        milvus_collection.insert([[input_name], [embedding_np.tolist()]])
                        milvus_collection.flush()
                        reload_face_database()
                        print("โหลดฐานข้อมูลใหม่เรียบร้อย")
                except Exception as e:
                    print(f"เกิดข้อผิดพลาดขณะบันทึก: {e}")

            threading.Thread(target=capture_and_save).start()

        elif key == ord('q'):
            global should_exit
            should_exit = True
            break

    for t in threads:
        t.join()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
