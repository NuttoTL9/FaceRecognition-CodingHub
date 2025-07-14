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
from grpc_client.milvus_grpc_utils import encode_vector_with_grpc
from videostreamthread import videostreamthread
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

RTSP_URLS = [
    0,
]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device, post_process=False)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

connections.connect("default", host="192.168.1.27", port=19530)


COLLECTION_NAME = "face_vectors"

camera_streams = {}
camera_frames = {}
embedding_lock = threading.Lock()

shared_embeddings = torch.empty(0, 512).to(device)
shared_names = []
shared_employee_ids = []

should_exit = False
MIN_LOG_INTERVAL = 60  # วินาที
last_log_times = {}
person_states = {}


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


def load_face_database_from_milvus():
    vectors, names, employee_ids = [], [], []
    try:
        milvus_collection.load()
        results = milvus_collection.query(expr="", output_fields=["employee_id", "name", "embedding"], limit=10000)
        for item in results:
            vectors.append(torch.tensor(item['embedding']).to(device))
            names.append(item['name'])
            employee_ids.append(item['employee_id'])
        if not vectors:
            return torch.empty(0, 512).to(device), names, employee_ids
        return torch.stack(vectors), names, employee_ids
    except Exception as e:
        print("Failed to load from Milvus:", e)
        return torch.empty(0, 512).to(device), [], []


def reload_face_database():
    global shared_embeddings, shared_names, shared_employee_ids
    embeddings, names, employee_ids = load_face_database_from_milvus()
    print("Loaded employee_ids and names from Milvus:", [f"{eid}:{name}" for eid, name in zip(employee_ids, names)])
    with embedding_lock:
        shared_employee_ids = employee_ids
        shared_names = names
        shared_embeddings = embeddings


def find_closest_match(face_embedding, db_embeddings, db_employee_ids, db_names, threshold=1.0):
    if db_embeddings.shape[0] == 0:
        return "Unknown", "Unknown", float('inf')
    distances = torch.norm(db_embeddings - face_embedding, dim=1)
    min_dist, idx = torch.min(distances, dim=0)
    if min_dist.item() <= threshold:
        return db_employee_ids[idx], db_names[idx], min_dist.item()
    else:
        return "Unknown", "Unknown", min_dist.item()


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


def process_camera(rtsp_url, window_name):
    stream = videostreamthread(rtsp_url)
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
                    local_employee_ids = shared_employee_ids.copy()

                for embedding, box in zip(embeddings, boxes):
                    employee_id, name, distance = find_closest_match(embedding.unsqueeze(0), local_embeddings, local_employee_ids, local_names)
                    x1, y1, x2, y2 = map(int, box)

                    label = f"{name} ({distance:.2f})"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    if employee_id != "Unknown" and distance < 0.7:
                        now = time.time()
                        last_time = last_log_times.get(employee_id, 0)

                        if now - last_time >= MIN_LOG_INTERVAL:
                            last_state = person_states.get(employee_id, "out")
                            new_event = "in" if last_state == "out" else "out"

                            try:
                                payload = {"name": employee_id, "event": new_event}
                                res = requests.post("http://192.168.1.27:8989/log_event/", json=payload, timeout=3)

                                if res.ok:
                                    print(f"Logged {name} [{new_event}] at {datetime.datetime.now().isoformat()}")
                                    person_states[employee_id] = new_event
                                    last_log_times[employee_id] = now
                                else:
                                    print("Log failed:", res.status_code, res.text)

                            except Exception as e:
                                print("Logging error:", e)

        camera_frames[window_name] = frame.copy()


def main():
    print("Loading database from Milvus...")
    reload_face_database()

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

                    input_employee_id = input("Employee ID: ").strip()
                    input_name = input("ชื่อของบุคคล: ").strip()
                    if not input_employee_id or not input_name:
                        print("Employee ID หรือ ชื่อไม่ถูกต้อง")
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
                        grpc_response = encode_vector_with_grpc(
                            embedding_np.tolist(),
                            employee_id=input_employee_id,
                            name=input_name
                        )
                        reload_face_database()
                        print("บันทึกข้อมูลเรียบร้อย")
                        return grpc_response

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
