import threading
import time
import datetime
import cv2
import requests
import torch

from config import DEVICE, MIN_LOG_INTERVAL, MIN_FACE_AREA, LOG_EVENT_URL
from recognition.face_models import mtcnn, resnet
from recognition.face_utils import preprocess_face, find_closest_match
from videostreamthread import videostreamthread
from database.milvus_database import load_face_database

camera_streams = {}
camera_frames = {}
embedding_lock = threading.Lock()

person_states = {}
last_log_times = {}

shared_embeddings = torch.empty(0, 512).to(DEVICE)
shared_names = []
shared_employee_ids = []

should_exit = [False]  # ใช้ list เพื่อแชร์ mutable flag ข้ามโมดูล

def reload_face_database():
    global shared_embeddings, shared_names, shared_employee_ids
    embeddings, names, employee_ids = load_face_database()
    print("Loaded employee_ids and names from Milvus:", [f"{eid}:{name}" for eid, name in zip(employee_ids, names)])
    with embedding_lock:
        shared_employee_ids = employee_ids
        shared_names = names
        shared_embeddings = embeddings

def process_camera(rtsp_url, window_name):
    stream = videostreamthread(rtsp_url)
    camera_streams[window_name] = stream
    print(f"Started: {window_name}")

    while not should_exit[0]:
        ret, frame = stream.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs, landmarks = mtcnn.detect(frame_rgb, landmarks=True)

        if boxes is not None:
            face_tensors = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                box_area = (x2 - x1) * (y2 - y1)
                if box_area < MIN_FACE_AREA:
                    continue
                face_tensor = preprocess_face(frame_rgb, box)
                if face_tensor is not None:
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
                    employee_id, name, distance = find_closest_match(
                        embedding.unsqueeze(0),
                        local_embeddings,
                        local_employee_ids,
                        local_names
                    )
                    x1, y1, x2, y2 = map(int, box)

                    label = f"{name} ({distance:.2f})"
                    color = (0, 255, 0) if distance < 0.7 else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    if employee_id != "Unknown" and distance < 0.7:
                        now = time.time()
                        last_time = last_log_times.get(employee_id, 0)

                        if now - last_time >= MIN_LOG_INTERVAL:
                            last_state = person_states.get(employee_id, "out")
                            new_event = "in" if last_state == "out" else "out"

                            try:
                                payload = {"name": employee_id, "event": new_event}
                                res = requests.post(LOG_EVENT_URL, json=payload, timeout=3)
                                if res.ok:
                                    print(f"Logged {name} [{new_event}] at {datetime.datetime.now().isoformat()}")
                                    person_states[employee_id] = new_event
                                    last_log_times[employee_id] = now
                                else:
                                    print("Log failed:", res.status_code, res.text)
                            except Exception as e:
                                print("Logging error:", e)

        camera_frames[window_name] = frame.copy()
