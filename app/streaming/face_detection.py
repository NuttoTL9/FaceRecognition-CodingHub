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
from notify.notify import send_discord_alert
# from streaming.blink_detector import detect_blink  # ตัดออก

camera_streams = {}
camera_frames = {}
embedding_lock = threading.Lock()

person_states = {}
last_log_times = {}

shared_embeddings = torch.empty(0, 512).to(DEVICE)
shared_names = []
shared_employee_ids = [] 

should_exit = [False]

last_unknown_alert_time = 0
UNKNOWN_ALERT_INTERVAL = 60 
pending_unknown_alert = {"time": None, "frame": None}

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
        boxes, _probs, _landmarks = mtcnn.detect(frame_rgb, landmarks=True)

        if boxes is not None:
            print(f"DEBUG: Detected {len(boxes)} faces")
            face_tensors, valid_boxes = extract_valid_faces(frame_rgb, boxes)
            if face_tensors:
                print(f"DEBUG: {len(face_tensors)} valid faces after filtering")
                embeddings = get_embeddings(face_tensors)
                identify_and_log_faces(frame, embeddings, valid_boxes)
            else:
                print("DEBUG: No valid faces after filtering")
        else:
            print("DEBUG: No faces detected")

        camera_frames[window_name] = frame.copy()


def extract_valid_faces(frame_rgb, boxes):
    face_tensors = []
    valid_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        box_area = (x2 - x1) * (y2 - y1)
        if box_area < MIN_FACE_AREA:
            continue
        face_tensor = preprocess_face(frame_rgb, box)
        if face_tensor is not None:
            face_tensors.append(face_tensor)
            valid_boxes.append(box)
    return face_tensors, valid_boxes


def get_embeddings(face_tensors):
    faces_batch = torch.cat(face_tensors)
    with torch.no_grad():
        embeddings = resnet(faces_batch)
    return embeddings


def identify_and_log_faces(frame, embeddings, boxes):
    global last_unknown_alert_time, pending_unknown_alert
    found_known = False
    found_unknown = False

    for embedding, box in zip(embeddings, boxes):
        employee_id, name, distance = find_closest_match(
            embedding.unsqueeze(0),
            shared_embeddings,
            shared_employee_ids,
            shared_names
        )
        x1, y1, x2, y2 = map(int, box)
        label = f"{name} ({distance:.2f})"
        color = (0, 255, 0) if employee_id != "Unknown" and distance < 0.7 else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ตัด blink detection ออก
        if employee_id != "Unknown" and distance < 0.7:
            print(f"DEBUG: Recognized known person - {name} (ID: {employee_id}) with distance: {distance}")
            found_known = True
            log_recognition_event(employee_id, name, frame)
        else:
            print(f"DEBUG: Unknown person or high distance - {name} (ID: {employee_id}) with distance: {distance}")

        elif employee_id == "Unknown" or distance >= 0.7:
            found_unknown = True

    now = time.time()
    if found_known:
        pending_unknown_alert["time"] = None
        pending_unknown_alert["frame"] = None
    elif found_unknown:
        now = time.time()
        if pending_unknown_alert["time"] is None:
            pending_unknown_alert["time"] = now
            pending_unknown_alert["frame"] = frame.copy()
        elif now - pending_unknown_alert["time"] > 2:
            if now - last_unknown_alert_time > UNKNOWN_ALERT_INTERVAL:
                now_dt = datetime.datetime.now()
                date_str = now_dt.strftime("%d/%m/%Y")
                time_str = now_dt.strftime("%H.%M") + " น."
                message = f"พบใบหน้าที่ไม่รู้จัก! วันที่ {date_str} เวลา {time_str}"
                _, img_encoded = cv2.imencode('.jpg', pending_unknown_alert["frame"])
                send_discord_alert(message, img_encoded.tobytes())
                last_unknown_alert_time = now
            pending_unknown_alert["time"] = None
            pending_unknown_alert["frame"] = None


def log_recognition_event(employee_id, name, frame):
    now = time.time()
    last_time = last_log_times.get(employee_id, 0)

    now_dt = datetime.datetime.now()
    hour = now_dt.hour
    
    # Debug: Print current time and conditions
    print(f"DEBUG: Current hour: {hour}, employee_id: {employee_id}, name: {name}")
    
    if hour < 9:
        new_event = "in"
    elif hour >= 12:
        new_event = "out"
    else:
        print(f"DEBUG: Hour {hour} is between 9-12, skipping log")
        return

    last_state = person_states.get(employee_id, None)
    time_since_last = now - last_time
    
    print(f"DEBUG: new_event: {new_event}, last_state: {last_state}, time_since_last: {time_since_last}, MIN_LOG_INTERVAL: {MIN_LOG_INTERVAL}")
    
    if last_state == new_event and time_since_last < MIN_LOG_INTERVAL:
        print(f"DEBUG: Skipping log - same event {new_event} within interval")
        return

    # Debug: Print LOG_EVENT_URL to check if it's configured
    print(f"LOG_EVENT_URL: {LOG_EVENT_URL}")
    
    if not LOG_EVENT_URL:
        print("ERROR: LOG_EVENT_URL is not configured!")
        return

    try:
        payload = {"name": employee_id, "event": new_event}
        print(f"Sending payload: {payload} to {LOG_EVENT_URL}")
        res = requests.post(LOG_EVENT_URL, json=payload, timeout=3)
        if res.ok:
            print(f"Logged {name} [{new_event}] at {now_dt.isoformat()}")
            person_states[employee_id] = new_event
            last_log_times[employee_id] = now
            send_log_with_image(name, employee_id, new_event, frame, LOG_EVENT_URL.replace('/log_event/', '/log_event_with_snap/'))
        else:
            print("Log failed:", res.status_code, res.text)
    except Exception as e:
        print("Logging error:", e)


def send_log_with_image(employee_id, name, event, frame, server_url):
    _, img_encoded = cv2.imencode('.jpg', frame)

    files = {
        'snap_file': ('snap.jpg', img_encoded.tobytes(), 'image/jpeg')  # ต้องใช้ชื่อฟิลด์ snap_file
    }
    data = {
        'name': name,  # ตรงกับ Form(name)
        'event': event,
        'employee_id': employee_id  # ส่ง employee_id ด้วย
    }
    print("Sending data:", data)
    print(f"Sending to server_url: {server_url}")

    try:
        response = requests.post(server_url, data=data, files=files, timeout=5)
        print('Status code:', response.status_code)
        if response.headers.get('content-type', '').startswith('application/json'):
            response_json = response.json()
            print('Response JSON:', response_json)
        else:
            print('Response text:', response.text)
    except requests.exceptions.RequestException as e:
        print('API log image error:', e)
    except ValueError:
        print('Response is not valid JSON:', response.text)