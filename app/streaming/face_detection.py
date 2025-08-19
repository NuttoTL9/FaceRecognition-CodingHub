import threading
import time
import datetime
import cv2
import requests
import torch
import queue

from config import DEVICE, MIN_LOG_INTERVAL, MIN_FACE_AREA, LOG_EVENT_URL
from recognition.face_models import mtcnn, resnet
from recognition.face_utils import preprocess_face, find_closest_match
from videostreamthread import videostreamthread
from database.milvus_database import load_face_database
from notify.notify import send_discord_alert
from collections import deque

camera_streams = {}
camera_frames = {}
embedding_lock = threading.Lock()

person_states = {}
last_log_times = {}

_last_dets = {}
DETS_TTL = 5

_trackers = {}
TRACKER_TTL = 90
MIN_BOX_SIZE = 8

shared_embeddings = torch.empty(0, 512).to(DEVICE)
shared_names = []
shared_employee_ids = [] 

should_exit = [False]

last_unknown_alert_time = 0
UNKNOWN_ALERT_INTERVAL = 60 
pending_unknown_alert = {"time": None, "frame": None}

_log_queue = queue.Queue(maxsize=256)
_ui_event_queue = deque(maxlen=200)

try:
    mtcnn.eval()
except Exception:
    pass
try:
    resnet.eval()
except Exception:
    pass

torch.set_grad_enabled(False)

USE_FP16 = torch.cuda.is_available()

if USE_FP16:
    try:
        resnet.half()
    except Exception:
        USE_FP16 = False

def _xyxy_to_xywh(x1, y1, x2, y2):
    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

def _xywh_to_xyxy(x, y, w, h):
    return (int(x), int(y), int(x + w), int(y + h))

def _iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    a = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    b = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / float(a + b - inter + 1e-6)

def _normalize_embeddings(embs: torch.Tensor):
    if embs.numel() == 0:
        return embs
    return torch.nn.functional.normalize(embs, p=2, dim=1)

def _maybe_half(t: torch.Tensor):
    return t.half() if (USE_FP16 and t.is_floating_point()) else t

def get_ui_events(max_items=10):
    """ดึง event สำหรับ GUI ทีละชุด (ไม่บล็อกลูป)"""
    out = []
    for _ in range(min(max_items, len(_ui_event_queue))):
        out.append(_ui_event_queue.popleft())
    return out

def _create_tracker():
    tracker = None
    if hasattr(cv2, "legacy"):
        if hasattr(cv2.legacy, "TrackerMOSSE_create"):
            tracker = cv2.legacy.TrackerMOSSE_create()
        elif hasattr(cv2.legacy, "TrackerKCF_create"):
            tracker = cv2.legacy.TrackerKCF_create()
    else:
        if hasattr(cv2, "TrackerMOSSE_create"):
            tracker = cv2.TrackerMOSSE_create()
        elif hasattr(cv2, "TrackerKCF_create"):
            tracker = cv2.TrackerKCF_create()

    if tracker is None:
        raise RuntimeError("No MOSSE/KCF tracker available in this OpenCV build")
    return tracker

def reload_face_database():
    global shared_embeddings, shared_names, shared_employee_ids
    embeddings, names, employee_ids = load_face_database()
    
    # แสดงเฉพาะพนักงานที่ไม่ซ้ำพร้อมจำนวน embedding
    from collections import Counter
    employee_counts = Counter(employee_ids)
    unique_employees = {}
    for i, (eid, name) in enumerate(zip(employee_ids, names)):
        if eid not in unique_employees:
            unique_employees[eid] = name
    
    print("Loaded unique employees from Milvus:")
    for eid, name in unique_employees.items():
        count = employee_counts[eid]
        print(f"   - {eid}: {name} ({count} embeddings)")
    
    with embedding_lock:
        shared_employee_ids = employee_ids
        shared_names = names
        if isinstance(embeddings, torch.Tensor):
            emb = embeddings
        else:
            emb = torch.as_tensor(embeddings, dtype=torch.float32)
        emb = emb.to(DEVICE, non_blocking=True)
        emb = _normalize_embeddings(emb)
        emb = _maybe_half(emb)
        shared_embeddings = emb


def process_camera(rtsp_url, window_name):
    stream = videostreamthread(rtsp_url)
    camera_streams[window_name] = stream
    print(f"Started: {window_name}")

    FRAME_SKIP = 2
    DOWNSCALE  = 0.5
    frame_i = 0

    while not should_exit[0]:
        ret, frame = stream.read()
        if not ret:
            continue

        frame_i += 1
        do_detect = (frame_i % FRAME_SKIP == 0)

        if do_detect:
            small = cv2.resize(frame, (0, 0), fx=DOWNSCALE, fy=DOWNSCALE, interpolation=cv2.INTER_LINEAR)
            small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            boxes = None
            try:
                boxes, _probs, _lands = mtcnn.detect(small_rgb, landmarks=True)
            except Exception:
                boxes = None

            if boxes is not None and len(boxes):
                boxes = boxes * (1.0 / DOWNSCALE)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                face_tensors, valid_boxes = extract_valid_faces(frame_rgb, boxes)
                if face_tensors:
                    faces_batch = torch.cat(face_tensors, dim=0).to(DEVICE, non_blocking=True)
                    faces_batch = _maybe_half(faces_batch)
                    with torch.no_grad():
                        embeddings = resnet(faces_batch)
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                    items = identify_and_log_faces(frame, embeddings, valid_boxes)

                    if items:
                        _init_trackers_from_items(window_name, frame, items)
            else:
                _update_trackers(window_name, frame)
            camera_frames[window_name] = frame
            

        else:
            cache = _last_dets.get(window_name)
            if cache and cache.get("ttl", 0) > 0:
                for (x1, y1, x2, y2, label, color) in cache["items"]:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                cache["ttl"] -= 1

    
def _init_trackers_from_items(window_name, frame, drawn_items):

    h, w = frame.shape[:2]
    cur_list = _trackers.get(window_name, [])
    used_old = [False] * len(cur_list)
    new_list = []

    for (x1, y1, x2, y2, label, color) in drawn_items:
        if x2 - x1 < MIN_BOX_SIZE or y2 - y1 < MIN_BOX_SIZE:
            continue
        best_iou, best_j = 0.0, -1
        for j, old in enumerate(cur_list):
            ox1, oy1, ox2, oy2 = old.get("last_xyxy", (0, 0, 0, 0))
            iou = _iou((x1, y1, x2, y2), (ox1, oy1, ox2, oy2))
            if iou > best_iou:
                best_iou, best_j = iou, j

        if best_iou > 0.2 and best_j >= 0:
            try:
                t = _create_tracker()
                t.init(frame, _xyxy_to_xywh(x1, y1, x2, y2))
                cur_list[best_j]["tracker"] = t
                cur_list[best_j]["label"] = label
                cur_list[best_j]["color"] = color
                cur_list[best_j]["ttl"] = TRACKER_TTL
                cur_list[best_j]["last_xyxy"] = (x1, y1, x2, y2)
                new_list.append(cur_list[best_j])
                used_old[best_j] = True
            except Exception:
                pass
        else:
            try:
                t = _create_tracker()
                t.init(frame, _xyxy_to_xywh(x1, y1, x2, y2))
                new_list.append({
                    "tracker": t,
                    "label": label,
                    "color": color,
                    "ttl": TRACKER_TTL,
                    "last_xyxy": (x1, y1, x2, y2),
                })
            except Exception:
                pass

    for j, old in enumerate(cur_list):
        if used_old[j]:
            continue
        old["ttl"] -= 1
        if old["ttl"] > 0:
            new_list.append(old)

    _trackers[window_name] = new_list

def _update_trackers(window_name, frame):
    lst = _trackers.get(window_name, [])
    new_lst = []
    for obj in lst:
        t = obj["tracker"]
        ok, bbox = t.update(frame)
        if not ok:
            obj["ttl"] -= 3
            continue
        x, y, w, h = bbox
        if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
            obj["ttl"] -= 3
            continue

        x1, y1, x2, y2 = _xywh_to_xyxy(x, y, w, h)
        h_img, w_img = frame.shape[:2]
        if x2 <= 0 or y2 <= 0 or x1 >= w_img or y1 >= h_img:
            obj["ttl"] -= 3
            continue

        obj["last_xyxy"] = (x1, y1, x2, y2)
        obj["ttl"] -= 1

        color = obj["color"]; label = obj["label"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        if obj["ttl"] > 0:
            new_lst.append(obj)

    _trackers[window_name] = new_lst


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
    logging_now_set = set()

    THRESH = 0.73
    drawn_items = []

    for emb, box in zip(embeddings, boxes):
        employee_id, name, distance = find_closest_match(
            emb.unsqueeze(0),
            shared_embeddings,
            shared_employee_ids,
            shared_names
        )
        x1, y1, x2, y2 = map(int, box)
        is_known = (employee_id != "Unknown" and distance < THRESH)
        color = (0, 255, 0) if is_known else (0, 0, 255)
        label = f"{name if is_known else 'Unknown'} {distance:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        drawn_items.append((x1, y1, x2, y2, label, color))

        if is_known:
            found_known = True
            if employee_id not in logging_now_set:
                logging_now_set.add(employee_id)
                log_recognition_event(employee_id, name, frame, (x1, y1, x2, y2))
        else:
            found_unknown = True

    now = time.time()
    if found_known:
        pending_unknown_alert["time"] = None
        pending_unknown_alert["frame"] = None
    elif found_unknown:
        if pending_unknown_alert["time"] is None:
            pending_unknown_alert["time"] = now
            pending_unknown_alert["frame"] = frame
        elif now - pending_unknown_alert["time"] > 2:
            if now - last_unknown_alert_time > UNKNOWN_ALERT_INTERVAL:
                now_dt = datetime.datetime.now()
                date_str = now_dt.strftime("%d/%m/%Y")
                time_str = now_dt.strftime("%H.%M") + " น."
                message = f"พบใบหน้าที่ไม่รู้จัก! วันที่ {date_str} เวลา {time_str}"
                try:
                    _, img_encoded = cv2.imencode('.jpg', pending_unknown_alert["frame"])
                    send_discord_alert(message, img_encoded.tobytes())
                    last_unknown_alert_time = now
                except Exception:
                    pass
            pending_unknown_alert["time"] = None
            pending_unknown_alert["frame"] = None

    return drawn_items

def _log_worker():
    while True:
        try:
            item = _log_queue.get()
            if item is None:
                break
            employee_id, name, event, frame = item
            try:
                send_log_with_image(
                    employee_id, name, event, frame,
                    LOG_EVENT_URL.replace('/log_event/', '/log_event_with_snap/')
                )
            except Exception as e:
                print("Logging error:", e)
        except Exception as e:
            print("Log worker error:", e)
        finally:
            _log_queue.task_done()
_log_thread = threading.Thread(target=_log_worker, daemon=True)
_log_thread.start()

def log_recognition_event(employee_id, name, frame, box):
    now = time.time()
    last_time = last_log_times.get(employee_id, 0)

    now_dt = datetime.datetime.now()
    hour = now_dt.hour
    if hour < 9:
        new_event = "in"
    elif hour >= 12:
        new_event = "out"
    else:
        return

    last_state = person_states.get(employee_id, None)
    if (now - last_time) < MIN_LOG_INTERVAL and last_state == new_event:
        return

    person_states[employee_id] = new_event
    last_log_times[employee_id] = now

    try:
        x1, y1, x2, y2 = box
        x1 = max(0, x1); y1 = max(0, y1)
        face_roi = frame[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else frame
        thumb = cv2.resize(face_roi, (100, 100), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode('.jpg', thumb)
        if ok:
            _ui_event_queue.append({
                "employee_id": employee_id,
                "name": name,
                "event": new_event,
                "time": now_dt.strftime("%d-%m-%Y %H:%M:%S"),
                "image_jpg": jpg.tobytes()
            })
    except Exception:
        pass

    try:
        if not _log_queue.full():
            _log_queue.put_nowait((employee_id, name, new_event, frame))
    except Exception:
        pass


def send_log_with_image(employee_id, name, event, frame, server_url):
    _, img_encoded = cv2.imencode('.jpg', frame)

    files = {
        'snap_file': ('snap.jpg', img_encoded.tobytes(), 'image/jpeg')
    }
    data = {
        'employee_id': employee_id,
        'name': name,
        'event': event
    }
    print("Sending data:", data)

    try:
        response = requests.post(server_url, data=data, files=files, timeout=5)
        print('Status code:', response.status_code)
        response_json = response.json()
    except requests.exceptions.RequestException as e:
        print('API log image error:', e)
    except ValueError:
        print('Response is not valid JSON:', response.text)