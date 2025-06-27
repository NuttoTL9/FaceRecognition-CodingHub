import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import datetime
import time
import requests
from VideoStreamThread import VideoStreamThread
from glasses_overlay import overlay_glasses_with_eyes
from mask_overlay import overlay_mask_with_eyes

RTSP_URL = "rtsp://admin:Codinghub12@192.168.1.64:554/Streaming/Channels/102"
CAMERA_INDEX = 0
SHEETDB_API_URL = "https://sheetdb.io/api/v1/vq3gqcx2oz3kt"
FACE_DB_FOLDER = os.path.join(os.path.dirname(__file__), 'data')
FACE_DB_CROP = os.path.join(os.path.dirname(__file__), 'cropdata')
MIN_LOG_INTERVAL = 60
original_frame = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

attendance_status = {}
last_logged_time = {}

def load_face_database(*folder_paths):
    embeddings = []
    names = []

    for folder_path in folder_paths:
        for root, _, files in os.walk(folder_path):  # ✅ เข้า subfolder ได้ทุกระดับ
            for filename in files:
                if filename.lower().endswith(('.jpg', '.png')):
                    name = os.path.splitext(filename)[0].split('_')[0]
                    img_path = os.path.join(root, filename)
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((160, 160))
                    face_tensor = mtcnn(img)
                    if face_tensor is not None:
                        if face_tensor.dim() == 3:
                            face_tensor = face_tensor.unsqueeze(0)
                        face_embedding = resnet(face_tensor.to(device))
                        embeddings.append(face_embedding)
                        names.append(name)
                        print(f"Loaded face: {name} from {img_path}")
                    else:
                        print(f"No face detected in: {img_path}")

    if not embeddings:
        print("❌ No faces found in the database!")
    return embeddings, names

print("Loading face database...")
database_embeddings, database_names = load_face_database(FACE_DB_FOLDER, FACE_DB_CROP)

def log_attendance(name):
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

    last_time = last_logged_time.get(name)
    if last_time and (now - last_time).total_seconds() < MIN_LOG_INTERVAL:
        return

    if name not in attendance_status:
        status = "Check-in"
        attendance_status[name] = "in"
    else:
        status = "Check-out" if attendance_status[name] == "in" else "Check-in"
        attendance_status[name] = "out" if attendance_status[name] == "in" else "in"

    data = {
        "data": [
            {
                "Name": name,
                "Timestamp": now_str,
                "Status": status
            }
        ]
    }

    try:
        response = requests.post(SHEETDB_API_URL, json=data)
        if response.status_code == 201:
            print(f"Logged {name} - {status} at {now_str}")
            last_logged_time[name] = now
        else:
            print("Failed to log attendance:", response.text)
    except Exception as e:
        print("Error sending data to SheetDB:", e)

def find_closest_match(face_embedding, database_embeddings, database_names):
    min_distance = float('inf')
    closest_name = "Unknown"
    for db_embedding, name in zip(database_embeddings, database_names):
        dist = (face_embedding - db_embedding).norm().item()
        if dist < min_distance:
            min_distance = dist
            closest_name = name
    if min_distance > 1.0:
        closest_name = "Unknown"
    return closest_name, min_distance

print("Connecting to IP Camera (RTSP)...")
#video_stream = VideoStreamThread(RTSP_URL)
video_stream = VideoStreamThread(CAMERA_INDEX)

while True:
    ret, frame = video_stream.read()
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
    saved_landmarks = landmarks if landmarks is not None else []
    original_frame = frame.copy()
    if not ret:
        print("No frame received from the camera")
        continue

    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            left_eye = landmarks[i][0]
            right_eye = landmarks[i][1]

            face_img = img.crop((x1, y1, x2, y2)).resize((160, 160))
            face_tensor = torch.tensor(np.array(face_img)).permute(2, 0, 1).float() / 255.0
            face_embedding = resnet(face_tensor.unsqueeze(0).to(device))
            name, distance = find_closest_match(face_embedding, database_embeddings, database_names)
            if name != "Unknown":
                base_name = name.split('_')[0]
                log_attendance(base_name)

            face_np = np.array(face_img)[:, :, ::-1].copy()

            glasses_path = os.path.join(os.getcwd(), "assets/glasses.png")
            glasses_img = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED) if os.path.exists(glasses_path) else None

            if glasses_img is not None:
                face_np = overlay_glasses_with_eyes(face_np, glasses_img, left_eye, right_eye, (x1,y1,x2,y2))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            display_name = name.split('_')[0]
            label = f"{display_name} ({distance:.2f})"

            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

    cv2.imshow('Face Recognition (RTSP)', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            left_eye = saved_landmarks[i][0]
            right_eye = saved_landmarks[i][1]

            input_name = input(f"Enter name for face #{i+1}: ").strip()
            if not input_name:
                print("Invalid name, skipping saving this face image")
                continue

            full_img = original_frame.copy()
            filename_full = f"{input_name}_full.jpg"
            save_path_full = os.path.join(FACE_DB_FOLDER, filename_full)
            cv2.imwrite(save_path_full, full_img)

            crop_img = original_frame[y1:y2, x1:x2]
            filename_crop = f"{input_name}_crop.jpg"
            save_path_crop = os.path.join(FACE_DB_FOLDER, filename_crop)
            cv2.imwrite(save_path_crop, crop_img)

            mask_path = os.path.join(os.getcwd(), "image/mask/blue.png")
            if os.path.exists(mask_path):
                mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if mask_img is not None:
                    crop_mask_img = overlay_mask_with_eyes(crop_img.copy(), mask_img, left_eye, right_eye, (x1,y1,x2,y2), scale=15.0)
                    filename_crop_mask = f"{input_name}_crop_mask.jpg"
                    save_path_crop_mask = os.path.join(FACE_DB_FOLDER, filename_crop_mask)
                    cv2.imwrite(save_path_crop_mask, crop_mask_img)
                else:
                    print("Failed to load mask image")
            else:
                print("Mask file not found:", mask_path)

            glasses_path = os.path.join(os.getcwd(), "image/glasses/glasses_black.png")
            if os.path.exists(glasses_path):
                glasses_img = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED)
                if glasses_img is not None:
                    crop_glasses_img = overlay_glasses_with_eyes(crop_img.copy(), glasses_img, left_eye, right_eye, (x1,y1,x2,y2))
                    filename_crop_glasses = f"{input_name}_crop_glasses.jpg"
                    save_path_crop_glasses = os.path.join(FACE_DB_FOLDER, filename_crop_glasses)
                    cv2.imwrite(save_path_crop_glasses, crop_glasses_img)
                else:
                    print("Failed to load glasses image")
            else:
                print("Glasses file not found:", glasses_path)

            print(f"Saved 4 image types for face: {input_name}")
            print("Reloading face database...")
            database_embeddings, database_names = load_face_database(FACE_DB_FOLDER, FACE_DB_CROP)
            print("Face database updated successfully")
        cv2.imshow('Face Recognition (RTSP)', frame)

    elif key == ord('d') and boxes is not None:
        input_name = input("Enter name for saving 20 face images: ").strip()
        if not input_name:
            print("Invalid name, cancel saving.")
            continue

        save_count = 0
        max_save = 20
        person_crop_folder = os.path.join(FACE_DB_CROP, input_name)
        os.makedirs(person_crop_folder, exist_ok=True)

        while save_count < max_save:
            ret, frame = video_stream.read()
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
            if boxes is not None:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    crop_img = frame[y1:y2, x1:x2]
                    filename = f"{input_name}_{save_count+1}.jpg"
                    save_path = os.path.join(person_crop_folder, filename)
                    cv2.imwrite(save_path, crop_img)
                    print(f"Saved image: {filename}")
                    save_count += 1

                    time.sleep(0.8)

                    if save_count >= max_save:
                        break

            cv2.imshow('Face Recognition (Saving)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print("Finished saving 20 images.")
        database_embeddings, database_names = load_face_database(FACE_DB_FOLDER, FACE_DB_CROP)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_stream.stop()
cv2.destroyAllWindows()
