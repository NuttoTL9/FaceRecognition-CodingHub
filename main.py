import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import datetime
import requests

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
SHEETDB_API_URL = "https://sheetdb.io/api/v1/vq3gqcx2oz3kt"
attendance_status = {}
last_logged_time = {}
MIN_LOG_INTERVAL = 60  # seconds

def load_face_database(folder_path):
    embeddings = []
    names = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.png')):
            name = os.path.splitext(filename)[0].replace('_Mask', '')  # remove _Mask
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert('RGB')
            face_tensor = mtcnn(img)
            if face_tensor is not None:
                if face_tensor.dim() == 3:
                    face_tensor = face_tensor.unsqueeze(0)
                face_embedding = resnet(face_tensor.to(device))
                embeddings.append(face_embedding)
                names.append(name)
                print(f"✅ Loaded face: {name}")
            else:
                print(f"⚠️ No face detected in: {filename}")

    if not embeddings:
        print("❌ No faces found in the database! Please check the 'data/' folder.")
    return embeddings, names

print("Loading face database...")
database_embeddings, database_names = load_face_database(
    folder_path=r'C:\Users\arena\OneDrive\เอกสาร\CodingHub\facenet-pytorch\data')

def log_attendance(name):
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

    last_time = last_logged_time.get(name)
    if last_time and (now - last_time).total_seconds() < MIN_LOG_INTERVAL:
        return  # Skip if already logged recently

    if name not in attendance_status:
        status = "เข้างาน"
        attendance_status[name] = "in"
    else:
        if attendance_status[name] == "in":
            status = "เลิกงาน"
            attendance_status[name] = "out"
        else:
            status = "เข้างาน"
            attendance_status[name] = "in"

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
            print(f"📥 Logged {name} - {status} at {now_str}")
            last_logged_time[name] = now
        else:
            print("⚠️ Failed to log attendance:", response.text)
    except Exception as e:
        print("❌ Error sending data to SheetDB:", e)


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


cap = cv2.VideoCapture(0)
print("Starting camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    boxes, probs = mtcnn.detect(img)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            face_img = img.crop((x1, y1, x2, y2)).resize((160, 160))
            face_tensor = torch.tensor(np.array(face_img)).permute(2, 0, 1).float() / 255.0
            face_embedding = resnet(face_tensor.unsqueeze(0).to(device))
            name, distance = find_closest_match(face_embedding, database_embeddings, database_names)
            if name != "Unknown":
                log_attendance(name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{name} ({distance:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
