import os
import time
import datetime
import requests

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

RTSP_URL = "rtsp://admin:Codinghub12@192.168.1.64:554/Streaming/Channels/102"
CAMERA_INDEX = 0  # 0 for default webcam; change if using RTSP, replace with RTSP_URL
SHEETDB_API_URL = "https://sheetdb.io/api/v1/vq3gqcx2oz3kt"

BASE_DIR = os.path.dirname(__file__)
FACE_DB_FOLDER = os.path.join(BASE_DIR, 'data')
FACE_DB_CROP = os.path.join(BASE_DIR, 'cropdata')

MIN_LOG_INTERVAL = 60  # seconds between logs for the same person


# ---------------- Initialize Models ---------------- #

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
connections.connect("default", host="127.0.0.1", port="19530")


# ---------------- Global State ---------------- #

attendance_status = {}     # Track in/out status for each person
last_logged_time = {}      # Timestamp of last log for each person

# ---------------- Utility Functions ---------------- #

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
        index_params={
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
    )
    collection.load()
    return collection

def load_face_database_from_milvus():
    vectors = []
    names = []

    try:
        milvus_collection.load()
        results = milvus_collection.query(
            expr="",
            output_fields=["name", "embedding"],
            limit=10000  # หรือมากกว่านี้ตามที่ต้องการ
        )

        for item in results:
            name = item['name']
            embedding = torch.tensor(item['embedding']).to(device)
            vectors.append(embedding)
            names.append(name)

        if not vectors:
            return torch.empty(0, 512).to(device), []

        embeddings_tensor = torch.stack(vectors)
        return embeddings_tensor, names

    except Exception as e:
        print("❌ Failed to load from Milvus:", e)
        return torch.empty(0, 512).to(device), []

# def load_face_database(*folder_paths):
#     """
#     Load face embeddings and associated names from image folders.
#     Returns:
#         embeddings_tensor: torch.Tensor (N, 512)
#         names: list of str
#     """
#     embeddings = []
#     names = []

#     for folder_path in folder_paths:
#         for root, _, files in os.walk(folder_path):
#             for filename in files:
#                 if filename.lower().endswith(('.jpg', '.png')):
#                     name = os.path.splitext(filename)[0].split('_')[0]
#                     img_path = os.path.join(root, filename)
#                     img = Image.open(img_path).convert('RGB').resize((160, 160))

#                     face_tensor = mtcnn(img)
#                     if face_tensor is not None:
#                         if face_tensor.dim() == 3:
#                             face_tensor = face_tensor.unsqueeze(0)
#                         with torch.no_grad():
#                             face_embedding = resnet(face_tensor.to(device))
#                         embeddings.append(face_embedding.detach())
#                         names.append(name)
#                         print(f"✅ Loaded face: {name} from {img_path}")
#                     else:
#                         print(f"❌ No face detected in: {img_path}")

#     if not embeddings:
#         print("❌ No faces found in the database!")
#         return torch.empty(0, 512).to(device), []

#     embeddings_tensor = torch.cat(embeddings).to(device)  # shape (N, 512)
#     return embeddings_tensor, names


def log_attendance(name):
    """
    Log attendance to remote SheetDB if minimum log interval passed.
    """
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

    last_time = last_logged_time.get(name)
    if last_time and (now - last_time).total_seconds() < MIN_LOG_INTERVAL:
        return

    # Toggle in/out status
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


def find_closest_match(face_embedding, database_embeddings, database_names, threshold=1.0):
    """
    Compare face_embedding against database embeddings, return closest match.
    Returns:
        name (str): matched person's name or 'Unknown'
        distance (float): embedding distance
    """
    if database_embeddings.shape[0] == 0:
        return "Unknown", float('inf')

    distances = torch.norm(database_embeddings - face_embedding, dim=1)
    min_dist, min_idx = torch.min(distances, dim=0)

    if min_dist.item() > threshold:
        return "Unknown", min_dist.item()

    return database_names[min_idx], min_dist.item()


def preprocess_face(img_pil, box):
    """
    Crop face by bounding box and resize to model input size.
    Returns tensor ready for embedding extraction.
    """
    x1, y1, x2, y2 = [int(coord) for coord in box]
    face_img = img_pil.crop((x1, y1, x2, y2)).resize((160, 160))
    face_np = np.array(face_img)
    face_tensor = torch.tensor(face_np).permute(2, 0, 1).float() / 255.0
    return face_tensor.unsqueeze(0).to(device), face_np



# ---------------- Main Program ---------------- #

def main():
    print("Loading face database...")
    #database_embeddings, database_names = load_face_database(FACE_DB_FOLDER, FACE_DB_CROP)
    database_embeddings, database_names = load_face_database_from_milvus()

    print("Starting video stream...")
    # Use RTSP or webcam
    # video_stream = VideoStreamThread(RTSP_URL)
    video_stream = VideoStreamThread(CAMERA_INDEX)

    while True:
        ret, frame = video_stream.read()
        if not ret:
            print("No frame received from camera")
            continue

        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, probs, landmarks = mtcnn.detect(img_pil, landmarks=True)
        saved_landmarks = landmarks if landmarks is not None else []
        original_frame = frame.copy()

        if boxes is not None:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = [int(coord) for coord in box]
                left_eye = landmarks[i][0]
                right_eye = landmarks[i][1]

                face_tensor, face_np = preprocess_face(img_pil, box)

                with torch.no_grad():
                    face_embedding = resnet(face_tensor)

                name, distance = find_closest_match(face_embedding, database_embeddings, database_names)

                if name != "Unknown":
                    base_name = name.split('_')[0]
                    log_attendance(base_name)

                # Overlay glasses if available
                glasses_path = os.path.join(os.getcwd(), "assets/glasses.png")
                glasses_img = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED) if os.path.exists(glasses_path) else None

                if glasses_img is not None:
                    face_np = overlay_glasses_with_eyes(face_np, glasses_img, left_eye, right_eye, (x1, y1, x2, y2))

                # Draw bounding box and label on original frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                display_name = name.split('_')[0]
                label = f"{display_name} ({distance:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

            cv2.imshow('Face Recognition', frame)

        key = cv2.waitKey(1) & 0xFF

        # 's' key: save detected faces with metadata (name input)
        if key == ord('s') and boxes is not None:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = [int(coord) for coord in box]
                left_eye = saved_landmarks[i][0]
                right_eye = saved_landmarks[i][1]

                input_name = input(f"Enter name for face #{i+1}: ").strip()
                if not input_name:
                    print("Invalid name, skipping saving this face image")
                    continue

                # Save full frame image
                filename_full = f"{input_name}_full.jpg"
                cv2.imwrite(os.path.join(FACE_DB_FOLDER, filename_full), original_frame)
                # --- Convert crop_img to embedding --- #
                crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(crop_rgb).resize((160, 160))
                face_tensor = mtcnn(face_pil)
                if face_tensor is not None:
                    with torch.no_grad():
                        embedding = resnet(face_tensor.unsqueeze(0).to(device)).cpu().numpy()[0].tolist()

                    # --- Insert into Milvus --- #
                    milvus_collection.insert([ [input_name], [embedding] ])
                    print(f"📡 Sent vector of '{input_name}' to Milvus.")
                else:
                    print("❌ Failed to extract embedding from cropped image.")

                # Save cropped face image
                crop_img = original_frame[y1:y2, x1:x2]
                filename_crop = f"{input_name}_crop.jpg"
                cv2.imwrite(os.path.join(FACE_DB_FOLDER, filename_crop), crop_img)
                # --- Convert crop_img to embedding --- #
                crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(crop_rgb).resize((160, 160))
                face_tensor = mtcnn(face_pil)
                if face_tensor is not None:
                    with torch.no_grad():
                        embedding = resnet(face_tensor.unsqueeze(0).to(device)).cpu().numpy()[0].tolist()

                    # --- Insert into Milvus --- #
                    milvus_collection.insert([ [input_name], [embedding] ])
                    print(f"📡 Sent vector of '{input_name}' to Milvus.")
                else:
                    print("❌ Failed to extract embedding from cropped image.")

                # Save mask overlay if available
                mask_path = os.path.join(os.getcwd(), "image/mask/blue.png")
                if os.path.exists(mask_path):
                    mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                    if mask_img is not None:
                        crop_mask_img = overlay_mask_with_eyes(crop_img.copy(), mask_img, left_eye, right_eye, (x1, y1, x2, y2), scale=15.0)
                        filename_crop_mask = f"{input_name}_crop_mask.jpg"
                        cv2.imwrite(os.path.join(FACE_DB_FOLDER, filename_crop_mask), crop_mask_img)
                            # --- Convert crop_img to embedding --- #
                        crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                        face_pil = Image.fromarray(crop_rgb).resize((160, 160))
                        face_tensor = mtcnn(face_pil)
                        if face_tensor is not None:
                            with torch.no_grad():
                                embedding = resnet(face_tensor.unsqueeze(0).to(device)).cpu().numpy()[0].tolist()

                            # --- Insert into Milvus --- #
                            milvus_collection.insert([ [input_name], [embedding] ])
                            print(f"📡 Sent vector of '{input_name}' to Milvus.")
                        else:
                            print("❌ Failed to extract embedding from cropped image.")
                    else:
                        print("Failed to load mask image")
                else:
                    print(f"Mask file not found: {mask_path}")

                # Save glasses overlay if available
                glasses_path = os.path.join(os.getcwd(), "image/glasses/glasses_black.png")
                if os.path.exists(glasses_path):
                    glasses_img = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED)
                    if glasses_img is not None:
                        crop_glasses_img = overlay_glasses_with_eyes(crop_img.copy(), glasses_img, left_eye, right_eye, (x1, y1, x2, y2))
                        filename_crop_glasses = f"{input_name}_crop_glasses.jpg"
                        cv2.imwrite(os.path.join(FACE_DB_FOLDER, filename_crop_glasses), crop_glasses_img)
                        # --- Convert crop_img to embedding --- #
                        crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                        face_pil = Image.fromarray(crop_rgb).resize((160, 160))
                        face_tensor = mtcnn(face_pil)
                        if face_tensor is not None:
                            with torch.no_grad():
                                embedding = resnet(face_tensor.unsqueeze(0).to(device)).cpu().numpy()[0].tolist()

                            # --- Insert into Milvus --- #
                            milvus_collection.insert([ [input_name], [embedding] ])
                            print(f"📡 Sent vector of '{input_name}' to Milvus.")
                        else:
                            print("❌ Failed to extract embedding from cropped image.")
                    else:
                        print("Failed to load glasses image")
                else:
                    print(f"Glasses file not found: {glasses_path}")

                print(f"Saved 4 image types for face: {input_name}")

            # Reload database after saving new faces
            print("Reloading face database...")
            #database_embeddings, database_names = load_face_database(FACE_DB_FOLDER, FACE_DB_CROP)
            database_embeddings, database_names = load_face_database_from_milvus()
            print("Face database updated successfully")

        # 'd' key: save multiple cropped face images for one person
        elif key == ord('d'):
            input_name = input("Enter name for 5 vector captures: ").strip()
            if not input_name:
                print("Invalid name, cancel capture.")
                continue

            save_count = 0
            max_save = 5

            print(f"Capturing {max_save} face vectors for '{input_name}'... Press 'q' to stop early.")

            while save_count < max_save:
                ret, frame = video_stream.read()
                if not ret:
                    print("No frame received.")
                    break

                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                boxes, probs, landmarks = mtcnn.detect(img_pil, landmarks=True)

                if boxes is not None:
                    for box in boxes:
                        if save_count >= max_save:
                            break
                        x1, y1, x2, y2 = [int(coord) for coord in box]
                        crop_img = frame[y1:y2, x1:x2]

                        # Convert crop to embedding
                        crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                        face_pil = Image.fromarray(crop_rgb).resize((160, 160))
                        face_tensor = mtcnn(face_pil)

                        if face_tensor is not None:
                            with torch.no_grad():
                                embedding = resnet(face_tensor.unsqueeze(0).to(device)).cpu().numpy()[0].tolist()

                            # Insert to Milvus
                            milvus_collection.insert([[input_name], [embedding]])
                            print(f"📡 Inserted vector #{save_count + 1} for '{input_name}'")
                            save_count += 1
                            time.sleep(0.5)
                        else:
                            print("❌ Failed to extract face embedding")

                cv2.imshow("Capturing", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Manual stop.")
                    break

            print("✅ Done. Reloading face database...")
            database_embeddings, database_names = load_face_database_from_milvus()

        elif key == ord('c'):

            try:
                milvus_collection.flush()  # 🔄 sync data ก่อน

                if milvus_collection.num_entities == 0:
                    print("⚠️ ไม่มี vector อยู่ใน Milvus collection ตอนนี้")
                else:
                    print(f"📦 Total vectors: {milvus_collection.num_entities}")
                    results = milvus_collection.query(
                        expr="",
                        output_fields=["name", "embedding"],
                        limit=5
                    )
                    print("📄 ตัวอย่างข้อมูล (สูงสุด 5 รายการ):")
                    for i, item in enumerate(results, start=1):
                        name = item.get("name", "Unknown")
                        print(f"{i}. Name: {name}")
                        # หากอยากดู embedding ด้วย:
                        print(f"   Embedding: {item['embedding'][:5]}...")
            except Exception as e:
                print("❌ Error while querying Milvus:", e)

        # 'q' key: quit program
        if key == ord('q'):
            print("Exiting...")
            break

    video_stream.stop()
    cv2.destroyAllWindows()

milvus_collection = create_milvus_collection()
if __name__ == "__main__":
    main()
