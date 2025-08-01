import cv2
import torch
import requests
from PIL import Image
from recognition.face_models import mtcnn, resnet
from recognition.face_utils import preprocess_face
from streaming.face_detection import camera_streams, camera_frames, reload_face_database
from config import FASTAPI_URL

def capture_and_save():
    try:

        cam_choice = input(f"เลือกกล้องที่จะใช้ (1 - {len(camera_streams)}): ").strip()
        if not cam_choice.isdigit() or int(cam_choice) < 1 or int(cam_choice) > len(camera_streams):
            print("❌ เลือกกล้องไม่ถูกต้อง")
            return

        cam_name = f"Camera-{cam_choice}"
        stream = camera_streams.get(cam_name)
        frame = camera_frames.get(cam_name)
        if not stream or frame is None:
            print("❌ ไม่พบกล้องหรือภาพจากกล้อง")
            return

        firstname = input("ชื่อจริง (Firstname): ").strip()
        lastname = input("นามสกุล (Lastname): ").strip()
        gender = input("เพศ (ชาย/หญิง): ").strip()
        date_of_joining = input("วันที่เริ่มงาน (YYYY-MM-DD): ").strip()
        date_of_birth = input("วันเกิด (YYYY-MM-DD): ").strip()
        company = input("ชื่อบริษัท (Company): ").strip()

        if not all([firstname, lastname, gender, date_of_joining, date_of_birth, company]):
            print("ข้อมูลไม่ครบถ้วน")
            return

        employee_payload = {
            "firstname": firstname,
            "lastname": lastname,
            "gender": gender,
            "date_of_joining": date_of_joining,
            "date_of_birth": date_of_birth,
            "company": company
        }

        print("ส่งคำขอสร้าง Employee...")
        emp_response = requests.post(f"{FASTAPI_URL}/api/resource/Employee", json=employee_payload)
        if not emp_response.ok:
            print("สร้าง Employee ไม่สำเร็จ:", emp_response.status_code, emp_response.text)
            return

        employee_id = emp_response.json().get("employee_id") or emp_response.json().get("data", {}).get("name")
        if not employee_id:
            print("ไม่ได้รับ employee_id จาก server")
            return
        print(f"ได้ employee_id: {employee_id}")

        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, _ = mtcnn.detect(img_pil)
        if boxes is None:
            print("ไม่พบใบหน้าในภาพ")
            return

        for box in boxes:
            face_tensor = preprocess_face(frame, box)
            if face_tensor is None:
                print("ไม่สามารถสร้าง face tensor ได้")
                continue

            with torch.no_grad():
                embedding = resnet(face_tensor)

            embedding_np = embedding.squeeze(0).cpu().numpy().astype("float32")

            vector_payload = {
                "employee_id": employee_id,
                "name": f"{firstname} {lastname}",
                "embedding": embedding_np.tolist()
            }

            print("ส่ง embedding ไปยัง FastAPI...")
            vector_response = requests.post(f"{FASTAPI_URL}/add_face_vector/", json=vector_payload)
            if not vector_response.ok:
                print("บันทึก embedding ไม่สำเร็จ:", vector_response.status_code, vector_response.text)
                return

            print("บันทึกข้อมูล embedding และพนักงานเรียบร้อย")
            reload_face_database()
            return vector_response.json()

    except Exception as e:
        print(f"เกิดข้อผิดพลาดขณะบันทึก: {e}")
