import cv2
import torch
from PIL import Image
from recognition.face_models import mtcnn, resnet
from recognition.face_utils import preprocess_face
from streaming.face_detection import camera_streams, camera_frames, reload_face_database
from grpc_client.milvus_grpc_utils import encode_vector_with_grpc

def capture_and_save():
    try:
        cam_choice = input(f"เลือกกล้องที่จะใช้ (1 - {len(camera_streams)}): ").strip()
        if not cam_choice.isdigit() or int(cam_choice) < 1 or int(cam_choice) > len(camera_streams):
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
            print("ข้อมูลไม่ถูกต้อง")
            return

        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, _ = mtcnn.detect(img_pil)
        if boxes is None:
            print("ไม่พบใบหน้า")
            return

        for box in boxes:
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
