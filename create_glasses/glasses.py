import os
import numpy as np
import face_recognition
from PIL import Image
import cv2


def create_glasses(image_path, glasses_path):
    key_facial_feature = ('left_eye', 'right_eye')

    face_image_np = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(face_image_np, model='hog')
    face_landmarks = face_recognition.face_landmarks(face_image_np, face_locations)
    face_img = Image.open(image_path)
    glasses_img = Image.open(glasses_path)

    found_face = False
    for face_landmark in face_landmarks:
        if all(feature in face_landmark for feature in key_facial_feature):
            found_face = True
            glasses_face(face_landmark, glasses_img, face_img, image_path)
    
    if not found_face:
        print(f"No face found in {image_path}")




def overlay_transparent(background, overlay, x, y):
    h, w = overlay.shape[:2]
    
    if x + w > background.shape[1] or y + h > background.shape[0]:
        return background
    
    overlay_img = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background_crop = background[y:y+h, x:x+w]

    background_crop[:] = (1.0 - mask) * background_crop + mask * overlay_img
    return background


def create_glasses_single(image, glasses_path):
    """รับ image ที่เปิดมาแล้ว และวางแว่นลงไป"""
    glasses = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED)
    if glasses is None:
        print(f"Cannot read glasses: {glasses_path}")
        return image

    # ปรับขนาดแว่นพอดีกับใบหน้า สมมุติวางช่วงตา
    scale_factor = 0.4
    new_w = int(image.shape[1] * scale_factor)
    new_h = int(glasses.shape[0] * (new_w / glasses.shape[1]))
    glasses = cv2.resize(glasses, (new_w, new_h))

    x = (image.shape[1] - new_w) // 2
    y = int(image.shape[0] * 0.35)

    image = overlay_transparent(image, glasses, x, y)
    return image


def glasses_face(face_landmark, glasses_img, face_img, image_path):
    left_eye = face_landmark['left_eye']
    right_eye = face_landmark['right_eye']

    # จุดศูนย์กลางตา
    left_eye_center = np.mean(left_eye, axis=0).astype(int)
    right_eye_center = np.mean(right_eye, axis=0).astype(int)

    # มุมระหว่างดวงตา
    dx = right_eye_center[0] - left_eye_center[0]
    dy = right_eye_center[1] - left_eye_center[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # ระยะห่างระหว่างตา
    eye_width = int(np.linalg.norm(right_eye_center - left_eye_center))
    glasses_width = int(eye_width * 2.0)  # แว่นกว้างกว่าระยะตา

    # ลดความสูงของแว่นลง เช่น เอาแค่ 70% ของอัตราส่วนเดิม
    aspect_ratio = glasses_img.height / glasses_img.width
    desired_aspect_ratio = aspect_ratio * 0.7  # ปรับลดความสูง 30%
    glasses_height = int(glasses_width * desired_aspect_ratio)

    resized_glasses = glasses_img.resize((glasses_width, glasses_height), Image.LANCZOS)

    # หมุนแว่นตามมุมที่คำนวณได้
    rotated_glasses = resized_glasses.rotate(angle, expand=True)

    # วางแว่นให้อยู่กึ่งกลางระหว่างตา ขยับขึ้นเล็กน้อย
    center_x = (left_eye_center[0] + right_eye_center[0]) // 2
    center_y = (left_eye_center[1] + right_eye_center[1]) // 2 - glasses_height // 3

    box_x = center_x - rotated_glasses.width // 2
    box_y = center_y - rotated_glasses.height // 2

    # วางแว่นบนภาพ พร้อม Alpha Mask
    face_img.paste(rotated_glasses, (box_x, box_y), rotated_glasses)

    # สร้างโฟลเดอร์และบันทึกภาพ
    path = image_path.split(os.path.sep)
    name = path[-2]
    image = path[-1]
    save_dir = f'dataset_with_glasses/{name}'
    os.makedirs(save_dir, exist_ok=True)
    face_img.save(f'{save_dir}/{image}')
