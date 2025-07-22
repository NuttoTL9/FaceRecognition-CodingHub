import cv2
import numpy as np

def overlay_glasses_with_eyes(face_img, glasses_img, left_eye, right_eye, box):
    """
    face_img: numpy BGR (ใบหน้าขนาด crop)
    glasses_img: numpy RGBA (แว่นตา)
    left_eye, right_eye: (x,y) ตำแหน่งดวงตาบนภาพเต็ม (float)
    box: (x1,y1,x2,y2) กล่องหน้า (int)

    คืนค่า: face_img ที่วางแว่นแล้ว
    """

    x1, y1, _x2, _y2 = box

    # ปรับตำแหน่งดวงตาให้สัมพันธ์กับ crop face
    left_eye_rel = (left_eye[0] - x1, left_eye[1] - y1)
    right_eye_rel = (right_eye[0] - x1, right_eye[1] - y1)

    # ระยะห่างระหว่างตาซ้าย-ขวา
    eye_dist = np.linalg.norm(np.array(right_eye_rel) - np.array(left_eye_rel))

    # ขนาดแว่น (กว้างเท่ากับ eye_dist * scale)
    glasses_w = int(eye_dist * 2.0)  # ขยายกว้างกว่า eye_dist เล็กน้อย
    scale = glasses_w / glasses_img.shape[1]
    glasses_h = int(glasses_img.shape[0] * scale)

    # ย่อขนาดแว่น
    resized_glasses = cv2.resize(glasses_img, (glasses_w, glasses_h), interpolation=cv2.INTER_AREA)

    # ตำแหน่งแว่นให้อยู่กึ่งกลางระหว่างตาซ้าย-ขวา และเลื่อนขึ้นนิดหน่อย (ปรับตามต้องการ)
    center_x = int((left_eye_rel[0] + right_eye_rel[0]) / 2)
    center_y = int((left_eye_rel[1] + right_eye_rel[1]) / 2)

    x_offset = int(center_x - glasses_w / 2)
    y_offset = int(center_y - glasses_h / 2.5)  # เลื่อนขึ้นนิดหน่อย

    # ตรวจสอบขอบเขตไม่เกินใบหน้า
    if x_offset < 0:
        x_offset = 0
    if y_offset < 0:
        y_offset = 0
    if x_offset + glasses_w > face_img.shape[1]:
        glasses_w = face_img.shape[1] - x_offset
        resized_glasses = cv2.resize(glasses_img, (glasses_w, glasses_h), interpolation=cv2.INTER_AREA)
    if y_offset + glasses_h > face_img.shape[0]:
        glasses_h = face_img.shape[0] - y_offset
        resized_glasses = cv2.resize(glasses_img, (glasses_w, glasses_h), interpolation=cv2.INTER_AREA)

    # แยก alpha และ BGR ของแว่น
    b, g, r, a = cv2.split(resized_glasses)
    overlay_color = cv2.merge((b, g, r))
    mask = cv2.merge((a, a, a))

    roi = face_img[y_offset:y_offset+glasses_h, x_offset:x_offset+glasses_w]

    alpha = mask.astype(float) / 255
    overlay_color = overlay_color.astype(float)
    roi = roi.astype(float)

    blended = cv2.multiply(alpha, overlay_color) + cv2.multiply(1 - alpha, roi)
    blended = blended.astype(np.uint8)

    face_img[y_offset:y_offset+glasses_h, x_offset:x_offset+glasses_w] = blended

    return face_img
