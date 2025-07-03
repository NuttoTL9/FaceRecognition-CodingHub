import cv2
import numpy as np

def overlay_mask_with_eyes(face_img, mask_img, left_eye, right_eye, box, scale=10.0):
    """
    face_img: numpy BGR (crop face)
    mask_img: numpy RGBA (หน้ากากอนามัย)
    left_eye, right_eye: (x,y) ตำแหน่งดวงตาบนรูปเต็ม
    box: (x1,y1,x2,y2) กรอบใบหน้า
    scale: ตัวคูณขนาดแมส

    คืนค่า: face_img ที่วางแมสแล้ว
    """
    x1, y1, x2, y2 = box
    left_eye_rel = (left_eye[0] - x1, left_eye[1] - y1)
    right_eye_rel = (right_eye[0] - x1, right_eye[1] - y1)

    # ระยะห่างตา
    eye_dist = np.linalg.norm(np.array(right_eye_rel) - np.array(left_eye_rel))

    # ขนาดแมส (ใช้ scale ที่กำหนด)
    mask_w = int(eye_dist * scale)
    mask_h = int(mask_img.shape[0] * (mask_w / mask_img.shape[1]))

    # ย่อแมส
    resized_mask = cv2.resize(mask_img, (mask_w, mask_h), interpolation=cv2.INTER_AREA)

    # ตำแหน่งแมส ให้อยู่กลางใบหน้า และเลื่อนลงนิดหน่อย
    center_x = int((left_eye_rel[0] + right_eye_rel[0]) / 2)
    center_y = int((left_eye_rel[1] + right_eye_rel[1]) / 2)

    x_offset = int(center_x - mask_w / 2)
    y_offset = int(center_y + eye_dist * 0.5)

    # ตรวจสอบขอบเขตไม่เกินใบหน้า
    x_offset = max(0, x_offset)
    y_offset = max(0, y_offset)
    mask_w = min(mask_w, face_img.shape[1] - x_offset)
    mask_h = min(mask_h, face_img.shape[0] - y_offset)
    resized_mask = cv2.resize(mask_img, (mask_w, mask_h), interpolation=cv2.INTER_AREA)

    # แยก alpha และ BGR
    b, g, r, a = cv2.split(resized_mask)
    overlay_color = cv2.merge((b, g, r))
    mask_alpha = cv2.merge((a, a, a))

    roi = face_img[y_offset:y_offset+mask_h, x_offset:x_offset+mask_w]

    alpha = mask_alpha.astype(float) / 255
    overlay_color = overlay_color.astype(float)
    roi = roi.astype(float)

    blended = cv2.multiply(alpha, overlay_color) + cv2.multiply(1 - alpha, roi)
    blended = blended.astype(np.uint8)

    face_img[y_offset:y_offset+mask_h, x_offset:x_offset+mask_w] = blended

    return face_img
