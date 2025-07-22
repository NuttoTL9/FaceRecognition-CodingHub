import torch
import cv2

from config import DEVICE

def preprocess_face(img_np, box):
    x1, y1, x2, y2 = [int(coord) for coord in box]
    h, w, _ = img_np.shape
    x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
    y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    face_img = img_np[y1:y2, x1:x2]
    if face_img.size == 0:
        return None
    face_img = cv2.resize(face_img, (160, 160))
    face_tensor = torch.from_numpy(face_img).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
    return face_tensor

def find_closest_match(face_embedding, db_embeddings, db_employee_ids, db_names, threshold=1.0):
    if db_embeddings.shape[0] == 0:
        return "Unknown", "Unknown", float('inf')
    distances = torch.norm(db_embeddings - face_embedding, dim=1)
    min_dist, idx = torch.min(distances, dim=0)
    if min_dist.item() <= threshold:
        return db_employee_ids[idx], db_names[idx], min_dist.item()
    return "Unknown", "Unknown", min_dist.item()
