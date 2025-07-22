import pytest
import torch
import numpy as np
import cv2
import time
from unittest.mock import patch, MagicMock

from streaming.face_detection import (
    extract_valid_faces,
    get_embeddings,
    identify_and_log_faces,
    log_recognition_event,
)

from streaming import face_detection

def test_extract_valid_faces(dummy_image):
    boxes = [[60, 60, 220, 220]]
    tensors, valid_boxes = extract_valid_faces(dummy_image, boxes)
    assert len(tensors) == 1
    assert len(valid_boxes) == 1

@pytest.fixture
def dummy_image():
    # สร้างภาพ RGB dummy 300x300
    return np.ones((300, 300, 3), dtype=np.uint8) * 255


def test_extract_valid_faces_valid_and_small_boxes(dummy_image):
    # กล่องใหญ่เกิน MIN_FACE_AREA และกล่องเล็กเกิน MIN_FACE_AREA
    # สมมติ MIN_FACE_AREA = 10000 เพื่อทดสอบ (แก้ตามจริง)
    face_detection.MIN_FACE_AREA = 10000

    boxes = [
        [0, 0, 10, 10],        # area = 100 < 10000 -> กรองทิ้ง
        [0, 0, 150, 150],      # area = 22500 > 10000 -> รับไว้
    ]

    # mock preprocess_face ให้ return tensor
    with patch("streaming.face_detection.preprocess_face") as mock_preprocess:
        mock_preprocess.return_value = torch.randn(1, 3, 112, 112)

        tensors, valid_boxes = extract_valid_faces(dummy_image, boxes)

        assert len(tensors) == 1
        assert len(valid_boxes) == 1
        assert valid_boxes[0] == [0, 0, 150, 150]

        mock_preprocess.assert_called_once_with(dummy_image, boxes[1])

def test_extract_valid_faces_preprocess_none(dummy_image):
    face_detection.MIN_FACE_AREA = 0  # ให้ผ่าน area test

    boxes = [[0, 0, 50, 50]]
    with patch("streaming.face_detection.preprocess_face") as mock_preprocess:
        mock_preprocess.return_value = None  # กรองทิ้ง

        tensors, valid_boxes = extract_valid_faces(dummy_image, boxes)

        assert len(tensors) == 0
        assert len(valid_boxes) == 0

def test_get_embeddings_returns_tensor():
    # สร้าง face_tensors เป็น list ของ tensor (batch size 1)
    face_tensors = [torch.randn(1, 3, 112, 112), torch.randn(1, 3, 112, 112)]

    # mock resnet model ให้ return tensor embedding (batch 2, 512)
    with patch("streaming.face_detection.resnet") as mock_resnet:
        mock_resnet.return_value = torch.randn(2, 512)

        embeddings = get_embeddings(face_tensors)

        mock_resnet.assert_called_once()
        assert embeddings.shape == (2, 512)

def test_identify_and_log_faces_calls_log_and_draws_on_frame():
    frame = np.ones((300, 300, 3), dtype=np.uint8)
    embeddings = torch.randn(2, 512)
    boxes = [[10, 10, 100, 100], [110, 110, 200, 200]]

    # เตรียม shared data
    face_detection.shared_embeddings = torch.randn(5, 512)
    face_detection.shared_names = ["A", "B", "C", "D", "E"]
    face_detection.shared_employee_ids = ["idA", "idB", "idC", "idD", "idE"]

    # mock find_closest_match ให้ return (employee_id, name, distance)
    with patch("streaming.face_detection.find_closest_match") as mock_find:
        mock_find.side_effect = [
            ("idA", "Alice", 0.5),
            ("Unknown", "Unknown", 0.9),
        ]
        # mock log_recognition_event เพื่อเช็คว่าถูกเรียก
        with patch("streaming.face_detection.log_recognition_event") as mock_log:
            identify_and_log_faces(frame, embeddings, boxes)

            # เช็ค log_recognition_event เรียกแค่ครั้งเดียว (สำหรับ idA ที่ไม่ใช่ Unknown และ distance < 0.7)
            mock_log.assert_called_once_with("idA", "Alice")

    # เช็คว่า frame ถูกวาดกล่องและข้อความ (ไม่ตรวจละเอียดแต่ frame ต้องไม่เปลี่ยนขนาด)
    assert frame.shape == (300, 300, 3)

@patch("streaming.face_detection.requests.post")
def test_log_recognition_event_success(mock_post):
    face_detection.MIN_LOG_INTERVAL = 0  # ให้ log ได้ทุกครั้ง
    face_detection.last_log_times.clear()
    face_detection.person_states.clear()

    mock_response = MagicMock()
    mock_response.ok = True
    mock_post.return_value = mock_response

    # เรียกครั้งแรก ต้อง log สำเร็จ
    face_detection.log_recognition_event("emp1", "Alice")
    assert face_detection.person_states["emp1"] in ["in", "out"]
    assert "emp1" in face_detection.last_log_times

@patch("streaming.face_detection.requests.post")
def test_log_recognition_event_too_soon(mock_post):
    face_detection.MIN_LOG_INTERVAL = 10
    face_detection.last_log_times["emp1"] = time.time()

    face_detection.log_recognition_event("emp1", "Alice")
    mock_post.assert_not_called()  # ไม่เรียก log เพราะเร็วเกินไป

@patch("streaming.face_detection.requests.post")
def test_log_recognition_event_failure(mock_post):
    face_detection.MIN_LOG_INTERVAL = 0
    face_detection.last_log_times.clear()
    face_detection.person_states.clear()

    mock_response = MagicMock()
    mock_response.ok = False
    mock_response.status_code = 500
    mock_response.text = "error"
    mock_post.return_value = mock_response

    face_detection.log_recognition_event("emp1", "Alice")
    # ไม่มี error ต้องจับ exception และ print

