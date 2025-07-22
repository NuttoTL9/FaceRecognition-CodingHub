import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from streaming.face_detection import (
    process_camera,
    extract_valid_faces,
    get_embeddings,
    identify_and_log_faces,
    log_recognition_event,
)

from streaming import face_detection

@pytest.fixture
def dummy_image():
    return np.ones((300, 300, 3), dtype=np.uint8) * 255

# Dummy videostreamthread แทนของจริง (ให้ read() คืนภาพ dummy)
class DummyVideoStream:
    def __init__(self, rtsp_url, width=320, height=240):
        self.ret = True
        self.frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        self.stopped = False

    def read(self):
        return self.ret, self.frame.copy()

    def stop(self):
        self.stopped = True

# Mock videostreamthread ในโมดูล face_detection ให้เป็น DummyVideoStream
@patch("streaming.face_detection.videostreamthread", new=DummyVideoStream)
def test_process_camera_runs_without_camera(dummy_image):
    # เรียก process_camera แล้วมันจะใช้ DummyVideoStream แทนจริงๆ
    # ต้องทำแบบนี้เพื่อให้ loop ไม่ติด infinite
    import threading

    # รัน process_camera ใน thread แยก แล้วหยุดหลัง 1 วินาที
    def run_camera():
        face_detection.should_exit[0] = False
        process_camera("dummy_rtsp_url", "test_window")

    t = threading.Thread(target=run_camera)
    t.start()

    # ให้มันทำงานซักพัก แล้วสั่งหยุด
    import time
    time.sleep(1)
    face_detection.should_exit[0] = True
    t.join(timeout=2)

    assert "test_window" in face_detection.camera_frames
    frame = face_detection.camera_frames["test_window"]
    assert frame is not None
    assert frame.shape[2] == 3

# --- เทสอื่นๆ ตามที่คุณเขียนไป ---

@pytest.mark.parametrize("boxes,expected_count", [
    ([[60, 60, 220, 220]], 1),
])
def test_extract_valid_faces(dummy_image, boxes, expected_count):
    tensors, valid_boxes = extract_valid_faces(dummy_image, boxes)
    assert len(tensors) == expected_count
    assert len(valid_boxes) == expected_count

def test_extract_valid_faces_valid_and_small_boxes(dummy_image):
    face_detection.MIN_FACE_AREA = 10000

    boxes = [
        [0, 0, 10, 10],
        [0, 0, 150, 150],
    ]

    with patch("streaming.face_detection.preprocess_face") as mock_preprocess:
        mock_preprocess.return_value = torch.randn(1, 3, 112, 112)

        tensors, valid_boxes = extract_valid_faces(dummy_image, boxes)

        assert len(tensors) == 1
        assert len(valid_boxes) == 1
        assert valid_boxes[0] == [0, 0, 150, 150]

def test_extract_valid_faces_preprocess_none(dummy_image):
    face_detection.MIN_FACE_AREA = 0

    boxes = [[0, 0, 50, 50]]
    with patch("streaming.face_detection.preprocess_face") as mock_preprocess:
        mock_preprocess.return_value = None

        tensors, valid_boxes = extract_valid_faces(dummy_image, boxes)

        assert len(tensors) == 0
        assert len(valid_boxes) == 0

def test_get_embeddings_returns_tensor():
    face_tensors = [torch.randn(1, 3, 112, 112), torch.randn(1, 3, 112, 112)]

    with patch("streaming.face_detection.resnet") as mock_resnet:
        mock_resnet.return_value = torch.randn(2, 512)

        embeddings = get_embeddings(face_tensors)

        assert embeddings.shape == (2, 512)

def test_identify_and_log_faces_calls_log_and_draws_on_frame():
    frame = np.ones((300, 300, 3), dtype=np.uint8)
    embeddings = torch.randn(2, 512)
    boxes = [[10, 10, 100, 100], [110, 110, 200, 200]]

    face_detection.shared_embeddings = torch.randn(5, 512)
    face_detection.shared_names = ["A", "B", "C", "D", "E"]
    face_detection.shared_employee_ids = ["idA", "idB", "idC", "idD", "idE"]

    with patch("streaming.face_detection.find_closest_match") as mock_find:
        mock_find.side_effect = [
            ("idA", "Alice", 0.5),
            ("Unknown", "Unknown", 0.9),
        ]
        with patch("streaming.face_detection.log_recognition_event") as mock_log:
            identify_and_log_faces(frame, embeddings, boxes)
            mock_log.assert_called_once_with("idA", "Alice")

    assert frame.shape == (300, 300, 3)

import time
from unittest.mock import MagicMock
import requests

@patch("streaming.face_detection.requests.post")
def test_log_recognition_event_success(mock_post):
    face_detection.MIN_LOG_INTERVAL = 0
    face_detection.last_log_times.clear()
    face_detection.person_states.clear()

    mock_response = MagicMock()
    mock_response.ok = True
    mock_post.return_value = mock_response

    face_detection.log_recognition_event("emp1", "Alice")
    assert face_detection.person_states["emp1"] in ["in", "out"]
    assert "emp1" in face_detection.last_log_times

@patch("streaming.face_detection.requests.post")
def test_log_recognition_event_too_soon(mock_post):
    face_detection.MIN_LOG_INTERVAL = 10
    face_detection.last_log_times["emp1"] = time.time()

    face_detection.log_recognition_event("emp1", "Alice")
    mock_post.assert_not_called()

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
    # ควรไม่มี error เกิดขึ้น
