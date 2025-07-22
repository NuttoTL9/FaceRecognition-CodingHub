import builtins
import pytest
from unittest.mock import patch, MagicMock
import streaming.registration as registration

@pytest.fixture
def dummy_frame():
    import numpy as np
    return np.zeros((224, 224, 3), dtype=np.uint8)

def test_capture_and_save_success(dummy_frame):
    inputs = iter([
        '1',          # เลือกกล้อง
        'EMP001',     # Employee ID
        'Test User'   # ชื่อ
    ])

    with patch('builtins.input', side_effect=lambda _: next(inputs)), \
         patch('streaming.registration.camera_streams', {'Camera-1': MagicMock()}), \
         patch('streaming.registration.camera_frames', {'Camera-1': dummy_frame}), \
         patch('streaming.registration.mtcnn') as mock_mtcnn, \
         patch('streaming.registration.preprocess_face') as mock_preprocess, \
         patch('streaming.registration.resnet') as mock_resnet, \
         patch('streaming.registration.encode_vector_with_grpc') as mock_encode:
        # Mock MTCNN detection to return a face box
        
        mock_mtcnn.detect.return_value = ([ [50, 50, 150, 150] ], None)
        import torch
        mock_preprocess.return_value = torch.zeros((1, 3, 160, 160))
        mock_resnet.return_value = torch.zeros((1, 512))
        mock_encode.return_value = "mock_grpc_response"

        response = registration.capture_and_save()

        mock_encode.assert_called_once()
        assert response == "mock_grpc_response"
