# test_videostreamthread.py

import pytest
from unittest.mock import patch, MagicMock
import threading
import numpy as np
import time

from videostreamthread import videostreamthread

# ---------------------------
# Test สำหรับ OpenCV VideoCapture
# ---------------------------

@patch('cv2.VideoCapture')
def test_init_opencv_success(mock_video_capture):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    mock_video_capture.return_value = mock_cap

    stream = videostreamthread(0, width=320, height=240)
    assert stream.rtsp_url == 0
    assert stream.width == 320
    assert stream.height == 240
    assert stream.thread.is_alive()
    stream.stop()

@patch('cv2.VideoCapture')
def test_init_opencv_fail(mock_video_capture):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_video_capture.return_value = mock_cap
    with pytest.raises(ValueError, match="ไม่สามารถเปิดกล้อง notebook ด้วย OpenCV ได้"):
        videostreamthread(0)

# ---------------------------
# Test สำหรับ FFmpeg RTSP Stream
# ---------------------------

@patch('subprocess.Popen')
@patch('shutil.which', return_value='/usr/bin/ffmpeg')
def test_init_ffmpeg_success(_, mock_popen):  # ไม่ใช้ mock_which => ใช้ _
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.stdout.read.return_value = b'\x00' * (320 * 240 * 3)
    mock_popen.return_value = mock_proc

    stream = videostreamthread('rtsp://dummy_url', width=320, height=240)
    assert stream.rtsp_url == 'rtsp://dummy_url'
    assert stream.thread.is_alive()
    stream.stop()

@patch('shutil.which', return_value=None)
def test_init_ffmpeg_no_ffmpeg(_):  # ไม่ใช้ mock_which
    with pytest.raises(EnvironmentError, match="FFmpeg ไม่พบใน PATH"):
        videostreamthread('rtsp://dummy_url')

@patch('subprocess.Popen', side_effect=Exception('Popen failed'))
@patch('shutil.which', return_value='/usr/bin/ffmpeg')
def test_init_ffmpeg_popen_fail(_, mock_popen):
    with pytest.raises(ValueError, match="ไม่สามารถเปิด FFmpeg subprocess ได้"):
        videostreamthread('rtsp://dummy_url')

# ---------------------------
# Test ฟังก์ชัน update_opencv (thread loop)
# ---------------------------

@patch('cv2.VideoCapture')
def test_update_opencv_reads_frame(mock_video_capture):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap.read.side_effect = [(True, frame), (False, None), (True, frame)]
    mock_video_capture.return_value = mock_cap

    stream = videostreamthread(0, width=320, height=240)
    time.sleep(0.1)
    ret, frame = stream.read()
    assert ret is True
    assert frame.shape == (240, 320, 3)
    stream.stop()

# ---------------------------
# Test ฟังก์ชัน update_ffmpeg (thread loop)
# ---------------------------

@patch('subprocess.Popen')
@patch('shutil.which', return_value='/usr/bin/ffmpeg')
def test_update_ffmpeg_reads_frame(_, mock_popen):
    mock_proc = MagicMock()
    mock_proc.poll.side_effect = [None, None, 0]
    mock_proc.stdout.read.side_effect = [
        b'\x00' * (320 * 240 * 3),
        b'\x00' * (320 * 240 * 3),
        b''
    ]
    mock_popen.return_value = mock_proc

    stream = videostreamthread('rtsp://dummy_url', width=320, height=240)
    time.sleep(0.2)
    ret, frame = stream.read()
    assert ret is True
    assert frame.shape == (240, 320, 3)
    stream.stop()

# ---------------------------
# Test การปิดและปล่อย resource ด้วย stop()
# ---------------------------

@patch('subprocess.Popen')
@patch('cv2.VideoCapture')
@patch('shutil.which', return_value='C:\\ffmpeg\\bin\\ffmpeg.exe')
def test_stop_releases_resources(_, mock_video_capture, mock_popen):
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_popen.return_value = mock_proc

    stream = videostreamthread('rtsp://dummy_url', width=320, height=240)
    stream.stop()
    mock_proc.terminate.assert_called_once()
    mock_proc.wait.assert_called_once()

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_video_capture.return_value = mock_cap

    stream2 = videostreamthread(0)
    stream2.stop()
    mock_cap.release.assert_called_once()

# ---------------------------
# Test ฟังก์ชัน read() เมื่อยังไม่มี frame
# ---------------------------

def test_read_no_frame_yet():
    stream = videostreamthread.__new__(videostreamthread)
    stream.lock = threading.Lock()
    stream.frame = None
    stream.ret = False
    ret, frame = stream.read()
    assert ret is False
    assert frame is None
