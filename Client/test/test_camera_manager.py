from streaming import camera_manager

def test_start_all_cameras(monkeypatch):
    monkeypatch.setattr(camera_manager, "RTSP_URLS", ["0"])
    threads = camera_manager.start_all_cameras()
    assert isinstance(threads, list)
