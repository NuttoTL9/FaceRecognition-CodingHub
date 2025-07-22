import main

def test_main(monkeypatch):
    called = {}

    def mock_create_milvus_collection():
        called['create_milvus_collection'] = True

    def mock_reload_face_database():
        called['reload_face_database'] = True

    def mock_start_all_cameras():
        called['start_all_cameras'] = True
        class DummyThread:
            def join(self):
                called['thread_joined'] = True
        return [DummyThread(), DummyThread()]

    def mock_show_camera_frames():
        called['show_camera_frames'] = True

    monkeypatch.setattr(main, "create_milvus_collection", mock_create_milvus_collection)
    monkeypatch.setattr(main, "reload_face_database", mock_reload_face_database)
    monkeypatch.setattr(main, "start_all_cameras", mock_start_all_cameras)
    monkeypatch.setattr(main, "show_camera_frames", mock_show_camera_frames)

    main.main()

    assert called.get('create_milvus_collection')
    assert called.get('reload_face_database')
    assert called.get('start_all_cameras')
    assert called.get('show_camera_frames')
    assert called.get('thread_joined')
