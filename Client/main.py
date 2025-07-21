from config import RTSP_URLS
from database.milvus_database import create_milvus_collection
from streaming.camera_manager import start_all_cameras, show_camera_frames
from streaming.face_detection import reload_face_database

def main():
    print("Loading database from Milvus...")
    create_milvus_collection()
    reload_face_database()

    threads = start_all_cameras()
    show_camera_frames()

    for t in threads:
        t.join()

if __name__ == "__main__":
    main()