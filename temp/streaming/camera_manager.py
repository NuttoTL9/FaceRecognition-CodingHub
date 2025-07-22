import threading
import cv2
from config import RTSP_URLS
from streaming.face_detection import process_camera, camera_frames, camera_streams, should_exit

def start_all_cameras():
    threads = []
    for idx, rtsp in enumerate(RTSP_URLS):
        cam_name = f"Camera-{idx+1}"
        t = threading.Thread(target=process_camera, args=(rtsp, cam_name))
        t.start()
        threads.append(t)
    return threads

def show_camera_frames():
    while True:
        for name, frame in camera_frames.copy().items():
            cv2.imshow(name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            from streaming.registration import capture_and_save
            threading.Thread(target=capture_and_save).start()
        elif key == ord('q'):
            should_exit[0] = True
            break

    cv2.destroyAllWindows()
