import threading
import cv2
import math
import numpy as np
from config import RTSP_URLS
from streaming.face_detection import process_camera, camera_frames, camera_streams, should_exit

def start_all_cameras():
    should_exit[0] = False
    threads = []
    for idx, rtsp in enumerate(RTSP_URLS):
        cam_name = f"Camera-{idx+1}"
        t = threading.Thread(target=process_camera, args=(rtsp, cam_name), daemon=True)
        t.start()
        threads.append(t)
    return threads


def show_camera_frames():
    window_name = "Cameras"

    def compute_grid(n):
        if n <= 0:
            return 0, 0
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        return rows, cols

    def make_mosaic(frames_list):
        count = len(frames_list)
        if count == 0:
            return None
        rows, cols = compute_grid(count)
        # Determine tile size from the first valid frame
        base_h, base_w = None, None
        for f in frames_list:
            if f is not None:
                base_h, base_w = f.shape[0], f.shape[1]
                break
        if base_h is None:
            return None
        # Scale tiles to a reasonable size
        tile_w = max(200, min(640, base_w))
        tile_h = max(150, min(480, base_h))

        # Prepare black canvas
        mosaic = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

        for i in range(rows * cols):
            r = i // cols
            c = i % cols
            y1, y2 = r * tile_h, (r + 1) * tile_h
            x1, x2 = c * tile_w, (c + 1) * tile_w
            if i < count and frames_list[i] is not None:
                frame = cv2.resize(frames_list[i], (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
                mosaic[y1:y2, x1:x2] = frame
            else:
                # Leave black if no frame available
                pass
        return mosaic

    while True:
        frames_copy = camera_frames.copy()
        # Keep deterministic order by name
        items = sorted(frames_copy.items(), key=lambda kv: kv[0])
        frames_list = [frame for _name, frame in items if frame is not None]
        grid = make_mosaic(frames_list)
        if grid is not None:
            cv2.imshow(window_name, grid)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            from streaming.registration import capture_and_save
            threading.Thread(target=capture_and_save, daemon=True).start()
        elif key == ord('q'):
            should_exit[0] = True
            break

    cv2.destroyAllWindows()

