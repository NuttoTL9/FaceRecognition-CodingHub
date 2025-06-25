import cv2
import threading

class VideoStreamThread:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise Exception(f"❌ ไม่สามารถเชื่อมต่อกล้องได้: {src}")
        self.ret, self.frame = self.stream.read()
        self.lock = threading.Lock()
        self.stopped = False

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            if not ret:
                continue
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy()

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.stream.release()
