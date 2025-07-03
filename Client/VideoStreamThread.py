import subprocess as sp
import threading
import numpy as np
import cv2
import shutil

class VideoStreamThread:
    def __init__(self, rtsp_url, width=320, height=240):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.stopped = False
        self.lock = threading.Lock()
        self.ret = False
        self.frame = None
        self.pipe = None

        # ตรวจสอบว่า FFmpeg ติดตั้งไว้หรือไม่
        if not shutil.which("ffmpeg"):
            raise EnvironmentError("FFmpeg ไม่พบใน PATH. โปรดติดตั้งด้วย: sudo apt install ffmpeg")

        # FFmpeg command สำหรับอ่าน RTSP stream แบบ raw video
        self.command = [
            'ffmpeg',
            '-rtsp_transport', 'tcp',
            '-i', self.rtsp_url,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}',
            '-an', '-sn',
            '-loglevel', 'quiet',
            'pipe:1'
        ]

        try:
            self.pipe = sp.Popen(self.command, stdout=sp.PIPE, bufsize=10**8)
        except Exception as e:
            raise Exception(f"❌ ไม่สามารถเปิด FFmpeg subprocess ได้: {e}")

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            # ตรวจสอบว่า FFmpeg ยังทำงานอยู่หรือไม่
            if self.pipe.poll() is not None:
                print("⚠️ FFmpeg process ended unexpectedly.")
                self.stop()
                break

            raw_frame = self.pipe.stdout.read(self.width * self.height * 3)
            if len(raw_frame) != self.width * self.height * 3:
                continue  # ข้าม frame ที่ไม่สมบูรณ์

            frame = np.frombuffer(raw_frame, np.uint8).reshape((self.height, self.width, 3))
            with self.lock:
                self.ret = True
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            else:
                return False, None

    def stop(self):
        if not self.stopped:
            self.stopped = True
            if self.pipe:
                try:
                    self.pipe.terminate()
                    self.pipe.wait(timeout=2)
                except Exception as e:
                    print(f"⚠️ ปิด FFmpeg ไม่สมบูรณ์: {e}")
            # เช็คก่อนว่า thread ที่จะ join ไม่ใช่ thread ปัจจุบัน
            if self.thread.is_alive() and threading.current_thread() != self.thread:
                self.thread.join()


    def __del__(self):
        self.stop()