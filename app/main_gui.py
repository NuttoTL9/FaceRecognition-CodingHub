import sys
import cv2
import threading
import time
import requests
import torch
import base64
import io
from datetime import datetime, date

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QScrollArea, QPushButton, QLineEdit, QSizePolicy, QDialog,
    QFormLayout, QDateEdit, QComboBox, QMessageBox, QTextEdit
)
from PyQt5.QtCore import Qt, QTimer, QDate
from PyQt5.QtGui import QPixmap, QImage
from datetime import datetime

from config import LOG_EVENT_URL, FASTAPI_URL, MIN_LOG_INTERVAL
from recognition.face_models import mtcnn, resnet
from recognition.face_utils import preprocess_face, find_closest_match
from database.milvus_database import load_face_database
from streaming.face_detection import process_camera, camera_frames, reload_face_database, send_log_with_image, log_recognition_event

class AddFaceDialog(QDialog):
    def __init__(self, parent=None, current_frame=None):
        super().__init__(parent)
        self.current_frame = current_frame
        self.setWindowTitle("เพิ่มใบหน้าพนักงานใหม่")
        self.setModal(True)
        self.setFixedSize(450, 650)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # แสดงภาพตัวอย่าง
        if self.current_frame is not None:
            # ตรวจสอบจำนวนใบหน้าในภาพ
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(frame_rgb)
            face_count = len(boxes) if boxes is not None else 0
            
            preview_label = QLabel(f"ภาพตัวอย่าง (พบใบหน้า: {face_count} ใบหน้า):")
            layout.addWidget(preview_label)
            
            # แปลงภาพเป็น QPixmap และวาดกรอบรอบใบหน้า
            display_frame = self.current_frame.copy()
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, "Face", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img).scaled(200, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            image_label = QLabel()
            image_label.setPixmap(pixmap)
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setStyleSheet("border: 1px solid gray; padding: 5px;")
            layout.addWidget(image_label)
        
        # ฟอร์มสำหรับกรอกข้อมูล
        form_layout = QFormLayout()
        
        # ชื่อจริง
        self.firstname_edit = QLineEdit()
        form_layout.addRow("ชื่อจริง:", self.firstname_edit)
        
        # นามสกุล
        self.lastname_edit = QLineEdit()
        form_layout.addRow("นามสกุล:", self.lastname_edit)
        
        # เพศ
        self.gender_combo = QComboBox()
        self.gender_combo.addItems(["Male", "Female"])
        form_layout.addRow("เพศ:", self.gender_combo)
        
        # วันที่เริ่มงาน
        self.date_of_joining_edit = QDateEdit()
        self.date_of_joining_edit.setDate(QDate.currentDate())
        self.date_of_joining_edit.setCalendarPopup(True)
        form_layout.addRow("วันที่เริ่มงาน:", self.date_of_joining_edit)
        
        # วันเกิด
        self.date_of_birth_edit = QDateEdit()
        self.date_of_birth_edit.setDate(QDate.currentDate())
        self.date_of_birth_edit.setCalendarPopup(True)
        form_layout.addRow("วันเกิด:", self.date_of_birth_edit)
        
        # บริษัท
        company_layout = QHBoxLayout()
        self.company_combo = QComboBox()
        self.company_combo.setEditable(True)  # ให้พิมพ์เพิ่มได้
        self.company_combo.setInsertPolicy(QComboBox.InsertAtTop)  # เพิ่มใหม่ไว้ด้านบน
        
        self.refresh_company_btn = QPushButton("🔄")
        self.refresh_company_btn.setToolTip("รีเฟรชรายชื่อบริษัท")
        self.refresh_company_btn.setMaximumWidth(30)
        self.refresh_company_btn.clicked.connect(self.load_company_options)
        
        company_layout.addWidget(self.company_combo)
        company_layout.addWidget(self.refresh_company_btn)
        form_layout.addRow("บริษัท:", company_layout)
        
        layout.addLayout(form_layout)
        
        # ปุ่มบันทึกและยกเลิก
        button_layout = QHBoxLayout()
        self.save_button = QPushButton("บันทึก")
        self.save_button.clicked.connect(self.save_employee)
        self.cancel_button = QPushButton("ยกเลิก")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
        
        # ข้อความแนะนำ
        info_label = QLabel("💡 หมายเหตุ: ระบบจะใช้ภาพปัจจุบันจากกล้องเพื่อบันทึกใบหน้า\n📋 สามารถเลือกบริษัทที่มีอยู่หรือพิมพ์ชื่อใหม่ได้\n📸 รูปภาพใบหน้าจะถูกส่งไปยัง Employee profile ด้วย")
        info_label.setStyleSheet("color: #666; font-size: 10pt; padding: 5px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # ข้อความสถานะ
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setReadOnly(True)
        layout.addWidget(self.status_text)
        
        self.setLayout(layout)
        
        # โหลดรายชื่อบริษัท
        self.load_company_options()
    
    def log_status(self, message):
        self.status_text.append(f"{datetime.now().strftime('%H:%M:%S')} - {message}")
    
    def load_company_options(self):
        """โหลดรายชื่อบริษัทจาก FastAPI"""
        self.log_status("🔄 กำลังโหลดรายชื่อบริษัท...")
        try:
            response = requests.get(f"{FASTAPI_URL}/api/company-options/", timeout=5)
            if response.ok:
                data = response.json()
                companies = data.get("available_options", [])
                self.company_combo.clear()
                self.company_combo.addItems(companies)
                self.log_status(f"✅ โหลดรายชื่อบริษัทสำเร็จ ({len(companies)} บริษัท)")
                if companies:
                    self.log_status(f"📋 บริษัทที่มี: {', '.join(companies[:5])}{'...' if len(companies) > 5 else ''}")
            else:
                self.log_status("⚠️ ไม่สามารถโหลดรายชื่อบริษัทได้")
                # เพิ่มค่า default
                self.company_combo.addItems(["Default Company"])
        except Exception as e:
            self.log_status(f"❌ เกิดข้อผิดพลาดในการโหลดรายชื่อบริษัท: {str(e)}")
            # เพิ่มค่า default
            self.company_combo.addItems(["Default Company"])
    
    def save_employee(self):
        # ตรวจสอบข้อมูลที่กรอก
        firstname = self.firstname_edit.text().strip()
        lastname = self.lastname_edit.text().strip()
        gender = self.gender_combo.currentText()
        date_of_joining = self.date_of_joining_edit.date().toString("yyyy-MM-dd")
        date_of_birth = self.date_of_birth_edit.date().toString("yyyy-MM-dd")
        company = self.company_combo.currentText().strip()
        
        if not all([firstname, lastname, company]):
            QMessageBox.warning(self, "ข้อมูลไม่ครบ", "กรุณากรอกข้อมูลให้ครบถ้วน")
            return
        
        if self.current_frame is None:
            QMessageBox.warning(self, "ไม่พบภาพ", "ไม่พบภาพจากกล้อง")
            return
        
        self.log_status("เริ่มต้นการบันทึกข้อมูลพนักงาน...")
        
        # ตรวจสอบการเชื่อมต่อ FastAPI
        try:
            test_response = requests.get(f"{FASTAPI_URL}/docs", timeout=5)
            if not test_response.ok:
                self.log_status("❌ ไม่สามารถเชื่อมต่อ FastAPI ได้")
                QMessageBox.critical(self, "ข้อผิดพลาด", "ไม่สามารถเชื่อมต่อ FastAPI ได้")
                return
        except Exception as e:
            self.log_status(f"❌ ไม่สามารถเชื่อมต่อ FastAPI: {str(e)}")
            QMessageBox.critical(self, "ข้อผิดพลาด", f"ไม่สามารถเชื่อมต่อ FastAPI: {str(e)}")
            return
        
        try:
            # สร้าง Employee ใน ERPNext
            employee_payload = {
                "firstname": firstname,
                "lastname": lastname,
                "gender": gender,
                "date_of_joining": date_of_joining,
                "date_of_birth": date_of_birth,
                "company": company
            }
            
            self.log_status("ส่งคำขอสร้าง Employee...")
            emp_response = requests.post(f"{FASTAPI_URL}/api/resource/Employee", json=employee_payload, timeout=10)
            
            if not emp_response.ok:
                self.log_status(f"❌ สร้าง Employee ไม่สำเร็จ: {emp_response.status_code} {emp_response.text}")
                QMessageBox.warning(self, "ข้อผิดพลาด", f"สร้าง Employee ไม่สำเร็จ:\n{emp_response.status_code} {emp_response.text}")
                return
            
            employee_data = emp_response.json()
            employee_id = employee_data.get("employee_id") or employee_data.get("data", {}).get("name")
            
            if not employee_id:
                self.log_status("❌ ไม่ได้รับ employee_id จาก server")
                QMessageBox.critical(self, "ข้อผิดพลาด", "ไม่ได้รับ employee_id จาก server")
                return
            
            self.log_status(f"✅ ได้ employee_id: {employee_id}")
            
            # ประมวลผลภาพใบหน้า
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(frame_rgb)
            
            if boxes is None:
                self.log_status("❌ ไม่พบใบหน้าในภาพ")
                QMessageBox.warning(self, "ไม่พบใบหน้า", "ไม่พบใบหน้าในภาพจากกล้อง")
                return
            
            face_added = False
            for box in boxes:
                face_tensor = preprocess_face(self.current_frame, box)
                if face_tensor is None:
                    self.log_status("❌ ไม่สามารถสร้าง face tensor ได้")
                    continue
                
                with torch.no_grad():
                    embedding = resnet(face_tensor)
                
                embedding_np = embedding.squeeze(0).cpu().numpy().astype("float32")
                
                vector_payload = {
                    "employee_id": employee_id,
                    "name": f"{firstname} {lastname}",
                    "embedding": embedding_np.tolist()
                }
                
                self.log_status("ส่ง embedding ไปยัง FastAPI...")
                vector_response = requests.post(f"{FASTAPI_URL}/add_face_vector/", json=vector_payload, timeout=10)
                
                if not vector_response.ok:
                    self.log_status(f"❌ บันทึก embedding ไม่สำเร็จ: {vector_response.status_code} {vector_response.text}")
                    continue
                
                face_added = True
                self.log_status("✅ บันทึกข้อมูล embedding และพนักงานเรียบร้อย")
                self.log_status(f"✅ Response: {vector_response.json()}")
                
                break
            
            if face_added:
                # รีโหลดฐานข้อมูลใบหน้า
                reload_face_database()
                QMessageBox.information(self, "สำเร็จ", f"เพิ่มพนักงาน {firstname} {lastname} เรียบร้อยแล้ว")
                self.accept()
            else:
                self.log_status("❌ ไม่สามารถบันทึกใบหน้าได้")
                QMessageBox.warning(self, "ไม่สำเร็จ", "ไม่สามารถบันทึกใบหน้าได้")
                
        except Exception as e:
            self.log_status(f"❌ เกิดข้อผิดพลาดขณะบันทึก: {str(e)}")
            QMessageBox.critical(self, "ข้อผิดพลาด", f"เกิดข้อผิดพลาดขณะบันทึก: {str(e)}")

class FaceRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition with Real-time Log")
        self.setGeometry(100, 100, 1200, 600)

        self.video_source_name = "MainCam"
        self.last_logged_names = []
        self.last_log_times = {}
        self.person_states = {}
        self.last_log_times = {}

        self.known_embeddings, self.known_names, self.known_employee_ids = load_face_database()
        reload_face_database()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.init_ui()
        self.start_video()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        # ---------- Log Panel ----------
        self.log_panel = QVBoxLayout()
        self.log_container = QWidget()
        self.log_container.setLayout(self.log_panel)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.log_container)
        self.scroll.setMinimumWidth(300)

        # ---------- Camera Panel ----------
        self.input_rtsp = QLineEdit(self)
        self.input_rtsp.setPlaceholderText("Enter RTSP URL or leave empty for webcam")

        self.connect_btn = QPushButton("Connect Camera")
        self.connect_btn.clicked.connect(self.change_camera)

        # ปุ่ม Add Face
        self.add_face_btn = QPushButton("Add Face")
        self.add_face_btn.clicked.connect(self.show_add_face_dialog)
        self.add_face_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)

        # กล้อง
        self.camera_label = QLabel(self)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_label.setStyleSheet("background-color: #ddd;")  # สีพื้นหลังเวลาไม่มีภาพ
        self.camera_label.resizeEvent = lambda event: self.update_frame()

        # Layout กล้องฝั่งขวา
        camera_control_layout = QVBoxLayout()
        camera_control_layout.setContentsMargins(0, 0, 0, 0)
        camera_control_layout.setSpacing(5)
        camera_control_layout.addWidget(self.input_rtsp)
        camera_control_layout.addWidget(self.connect_btn)
        camera_control_layout.addWidget(self.add_face_btn)
        camera_control_layout.addWidget(self.camera_label)

        camera_widget = QWidget()
        camera_widget.setLayout(camera_control_layout)

        # ---------- Add to Main Layout ----------
        main_layout.addWidget(self.scroll)
        main_layout.addWidget(camera_widget)

        main_layout.setStretch(0, 1)  # log
        main_layout.setStretch(1, 4)  # camera

        self.setLayout(main_layout)

    def start_video(self):
        self.video_source = 0
        threading.Thread(target=process_camera, args=(self.video_source, self.video_source_name), daemon=True).start()
        self.timer.start(30)

    def change_camera(self):
        rtsp = self.input_rtsp.text().strip()
        self.video_source = rtsp if rtsp else 0
        threading.Thread(target=process_camera, args=(self.video_source, self.video_source_name), daemon=True).start()
        self.timer.start(30)

    def show_add_face_dialog(self):
        # รับภาพปัจจุบันจากกล้อง
        current_frame = camera_frames.get(self.video_source_name)
        if current_frame is None:
            QMessageBox.warning(self, "ไม่พบภาพ", "ไม่พบภาพจากกล้อง กรุณาตรวจสอบการเชื่อมต่อกล้อง")
            return
        
        # ตรวจสอบว่ามีใบหน้าในภาพหรือไม่
        frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(frame_rgb)
        if boxes is None or len(boxes) == 0:
            QMessageBox.warning(self, "ไม่พบใบหน้า", "ไม่พบใบหน้าในภาพจากกล้อง กรุณาตรวจสอบให้แน่ใจว่ามีใบหน้าอยู่ในกล้อง")
            return
        
        # ปิดการใช้งานปุ่มชั่วคราว
        self.add_face_btn.setEnabled(False)
        self.add_face_btn.setText("กำลังประมวลผล...")
        
        try:
            # แสดงป๊อปอัพสำหรับเพิ่มใบหน้า
            dialog = AddFaceDialog(self, current_frame)
            if dialog.exec_() == QDialog.Accepted:
                # รีโหลดฐานข้อมูลใบหน้าหลังจากเพิ่มสำเร็จ
                self.known_embeddings, self.known_names, self.known_employee_ids = load_face_database()
                print("✅ รีโหลดฐานข้อมูลใบหน้าเรียบร้อย")
        finally:
            # เปิดการใช้งานปุ่มอีกครั้ง
            self.add_face_btn.setEnabled(True)
            self.add_face_btn.setText("Add Face")

    def update_frame(self):
        frame = camera_frames.get(self.video_source_name)
        if frame is None:
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        q_img = QImage(rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0], QImage.Format_RGB888)

        scaled_pixmap = QPixmap.fromImage(q_img).scaled(
            self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.camera_label.setPixmap(scaled_pixmap)

        self.detect_and_log_faces(frame)


    def detect_and_log_faces(self, frame):
        boxes, _ = mtcnn.detect(frame)
        if boxes is None:
            return

        logging_now_set = set()

        for box in boxes:
            preprocessed = preprocess_face(frame, box)
            if preprocessed is None:
                continue

            embedding = resnet(preprocessed).detach().cpu()

            match_employee_id, match_name, match_dist = find_closest_match(
                embedding,
                self.known_embeddings,
                self.known_employee_ids,
                self.known_names,
            )

            if match_name and match_name.lower() != "unknown" and match_dist < 0.6:
                event_type = "in" if datetime.now().hour < 12 else "out"
                now = time.time()
                last_time = self.last_log_times.get(match_employee_id, 0)
                last_state = self.person_states.get(match_employee_id, None)

                if (now - last_time < MIN_LOG_INTERVAL) and (last_state == event_type):
                    continue

                if match_employee_id in logging_now_set:
                    continue  # ป้องกัน log ซ้ำใน frame เดียวกัน
                logging_now_set.add(match_employee_id)

                try:
                    # ส่ง log
                    payload = {"name": match_employee_id, "event": event_type}
                    res = requests.post(LOG_EVENT_URL, json=payload, timeout=3)
                    if res.ok:
                        print(f"✅ Log {match_name} [{event_type}]")

                        self.person_states[match_employee_id] = event_type
                        self.last_log_times[match_employee_id] = now

                        self.log_face(match_employee_id, match_name, frame)

                    else:
                        print("❌ Log ล้มเหลว:", res.status_code, res.text)

                except Exception as e:
                    print(f"❌ Error logging for {match_employee_id}: {e}")



    def log_face(self, employee_id, name, frame):
        rgb_face = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_face.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_face.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        now = datetime.now()
        date_str = now.strftime("%d-%m-%Y")
        time_str = now.strftime("%H:%M:%S")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(5, 5, 5, 5)

        pic_label = QLabel()
        pic_label.setPixmap(pixmap)
        pic_label.setAlignment(Qt.AlignCenter)
        pic_label.setStyleSheet("border: 1px solid gray;")

        employee_id_label = QLabel(employee_id)
        employee_id_label.setAlignment(Qt.AlignCenter)
        employee_id_label.setStyleSheet("color: #007ACC; font-weight: bold; font-size: 11pt;")

        name_label = QLabel(name)
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setStyleSheet("font-size: 10pt;")

        date_label = QLabel(date_str)
        date_label.setAlignment(Qt.AlignCenter)
        date_label.setStyleSheet("color: gray; font-size: 10pt;")

        time_label = QLabel(time_str)
        time_label.setAlignment(Qt.AlignCenter)
        time_label.setStyleSheet("color: gray; font-size: 10pt;")

        layout.addWidget(pic_label)
        layout.addWidget(employee_id_label)
        layout.addWidget(name_label)
        layout.addWidget(date_label)
        layout.addWidget(time_label)

        container.setStyleSheet("""
            background-color: white;
            padding: 5px;
        """)

        self.log_panel.insertWidget(0, container)


    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())
