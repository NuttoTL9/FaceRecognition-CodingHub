import sys
import cv2
import threading
import requests
import torch
import numpy as np
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QScrollArea, QPushButton, QLineEdit, QSizePolicy, QDialog,
    QFormLayout, QDateEdit, QComboBox, QMessageBox, QTextEdit,
    QGridLayout, QLayout
)

from PyQt5.QtCore import Qt, QTimer, QDate
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QFrame, QGroupBox
from datetime import datetime

from config import FASTAPI_URL, RTSP_URLS
from recognition.face_models import mtcnn, resnet
from recognition.face_utils import preprocess_face
from database.milvus_database import load_face_database, add_embedding_to_milvus, search_face
from streaming.face_detection import (
    process_camera, camera_frames, reload_face_database, get_ui_events
)


class AddFaceDialog(QDialog):
    def __init__(self, parent=None, current_frame=None):
        super().__init__(parent)
        self.current_frame = current_frame
        self.setWindowTitle("เพิ่มใบหน้าพนักงานใหม่")
        self.setModal(True)
        self.setMinimumSize(480, 560)
        try:
            self.setSizeGripEnabled(True)
        except Exception:
            pass
        self.setStyleSheet("""
            QDialog {
                background-color: #f8f9fa;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel {
                color: #2c3e50;
                font-weight: 500;
            }
            QLineEdit, QComboBox, QDateEdit {
                padding: 8px 12px;
                border: 2px solid #e9ecef;
                border-radius: 6px;
                background-color: white;
                font-size: 12px;
                min-height: 20px;
            }
            QLineEdit:focus, QComboBox:focus, QDateEdit:focus {
                border-color: #007bff;
                outline: none;
            }
            QPushButton {
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
                min-height: 20px;
            }
            QTextEdit {
                border: 2px solid #e9ecef;
                border-radius: 6px;
                background-color: white;
                padding: 8px;
            }
        """)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSizeConstraint(QLayout.SetMinimumSize)
        layout.setSizeConstraint(QLayout.SetMinimumSize)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Header
        header_label = QLabel("📷 เพิ่มพนักงานใหม่")
        header_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            padding: 6px 0;
            border-bottom: 2px solid #e9ecef;
        """)
        layout.addWidget(header_label)
        
        # Camera selection section
        camera_group = QGroupBox("เลือกกล้อง")
        camera_group.setStyleSheet("QGroupBox { font-weight: bold; color: #495057; }")
        camera_section = QVBoxLayout(camera_group)
        camera_section.setSpacing(8)
        camera_section.setContentsMargins(8, 8, 8, 8)
        
        camera_header = QLabel("🎥 เลือกกล้อง")
        camera_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #495057;")
        camera_section.addWidget(camera_header)
        
        camera_form = QFormLayout()
        camera_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        camera_form.setSpacing(10)
        self.camera_combo = QComboBox()
        if len(RTSP_URLS) > 1:
            camera_names = [f"Camera-{i+1}" for i in range(len(RTSP_URLS))]
        else:
            camera_names = ["MainCam"]
        self.camera_combo.addItems(camera_names)
        self.camera_combo.currentTextChanged.connect(self.on_camera_changed)
        camera_form.addRow("กล้อง:", self.camera_combo)
        
        # Camera preview
        preview_layout = QHBoxLayout()
        preview_layout.setSpacing(0)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        
        self.camera_preview = QLabel()
        self.camera_preview.setMinimumSize(320, 180)
        self.camera_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_preview.setStyleSheet("""
            border: 3px solid #dee2e6;
            border-radius: 8px;
            background-color: #f8f9fa;
            padding: 5px;
        """)
        self.camera_preview.setAlignment(Qt.AlignCenter)
        self.camera_preview.setText("รอภาพจากกล้อง...")
        
        preview_layout.addWidget(self.camera_preview)
        
        camera_section.addLayout(camera_form)
        camera_section.addLayout(preview_layout)
        layout.addWidget(camera_group)
        layout.addSpacing(10)
        
        # spacing before details section
        layout.addSpacing(8)
        
        # เพิ่ม timer สำหรับอัพเดตภาพตัวอย่าง
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self.update_preview)
        self.preview_timer.start(120)  # อัพเดตทุก 120ms
        
        # Employee information section
        info_section = QVBoxLayout()
        info_section.setSpacing(8)
        
        info_header = QLabel("👤 ข้อมูลพนักงาน")
        info_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #495057;")
        info_section.addWidget(info_header)
        
        form_layout = QFormLayout()
        form_layout.setSpacing(12)
        form_layout.setLabelAlignment(Qt.AlignRight)
        
        # Personal info
        self.firstname_edit = QLineEdit()
        self.firstname_edit.setPlaceholderText("กรอกชื่อจริง")
        form_layout.addRow("ชื่อจริง:", self.firstname_edit)
        
        self.lastname_edit = QLineEdit()
        self.lastname_edit.setPlaceholderText("กรอกนามสกุล")
        form_layout.addRow("นามสกุล:", self.lastname_edit)
        
        self.gender_combo = QComboBox()
        self.gender_combo.addItems(["Male", "Female"])
        form_layout.addRow("เพศ:", self.gender_combo)
        
        # Date info
        self.date_of_joining_edit = QDateEdit()
        self.date_of_joining_edit.setDate(QDate.currentDate())
        self.date_of_joining_edit.setCalendarPopup(True)
        form_layout.addRow("วันที่เริ่มงาน:", self.date_of_joining_edit)
        
        self.date_of_birth_edit = QDateEdit()
        self.date_of_birth_edit.setDate(QDate.currentDate())
        self.date_of_birth_edit.setCalendarPopup(True)
        form_layout.addRow("วันเกิด:", self.date_of_birth_edit)
        
        # Company info
        company_layout = QHBoxLayout()
        company_layout.setSpacing(8)
        self.company_combo = QComboBox()
        self.company_combo.setEditable(True)
        self.company_combo.setInsertPolicy(QComboBox.InsertAtTop)
        self.company_combo.setPlaceholderText("เลือกหรือพิมพ์ชื่อบริษัท")
        
        self.refresh_company_btn = QPushButton("🔄")
        self.refresh_company_btn.setToolTip("รีเฟรชรายชื่อบริษัท")
        self.refresh_company_btn.setFixedSize(35, 35)
        self.refresh_company_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        self.refresh_company_btn.clicked.connect(self.load_company_options)
        
        company_layout.addWidget(self.company_combo)
        company_layout.addWidget(self.refresh_company_btn)
        form_layout.addRow("บริษัท:", company_layout)
        
        info_section.addLayout(form_layout)
        layout.addLayout(info_section)
        
        # Buttons section
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        
        self.save_button = QPushButton("💾 บันทึก")
        self.save_button.clicked.connect(self.save_employee)
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 12px 25px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1e7e34;
            }
        """)
        
        self.cancel_button = QPushButton("❌ ยกเลิก")
        self.cancel_button.clicked.connect(self.reject)
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 12px 25px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
            QPushButton:pressed {
                background-color: #bd2130;
            }
        """)
        
        button_layout.addStretch()
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Info section
        info_label = QLabel("💡 หมายเหตุ: ระบบจะใช้ภาพปัจจุบันจากกล้องเพื่อบันทึกใบหน้า\n📋 สามารถเลือกบริษัทที่มีอยู่หรือพิมพ์ชื่อใหม่ได้")
        info_label.setStyleSheet("""
            color: #6c757d;
            font-size: 11px;
            padding: 10px;
            background-color: #e9ecef;
            border-radius: 6px;
            border-left: 4px solid #007bff;
        """)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Status section
        status_header = QLabel("📊 สถานะการทำงาน")
        status_header.setStyleSheet("font-size: 12px; font-weight: bold; color: #495057; margin-top: 10px;")
        layout.addWidget(status_header)
        
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(60)
        self.status_text.setReadOnly(True)
        self.status_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 10px;
            }
        """)
        layout.addWidget(self.status_text)
        
        self.setLayout(layout)
        try:
            self.adjustSize()
        except Exception:
            pass
        
        self.load_company_options()
    
    def on_camera_changed(self):
        self.camera_preview.setText("กำลังโหลดภาพ...")
        
    def update_preview(self):
        try:
            selected_camera = self.camera_combo.currentText()
            current_frame = camera_frames.get(selected_camera)
            if current_frame is not None:
                rgb_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img).scaled(
                    self.camera_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.camera_preview.setPixmap(pixmap)
            else:
                self.camera_preview.setText(f"ไม่พบภาพจาก {selected_camera}")
        except Exception:
            self.camera_preview.setText("เกิดข้อผิดพลาด")
    
    def log_status(self, message):
        self.status_text.append(f"{datetime.now().strftime('%H:%M:%S')} - {message}")
    
    def load_company_options(self):
        self.log_status("กำลังโหลดรายชื่อบริษัท...")
        try:
            response = requests.get(f"{FASTAPI_URL}/api/company-options/", timeout=5)
            if response.ok:
                data = response.json()
                companies = data.get("available_options", [])
                self.company_combo.clear()
                self.company_combo.addItems(companies)
                self.log_status(f"โหลดรายชื่อบริษัทสำเร็จ ({len(companies)} บริษัท)")
                if companies:
                    self.log_status(f"บริษัทที่มี: {', '.join(companies[:5])}{'...' if len(companies) > 5 else ''}")
            else:
                self.log_status("ไม่สามารถโหลดรายชื่อบริษัทได้")
                self.company_combo.addItems(["Default Company"])
        except Exception as e:
            self.log_status(f" เกิดข้อผิดพลาดในการโหลดรายชื่อบริษัท: {str(e)}")
            self.company_combo.addItems(["Default Company"])
    
    def save_employee(self):
        firstname = self.firstname_edit.text().strip()
        lastname = self.lastname_edit.text().strip()
        gender = self.gender_combo.currentText()
        date_of_joining = self.date_of_joining_edit.date().toString("yyyy-MM-dd")
        date_of_birth = self.date_of_birth_edit.date().toString("yyyy-MM-dd")
        company = self.company_combo.currentText().strip()
        
        if not all([firstname, lastname, company]):
            QMessageBox.warning(self, "ข้อมูลไม่ครบ", "กรุณากรอกข้อมูลให้ครบถ้วน")
            return
        
        # ใช้ภาพจากกล้องที่เลือก
        selected_camera = self.camera_combo.currentText()
        current_frame = camera_frames.get(selected_camera)
        if current_frame is None:
            QMessageBox.warning(self, "ไม่พบภาพ", f"ไม่พบภาพจากกล้อง {selected_camera}")
            return
        
        self.log_status("เริ่มต้นการบันทึกข้อมูลพนักงาน...")
        
        try:
            test_response = requests.get(f"{FASTAPI_URL}/docs", timeout=5)
            if not test_response.ok:
                self.log_status(" ไม่สามารถเชื่อมต่อ FastAPI ได้")
                QMessageBox.critical(self, "ข้อผิดพลาด", "ไม่สามารถเชื่อมต่อ FastAPI ได้")
                return
        except Exception as e:
            self.log_status(f" ไม่สามารถเชื่อมต่อ FastAPI: {str(e)}")
            QMessageBox.critical(self, "ข้อผิดพลาด", f"ไม่สามารถเชื่อมต่อ FastAPI: {str(e)}")
            return
        
        try:
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
                self.log_status(f" สร้าง Employee ไม่สำเร็จ: {emp_response.status_code} {emp_response.text}")
                QMessageBox.warning(self, "ข้อผิดพลาด", f"สร้าง Employee ไม่สำเร็จ:\n{emp_response.status_code} {emp_response.text}")
                return
            
            employee_data = emp_response.json()
            employee_id = employee_data.get("employee_id") or employee_data.get("data", {}).get("name")
            
            if not employee_id:
                self.log_status(" ไม่ได้รับ employee_id จาก server")
                QMessageBox.critical(self, "ข้อผิดพลาด", "ไม่ได้รับ employee_id จาก server")
                return
            
            self.log_status(f"ได้ employee_id: {employee_id}")
            
            frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(frame_rgb)
            
            if boxes is None:
                self.log_status(" ไม่พบใบหน้าในภาพ")
                QMessageBox.warning(self, "ไม่พบใบหน้า", f"ไม่พบใบหน้าในภาพจากกล้อง {selected_camera}")
                return
            
            face_added = False
            for box in boxes:
                face_tensor = preprocess_face(current_frame, box)
                if face_tensor is None:
                    self.log_status(" ไม่สามารถสร้าง face tensor ได้")
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
                    self.log_status(f"บันทึก embedding ไม่สำเร็จ: {vector_response.status_code} {vector_response.text}")
                    continue
                
                face_added = True
                self.log_status("บันทึกข้อมูล embedding และพนักงานเรียบร้อย")
                self.log_status(f"Response: {vector_response.json()}")
                
                break
            
            if face_added:
                reload_face_database()
                QMessageBox.information(self, "สำเร็จ", f"เพิ่มพนักงาน {firstname} {lastname} เรียบร้อยแล้ว")
                self.accept()
            else:
                self.log_status("ไม่สามารถบันทึกใบหน้าได้")
                QMessageBox.warning(self, "ไม่สำเร็จ", "ไม่สามารถบันทึกใบหน้าได้")
                
        except Exception as e:
            self.log_status(f"เกิดข้อผิดพลาดขณะบันทึก: {str(e)}")
            QMessageBox.critical(self, "ข้อผิดพลาด", f"เกิดข้อผิดพลาดขณะบันทึก: {str(e)}")

    def closeEvent(self, event):
        self.preview_timer.stop()
        super().closeEvent(event)


class AddImageToExistingDialog(QDialog):
    def __init__(self, parent=None, current_frame=None):
        super().__init__(parent)
        self.current_frame = current_frame
        self.setWindowTitle("เพิ่มรูปให้พนักงานที่มีอยู่แล้ว")
        self.setModal(True)
        self.setMinimumSize(520, 620)
        try:
            self.setSizeGripEnabled(True)
        except Exception:
            pass
        self.recording = False
        self.recorded_embeddings = []
        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_seconds = 5
        

        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self.update_preview)
        
        self.setStyleSheet("""
            QDialog {
                background-color: #f8f9fa;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel {
                color: #2c3e50;
                font-weight: 500;
            }
            QLineEdit, QComboBox {
                padding: 8px 12px;
                border: 2px solid #e9ecef;
                border-radius: 6px;
                background-color: white;
                font-size: 12px;
                min-height: 20px;
            }
            QLineEdit:focus, QComboBox:focus {
                border-color: #007bff;
                outline: none;
            }
            QPushButton {
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
                min-height: 20px;
            }
            QTextEdit {
                border: 2px solid #e9ecef;
                border-radius: 6px;
                background-color: white;
                padding: 8px;
            }
        """)
        
        self.init_ui()
        self.load_employee_list()
        
        self.preview_timer.start(100)

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Header
        header_label = QLabel("📸 เพิ่มรูปให้พนักงานที่มีอยู่แล้ว")
        header_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            padding: 6px 0;
            border-bottom: 2px solid #e9ecef;
        """)
        layout.addWidget(header_label)
        
        selection_section = QVBoxLayout()
        selection_section.setSpacing(8)
        
        selection_header = QLabel("👥 เลือกพนักงานและกล้อง")
        selection_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #495057;")
        selection_section.addWidget(selection_header)
        
        form_layout = QFormLayout()
        form_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form_layout.setSpacing(10)
        form_layout.setLabelAlignment(Qt.AlignRight)
        
        self.employee_combo = QComboBox()
        self.employee_combo.setPlaceholderText("เลือกพนักงาน")
        form_layout.addRow("พนักงาน:", self.employee_combo)
        
        self.camera_combo = QComboBox()
        if len(RTSP_URLS) > 1:
            camera_names = [f"Camera-{i+1}" for i in range(len(RTSP_URLS))]
        else:
            camera_names = ["MainCam"]
        self.camera_combo.addItems(camera_names)
        self.camera_combo.currentTextChanged.connect(self.on_camera_changed)
        form_layout.addRow("กล้อง:", self.camera_combo)
        
        selection_section.addLayout(form_layout)
        layout.addLayout(selection_section)
        
        separator = QLabel()
        separator.setStyleSheet("background-color: #dee2e6; min-height: 1px; max-height: 1px;")
        layout.addWidget(separator)

        preview_group = QGroupBox("ภาพตัวอย่างจากกล้อง")
        preview_group.setStyleSheet("QGroupBox { font-weight: bold; color: #495057; }")
        preview_section = QVBoxLayout(preview_group)
        preview_section.setSpacing(8)
        preview_section.setContentsMargins(12, 12, 12, 12)
        
        preview_layout = QHBoxLayout()
        preview_layout.setSpacing(15)
        
        self.camera_preview = QLabel()
        self.camera_preview.setMinimumSize(420, 236)
        self.camera_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_preview.setStyleSheet("""
            border: 3px solid #dee2e6;
            border-radius: 8px;
            background-color: #f8f9fa;
            padding: 5px;
        """)
        self.camera_preview.setAlignment(Qt.AlignCenter)
        self.camera_preview.setText("รอภาพจากกล้อง...")
        
        preview_layout.addWidget(self.camera_preview)
        
        preview_section.addLayout(preview_layout)
        layout.addWidget(preview_group)
        layout.addSpacing(10)

        # Recording section
        recording_group = QGroupBox("การบันทึก")
        recording_group.setStyleSheet("QGroupBox { font-weight: bold; color: #495057; }")
        recording_section = QVBoxLayout(recording_group)
        recording_section.setSpacing(10)
        recording_section.setContentsMargins(12, 12, 12, 12)
        
        recording_header = QLabel("🎬 การบันทึก")
        recording_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #495057;")
        recording_section.addWidget(recording_header)
        
        recording_layout = QHBoxLayout()
        recording_layout.setSpacing(15)
        recording_layout.setContentsMargins(0, 0, 0, 0)
        
        self.record_button = QPushButton("🎥 เริ่มบันทึก (5 วินาที)")
        self.record_button.clicked.connect(self.start_recording)
        self.record_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        
        self.countdown_label = QLabel("⏳ พร้อมบันทึก")
        self.countdown_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #495057;
                padding: 12px 20px;
                border: 2px solid #dee2e6;
                border-radius: 6px;
                background-color: #f8f9fa;
                min-width: 150px;
            }
        """)
        self.countdown_label.setAlignment(Qt.AlignCenter)
        
        recording_layout.addStretch()
        recording_layout.addWidget(self.record_button)
        recording_layout.addWidget(self.countdown_label)
        recording_layout.addStretch()
        recording_section.addLayout(recording_layout)

        # Status labels
        status_layout = QVBoxLayout()
        status_layout.setSpacing(8)
        
        self.progress_label = QLabel("📊 สถานะ: รอเริ่มบันทึก")
        self.progress_label.setWordWrap(True)
        self.progress_label.setStyleSheet("""
            font-size: 12px; 
            color: #6c757d; 
            padding: 8px 12px;
            background-color: #e9ecef;
            border-radius: 4px;
        """)
        status_layout.addWidget(self.progress_label)

        self.embeddings_count_label = QLabel("📈 จำนวน embedding ที่บันทึก: 0")
        self.embeddings_count_label.setWordWrap(True)
        self.embeddings_count_label.setStyleSheet("""
            font-size: 12px; 
            color: #007bff; 
            font-weight: bold;
            padding: 8px 12px;
            background-color: #d1ecf1;
            border-radius: 4px;
            border-left: 4px solid #007bff;
        """)
        status_layout.addWidget(self.embeddings_count_label)
        
        recording_section.addLayout(status_layout)
        layout.addWidget(recording_group)

        button_layout = QHBoxLayout()
        self.save_button = QPushButton("บันทึกทั้งหมด")
        self.save_button.clicked.connect(self.save_all_embeddings)
        self.save_button.setEnabled(False)
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        
        self.cancel_button = QPushButton("ยกเลิก")
        self.cancel_button.clicked.connect(self.reject)
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(120)
        self.status_text.setReadOnly(True)
        layout.addWidget(self.status_text)

        self.setLayout(layout)

    def log_status(self, message):
        self.status_text.append(f"{datetime.now().strftime('%H:%M:%S')} - {message}")

    def load_employee_list(self):
        try:
            self.log_status("กำลังโหลดรายชื่อพนักงาน...")
            response = requests.get(f"{FASTAPI_URL}/list_employees/", timeout=30)
            if response.ok:
                data = response.json()
                employees = data.get("employees", [])
                self.employee_combo.clear()
                for emp in employees:
                    display_text = f"{emp['employee_id']} - {emp['name']} ({emp.get('embedding_count', 0)} embeddings)"
                    self.employee_combo.addItem(display_text, emp["employee_id"])
                self.log_status(f"โหลดสำเร็จ ({len(employees)} คน)")
            else:
                self.log_status(f"โหลดรายชื่อไม่สำเร็จ: {response.status_code}")
        except Exception as e:
            self.log_status(f"เกิดข้อผิดพลาด: {str(e)}")

    def start_recording(self):
        if self.recording:
            return
            
        employee_id = self.employee_combo.currentData()
        if not employee_id:
            QMessageBox.warning(self, "ข้อมูลไม่ครบ", "กรุณาเลือกพนักงาน")
            return

        self.recording = True
        self.recorded_embeddings = []
        self.countdown_seconds = 5
        self.record_button.setEnabled(False)
        self.record_button.setText("กำลังบันทึก...")
        self.countdown_label.setText(f"เริ่มใน {self.countdown_seconds} วินาที")
        self.countdown_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: white;
                padding: 10px;
                border: 2px solid #4CAF50;
                border-radius: 5px;
                background-color: #4CAF50;
            }
        """)
        self.progress_label.setText("สถานะ: เริ่มบันทึกใน 5 วินาที")
        self.log_status("เริ่มการบันทึก embedding...")
        
        self.countdown_timer.start(1000)

    def update_countdown(self):
        self.countdown_seconds -= 1
        
        if self.countdown_seconds > 0:
            self.countdown_label.setText(f"เริ่มใน {self.countdown_seconds} วินาที")
            self.progress_label.setText(f"สถานะ: เริ่มบันทึกใน {self.countdown_seconds} วินาที")
        else:
            self.countdown_timer.stop()
            self.countdown_label.setText("กำลังบันทึก...")
            self.countdown_label.setStyleSheet("""
                QLabel {
                    font-size: 16px;
                    font-weight: bold;
                    color: white;
                    padding: 10px;
                    border: 2px solid #f44336;
                    border-radius: 5px;
                    background-color: #f44336;
                }
            """)
            self.progress_label.setText("สถานะ: กำลังบันทึก embedding...")
            self.log_status("เริ่มบันทึก embedding แล้ว! กรุณาหมุนใบหน้าให้ครบทุกมุม")
            
            self.recording_timer = QTimer()
            self.recording_timer.timeout.connect(self.capture_embedding)
            self.recording_timer.start(500)
            
            self.stop_timer = QTimer()
            self.stop_timer.timeout.connect(self.stop_recording)
            self.stop_timer.start(5000)

    def capture_embedding(self):
        if not self.recording:
            return
            
        try:
            selected_camera = self.camera_combo.currentText()
            current_frame = camera_frames.get(selected_camera)
            if current_frame is None:
                self.log_status(f"ไม่พบภาพจากกล้อง {selected_camera}")
                return

            frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(frame_rgb)
            if boxes is None or len(boxes) == 0:
                self.log_status("ไม่พบใบหน้าในภาพ")
                return

            box = boxes[0]
            face_tensor = preprocess_face(current_frame, box)
            if face_tensor is None:
                self.log_status("ไม่สามารถสร้าง face tensor ได้")
                return

            with torch.no_grad():
                embedding = resnet(face_tensor).squeeze().cpu().tolist()

            if len(embedding) != 512:
                self.log_status(f"Embedding dimension ไม่ถูกต้อง: {len(embedding)}")
                return

            self.recorded_embeddings.append(embedding)
            self.embeddings_count_label.setText(f"จำนวน embedding ที่บันทึก: {len(self.recorded_embeddings)}")
            self.log_status(f"บันทึก embedding #{len(self.recorded_embeddings)} สำเร็จ")

        except Exception as e:
            self.log_status(f"เกิดข้อผิดพลาดในการบันทึก: {str(e)}")

    def on_camera_changed(self):
        self.camera_preview.setText("กำลังโหลดภาพ...")
        
    def update_preview(self):
        try:
            selected_camera = self.camera_combo.currentText()
            current_frame = camera_frames.get(selected_camera)
            if current_frame is not None:
                rgb_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img).scaled(
                    self.camera_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.camera_preview.setPixmap(pixmap)
            else:
                self.camera_preview.setText(f"ไม่พบภาพจาก {selected_camera}")
        except Exception:
            self.camera_preview.setText("เกิดข้อผิดพลาด")

    def stop_recording(self):
        self.recording = False
        if hasattr(self, 'recording_timer'):
            self.recording_timer.stop()
        if hasattr(self, 'stop_timer'):
            self.stop_timer.stop()
        
        self.record_button.setEnabled(True)
        self.record_button.setText("เริ่มบันทึก (5 วินาที)")
        self.countdown_label.setText("บันทึกเสร็จสิ้น")
        self.countdown_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: white;
                padding: 10px;
                border: 2px solid #2196F3;
                border-radius: 5px;
                background-color: #2196F3;
            }
        """)
        self.progress_label.setText(f"สถานะ: บันทึกเสร็จสิ้น ({len(self.recorded_embeddings)} embedding)")
        
        if len(self.recorded_embeddings) > 0:
            self.save_button.setEnabled(True)
            self.log_status(f"บันทึกเสร็จสิ้น! ได้ {len(self.recorded_embeddings)} embedding")
        else:
            self.log_status("ไม่สามารถบันทึก embedding ได้ กรุณาลองใหม่อีกครั้ง")

    def save_all_embeddings(self):
        if len(self.recorded_embeddings) == 0:
            QMessageBox.warning(self, "ไม่มีข้อมูล", "ไม่มี embedding ที่จะบันทึก")
            return

        employee_id = self.employee_combo.currentData()
        employee_name = self.employee_combo.currentText().split(" - ")[1].split(" (")[0] if " - " in self.employee_combo.currentText() else "Unknown"

        try:
            self.log_status(f"กำลังบันทึก {len(self.recorded_embeddings)} embedding ให้ {employee_id}...")
            
            success_count = 0
            for i, embedding in enumerate(self.recorded_embeddings):
                try:
                    add_embedding_to_milvus(employee_id, employee_name, embedding)
                    success_count += 1
                    self.log_status(f"บันทึก embedding #{i+1} สำเร็จ")
                except Exception as e:
                    self.log_status(f"บันทึก embedding #{i+1} ล้มเหลว: {str(e)}")

            if success_count > 0:
                self.log_status(f"บันทึกสำเร็จ {success_count}/{len(self.recorded_embeddings)} embedding")
                
                self.load_employee_list()
                
                self.log_status("กำลังรีโหลดฐานข้อมูลใบหน้า...")
                from streaming.face_detection import reload_face_database
                reload_face_database()
                self.log_status("รีโหลดฐานข้อมูลใบหน้าเรียบร้อย")
                
                QMessageBox.information(self, "สำเร็จ", f"บันทึก {success_count} embedding ให้ Employee {employee_id} เรียบร้อยแล้ว\nตอนนี้ {employee_id} มี embedding ทั้งหมด {success_count} ตัว\nระบบได้รีโหลดฐานข้อมูลใบหน้าแล้ว")
                self.accept()
            else:
                QMessageBox.critical(self, "ข้อผิดพลาด", "ไม่สามารถบันทึก embedding ได้")

        except Exception as e:
            self.log_status(f"เกิดข้อผิดพลาด: {str(e)}")
            QMessageBox.critical(self, "ข้อผิดพลาด", f"เกิดข้อผิดพลาด: {str(e)}")

    def closeEvent(self, event):
        if self.recording:
            self.stop_recording()
        self.countdown_timer.stop()
        self.preview_timer.stop()
        super().closeEvent(event)



class FaceRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition with Real-time Log")
        self.setGeometry(100, 100, 1200, 600)

        # Multi-camera support
        self.multi_camera = len(RTSP_URLS) > 1
        if self.multi_camera:
            self.camera_names = [f"Camera-{i+1}" for i in range(len(RTSP_URLS))]
        else:
            self.camera_names = ["MainCam"]
        self.video_source_name = self.camera_names[0]
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

        self.log_panel = QVBoxLayout()
        self.log_container = QWidget()
        self.log_container.setLayout(self.log_panel)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.log_container)
        self.scroll.setMinimumWidth(300)



        self.add_face_btn = QPushButton("Add New Employee")
        self.add_face_btn.clicked.connect(self.show_add_face_dialog)


        self.add_image_existing_btn = QPushButton("Add Face to Existing Employee")
        self.add_image_existing_btn.clicked.connect(self.show_add_image_existing_dialog)
        self.add_image_existing_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)

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

        self.camera_container = QWidget()
        self.camera_layout = QGridLayout(self.camera_container)
        self.camera_layout.setContentsMargins(0, 0, 0, 0)
        self.camera_layout.setSpacing(6)
        self.camera_labels = []

        def _compute_grid(n):
            if n <= 0:
                return 0, 0
            import math
            cols = math.ceil(math.sqrt(n))
            rows = math.ceil(n / cols)
            return rows, cols

        rows, cols = _compute_grid(len(self.camera_names))
        if rows == 0:
            rows, cols = 1, 1
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= len(self.camera_names):
                    break
                lbl = QLabel(self)
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                lbl.setStyleSheet("background-color: #ddd;")
                self.camera_layout.addWidget(lbl, r, c)
                self.camera_labels.append(lbl)
                idx += 1
        if len(self.camera_labels) == 1:
            self.camera_labels[0].resizeEvent = lambda event: self.update_frame()

        camera_control_layout = QVBoxLayout()
        camera_control_layout.setContentsMargins(0, 0, 0, 0)
        camera_control_layout.setSpacing(5)
        camera_control_layout.addWidget(self.add_face_btn)

        camera_control_layout.addWidget(self.add_image_existing_btn)

        camera_control_layout.addWidget(self.camera_container)

        camera_widget = QWidget()
        camera_widget.setLayout(camera_control_layout)

        main_layout.addWidget(self.scroll)
        main_layout.addWidget(camera_widget)

        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 4)

        self.setLayout(main_layout)

    def start_video(self):
        if self.multi_camera:
            for idx, src in enumerate(RTSP_URLS[:len(self.camera_names)]):
                threading.Thread(target=process_camera, args=(src, self.camera_names[idx]), daemon=True).start()
        else:
            self.video_source = 0
            threading.Thread(target=process_camera, args=(self.video_source, self.video_source_name), daemon=True).start()
        self.timer.start(30)

    def change_camera(self):
        pass

    def show_add_face_dialog(self):
        # ตรวจสอบว่ามีกล้องใดบ้างที่มีภาพ
        available_cameras = []
        for camera_name in self.camera_names:
            if camera_frames.get(camera_name) is not None:
                available_cameras.append(camera_name)
        
        if not available_cameras:
            QMessageBox.warning(self, "ไม่พบภาพ", "ไม่พบภาพจากกล้องใดๆ กรุณาตรวจสอบการเชื่อมต่อกล้อง")
            return
        
        # ใช้กล้องแรกที่มีภาพเป็นค่าเริ่มต้น
        default_camera = available_cameras[0]
        current_frame = camera_frames.get(default_camera)
        
        frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(frame_rgb)
        if boxes is None or len(boxes) == 0:
            QMessageBox.warning(self, "ไม่พบใบหน้า", f"ไม่พบใบหน้าในภาพจากกล้อง {default_camera} กรุณาตรวจสอบให้แน่ใจว่ามีใบหน้าอยู่ในกล้อง")
            return
        
        self.add_face_btn.setEnabled(False)
        self.add_face_btn.setText("กำลังประมวลผล...")
        
        try:
            dialog = AddFaceDialog(self, current_frame)
            if dialog.exec_() == QDialog.Accepted:
                self.known_embeddings, self.known_names, self.known_employee_ids = load_face_database()
                print("รีโหลดฐานข้อมูลใบหน้าเรียบร้อย")
        finally:
            self.add_face_btn.setEnabled(True)
            self.add_face_btn.setText("Add New Employee")

    def show_add_image_existing_dialog(self):
        # ตรวจสอบว่ามีกล้องใดบ้างที่มีภาพ
        available_cameras = []
        for camera_name in self.camera_names:
            if camera_frames.get(camera_name) is not None:
                available_cameras.append(camera_name)
        
        if not available_cameras:
            QMessageBox.warning(self, "ไม่พบภาพ", "ไม่พบภาพจากกล้องใดๆ กรุณาตรวจสอบการเชื่อมต่อกล้อง")
            return

        # ใช้กล้องแรกที่มีภาพเป็นค่าเริ่มต้น
        default_camera = available_cameras[0]
        current_frame = camera_frames.get(default_camera)

        dialog = AddImageToExistingDialog(self, current_frame)
        if dialog.exec_() == QDialog.Accepted:
            self.known_embeddings, self.known_names, self.known_employee_ids = load_face_database()
            reload_face_database()
            print("รีโหลดฐานข้อมูลใบหน้าหลังจากเพิ่ม embedding ให้พนักงานที่มีอยู่แล้วเรียบร้อย")

    def update_frame(self):
        if self.multi_camera:
            names = self.camera_names
            labels = self.camera_labels
            for i in range(len(labels)):
                if i >= len(names):
                    break
                frame = camera_frames.get(names[i])
                if frame is None:
                    continue
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                q_img = QImage(rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0], QImage.Format_RGB888)
                scaled_pixmap = QPixmap.fromImage(q_img).scaled(
                    labels[i].size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                labels[i].setPixmap(scaled_pixmap)
        else:
            frame = camera_frames.get(self.video_source_name)
            if frame is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                q_img = QImage(rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0], QImage.Format_RGB888)
                scaled_pixmap = QPixmap.fromImage(q_img).scaled(
                    self.camera_labels[0].size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.camera_labels[0].setPixmap(scaled_pixmap)

        try:
            events = get_ui_events(10)
            for ev in events:
                self.log_face_from_jpg(ev["employee_id"], ev["name"], ev["image_jpg"], ev["time"])
        except Exception:
            pass

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


    def log_face_from_jpg(self, employee_id, name, jpg_bytes, when_text):
        img_array = np.frombuffer(jpg_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if bgr is None:
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        date_str, time_str = when_text.split(" ")
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(5, 5, 5, 5)

        pic_label = QLabel(); pic_label.setPixmap(pixmap)
        pic_label.setAlignment(Qt.AlignCenter); pic_label.setStyleSheet("border: 1px solid gray;")
        employee_id_label = QLabel(employee_id); employee_id_label.setAlignment(Qt.AlignCenter)
        employee_id_label.setStyleSheet("color: #007ACC; font-weight: bold; font-size: 11pt;")
        name_label = QLabel(name); name_label.setAlignment(Qt.AlignCenter); name_label.setStyleSheet("font-size: 10pt;")
        date_label = QLabel(date_str); date_label.setAlignment(Qt.AlignCenter); date_label.setStyleSheet("color: gray; font-size: 10pt;")
        time_label = QLabel(time_str); time_label.setAlignment(Qt.AlignCenter); time_label.setStyleSheet("color: gray; font-size: 10pt;")

        layout.addWidget(pic_label)
        layout.addWidget(employee_id_label)
        layout.addWidget(name_label)
        layout.addWidget(date_label)
        layout.addWidget(time_label)
        container.setStyleSheet("background-color: white; padding: 5px;")
        self.log_panel.insertWidget(0, container)

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())