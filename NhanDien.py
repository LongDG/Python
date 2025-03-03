import cv2
import mediapipe as mp
import numpy as np
import threading
import pygame
import os

# Đường dẫn tuyệt đối của file âm thanh cảnh báo
ALERT_SOUND = r"E:\ND\MatKetNoi-DuongDomic-16783113.mp3"

# Kiểm tra xem file có tồn tại không
if not os.path.exists(ALERT_SOUND):
    print(f"❌ Không tìm thấy file âm thanh: {ALERT_SOUND}")
    exit()

# Khởi tạo pygame mixer để phát âm thanh
pygame.mixer.init()

# Khởi tạo MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Điểm mắt trong Face Mesh
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Biến kiểm tra trạng thái phát cảnh báo
alert_playing = False

def play_alert():
    """Hàm phát âm thanh cảnh báo khi ngủ"""
    global alert_playing
    if not alert_playing:
        alert_playing = True
        pygame.mixer.music.load(ALERT_SOUND)
        pygame.mixer.music.play(-1)  # Phát lặp vô hạn

def stop_alert():
    """Hàm dừng âm thanh cảnh báo khi thức"""
    global alert_playing
    if alert_playing:
        pygame.mixer.music.stop()
        alert_playing = False

def eye_aspect_ratio(eye_points, landmarks):
    """Tính toán Eye Aspect Ratio (EAR)"""
    A = np.linalg.norm(np.array(landmarks[eye_points[1]]) - np.array(landmarks[eye_points[5]]))
    B = np.linalg.norm(np.array(landmarks[eye_points[2]]) - np.array(landmarks[eye_points[4]]))
    C = np.linalg.norm(np.array(landmarks[eye_points[0]]) - np.array(landmarks[eye_points[3]]))
    EAR = (A + B) / (2.0 * C)
    return EAR

# Mở webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển ảnh sang RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Lấy tọa độ của mắt
            landmarks = {i: (int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for i, point in enumerate(face_landmarks.landmark)}

            # Tính toán EAR
            left_ear = eye_aspect_ratio(LEFT_EYE, landmarks)
            right_ear = eye_aspect_ratio(RIGHT_EYE, landmarks)
            ear = (left_ear + right_ear) / 2.0

            # Kiểm tra trạng thái ngủ/thức
            if ear > 0.25:
                status = "Thức"
                color = (0, 255, 0)
                stop_alert()  # Dừng nhạc nếu đang phát
            else:
                status = "Ngủ"
                color = (0, 0, 255)
                if not alert_playing:
                    threading.Thread(target=play_alert).start()  # Chạy nhạc nếu chưa phát

            # Hiển thị trạng thái lên màn hình
            cv2.putText(frame, f"Trang thai: {status}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Hiển thị hình ảnh
    cv2.imshow("Sleep Detection", frame)

    # Nhấn 'Q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
