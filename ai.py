import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
from collections import deque

# YOLOv8 modelini yükleme
model = YOLO('yolov8n.pt')  # YOLOv8 Nano Model

# OCR için EasyOCR okuyucusunu başlatma
reader = easyocr.Reader(['en'])

# Video kaynağı
video_path = 'nba.mp4'
cap = cv2.VideoCapture(video_path)

# Oyuncu ve top takibi için veri yapıları
player_positions = {}
ball_position = None
pass_history = deque(maxlen=20)

def detect_team_color(roi):
    """
    ROI'deki baskın rengi analiz ederek takım rengini belirler.
    """
    avg_color = np.mean(roi, axis=(0, 1))
    if avg_color[2] > avg_color[0]:  # Kırmızı ağırlıklıysa
        return "Team Red"
    elif avg_color[0] > avg_color[2]:  # Mavi ağırlıklıysa
        return "Team Blue"
    else:
        return "Unknown Team"

# Video işleme döngüsü
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 ile nesne algılama
    results = model(frame)

    # Algılamaları döngüyle işleme
    detections = results[0].boxes.data.cpu().numpy()  # Algılanan nesneler

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cls = int(cls)
        if conf < 0.5:  # Güven eşiği
            continue

        if cls == 0:  # Oyuncular
            roi = frame[int(y1):int(y2), int(x1):int(x2)]
            team_color = detect_team_color(roi)

            # OCR ile forma numarası tanıma
            result = reader.readtext(roi)
            number = result[0][1] if result else "Unknown"

            player_id = f"{team_color} #{number}"
            player_positions[player_id] = ((x1 + x2) // 2, (y1 + y2) // 2)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, player_id, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        elif cls == 32:  # Top
            ball_position = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.circle(frame, ball_position, 8, (0, 0, 255), -1)
            cv2.putText(frame, "Ball", (ball_position[0] + 10, ball_position[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Pas tespiti
    if ball_position:
        closest_player, min_distance = None, float('inf')
        for player_id, pos in player_positions.items():
            distance = ((ball_position[0] - pos[0]) ** 2 + (ball_position[1] - pos[1]) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_player = player_id

        if closest_player:
            if pass_history and pass_history[-1] != closest_player:
                pass_history.append(closest_player)

    # Pasları ekrana yazdır
    for i, pas in enumerate(pass_history):
        cv2.putText(frame, f"Pass: {pas}", (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Frame gösterimi
    cv2.imshow('Basketball Analysis', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()