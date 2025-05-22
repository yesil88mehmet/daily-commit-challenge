import cv2
from ultralytics import YOLO

# Modeli yükle (best.pt yolunu kendi modeline göre ayarla)
model = YOLO('best.pt')

# Webcam aç (0 = varsayılan kamera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera açılamadı")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kameradan görüntü alınamadı")
        break

    # Modeli frame üzerinde çalıştır (stream moduyla daha hızlı)
    results = model(frame)

    # Sonuçları frame üzerine çiz
    annotated_frame = results[0].plot()

    # Görüntüyü göster
    cv2.imshow('YOLOv8 Webcam Detection', annotated_frame)

    # 'q' tuşuna basılırsa çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
