import cv2
from utils.detector import PlayerDetector
from utils.tracker import PlayerTracker
from utils.analyzer import PassAnalyzer
from config import Config


def main():
    config = Config()
    detector = PlayerDetector()
    tracker = PlayerTracker()
    analyzer = PassAnalyzer()

    cap = cv2.VideoCapture("ba.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Tespit
        boxes, class_ids = detector.detect(frame)

        # 2. Takip
        tracked_players = tracker.update(boxes)

        # 3. Analiz
        analyzer.update(tracked_players)

        # 4. Görselleştirme
        for track_id, player in tracked_players.items():
            x, y, w, h = player["box"]
            color = config.COLORS[config.CLASSES[class_ids[0]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"ID: {track_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Basketbol Analizi", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Sonuçları göster
    analyzer.visualize()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()