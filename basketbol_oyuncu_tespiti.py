import cv2
import requests

API_KEY = "EsS4tcNxLdEdo8QIiHGu"
MODEL_ENDPOINT = "https://detect.roboflow.com/basketball-players-fy4c2/25"
video_path = "ba.mp4"

def detect_players(frame):
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = img_encoded.tobytes()
    url = f"{MODEL_ENDPOINT}?api_key={API_KEY}"

    try:
        response = requests.post(url, files={"file": img_bytes})
        predictions = response.json()
        return predictions
    except Exception as e:
        print(f"Hata oluştu: {e}")
        return {"predictions": []}

def main():
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        predictions = detect_players(frame)
        print("Tahminler:", predictions.get("predictions", []))

        for pred in predictions.get("predictions", []):
            label = pred["class"]
            if label not in ["Ref", "Ball", "Player", "Hoop"]:
                continue  # Bu sınıfı atla

            x = int(pred["x"])
            y = int(pred["y"])
            w = int(pred["width"])
            h = int(pred["height"])
            conf = pred["confidence"]

            cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x - w // 2, y - h // 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Basketball Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
