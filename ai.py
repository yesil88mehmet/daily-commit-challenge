import cv2
import random
import numpy as np
import networkx as nx
from ultralytics import YOLO
from collections import deque
from scipy.optimize import linear_sum_assignment


# -- YOLO Dedektörü Sınıfı (Hem oyuncu hem top için) --
class YOLODetector:
    def __init__(self, model_path="yolov8n.pt", use_ball_model=False):
        # İki model yüklenebilir: oyuncu ve top için farklı modeller
        self.use_ball_model = use_ball_model
        self.model = YOLO(model_path)
        # Basketbol topu için COCO'da class 32 (top) yok, özel model gerekebilir.
        # Burada topu renk bazlı takipte kullanacağız.

    def detect_players(self, frame):
        results = self.model(frame)[0]
        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # 0: person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append((x1, y1, x2, y2, conf))
        return detections


# -- Top Takip Sınıfı (Renk bazlı basit takip) --
class BallTracker:
    def __init__(self, buffer_len=30):
        self.pts = deque(maxlen=buffer_len)
        # Basketbol topunun turuncu-sarı arası renk aralığı (HSV)
        self.lower_orange = np.array([5, 150, 150])
        self.upper_orange = np.array([20, 255, 255])

    def detect_ball(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_orange, self.upper_orange)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            if M["m00"] > 0 and radius > 5:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                self.pts.appendleft(center)
                return center, radius
        return None, None

    def draw_ball_trajectory(self, frame):
        for i in range(1, len(self.pts)):
            if self.pts[i - 1] is None or self.pts[i] is None:
                continue
            thickness = int(np.sqrt(len(self.pts) / float(i + 1)) * 2.5)
            cv2.line(frame, self.pts[i - 1], self.pts[i], (0, 0, 255), thickness)


# -- Oyuncu Sınıfı --
class Player:
    def __init__(self, bbox, player_id, centroid):
        self.bbox = bbox
        self.id = player_id
        self.centroid = centroid
        self.missed_frames = 0


# -- Oyuncu Takipçisi (Centroid + Hungarian Algoritması ile) --
class PlayerTracker:
    def __init__(self, max_missed=10, max_distance=100):
        self.next_id = 1
        self.players = []
        self.max_missed = max_missed
        self.max_distance = max_distance

    def update(self, detections):
        # detections: (x1,y1,x2,y2,conf)
        input_centroids = np.zeros((len(detections), 2), dtype="int")

        for (i, (x1, y1, x2, y2, conf)) in enumerate(detections):
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.players) == 0:
            for i in range(len(detections)):
                player = Player(detections[i][:4], self.next_id, input_centroids[i])
                self.players.append(player)
                self.next_id += 1
        else:
            object_centroids = np.array([p.centroid for p in self.players])
            D = self._distance_matrix(object_centroids, input_centroids)
            rows, cols = linear_sum_assignment(D)

            assigned_rows = set()
            assigned_cols = set()
            for row, col in zip(rows, cols):
                if D[row, col] > self.max_distance:
                    continue
                self.players[row].bbox = detections[col][:4]
                self.players[row].centroid = input_centroids[col]
                self.players[row].missed_frames = 0
                assigned_rows.add(row)
                assigned_cols.add(col)

            # Missed players
            for i, player in enumerate(self.players):
                if i not in assigned_rows:
                    player.missed_frames += 1

            # Remove players with too many missed frames
            self.players = [p for p in self.players if p.missed_frames <= self.max_missed]

            # Add new players
            for i in range(len(detections)):
                if i not in assigned_cols:
                    player = Player(detections[i][:4], self.next_id, input_centroids[i])
                    self.players.append(player)
                    self.next_id += 1
        return self.players

    def _distance_matrix(self, objects, inputs):
        D = np.linalg.norm(objects[:, np.newaxis] - inputs[np.newaxis, :], axis=2)
        return D


# -- Pas Analizörü --
class PassAnalyzer:
    def __init__(self, max_pass_distance=150, max_time_diff=10):
        self.last_holder = None
        self.last_holder_time = 0
        self.pass_history = []
        self.max_pass_distance = max_pass_distance
        self.max_time_diff = max_time_diff
        self.frame_idx = 0

    def analyze(self, players, ball_pos):
        self.frame_idx += 1
        if ball_pos is None or len(players) == 0:
            return None, None

        ball_x, ball_y = ball_pos

        # Topa en yakın oyuncu
        min_dist = float("inf")
        current_holder = None
        for player in players:
            cX, cY = player.centroid
            dist = np.linalg.norm([cX - ball_x, cY - ball_y])
            if dist < min_dist:
                min_dist = dist
                current_holder = player

        if min_dist > self.max_pass_distance:
            # Top herhangi bir oyuncuda değil
            current_holder = None

        passer, receiver = None, None
        if self.last_holder and current_holder and self.last_holder.id != current_holder.id:
            # Pas gerçekleşti
            time_diff = self.frame_idx - self.last_holder_time
            if time_diff <= self.max_time_diff:
                passer = self.last_holder
                receiver = current_holder
                self.pass_history.append((self.frame_idx, passer.id, receiver.id))

        if current_holder:
            self.last_holder = current_holder
            self.last_holder_time = self.frame_idx
        else:
            self.last_holder = None
            self.last_holder_time = 0

        return passer, receiver


# -- Pas Grafiği --
class PassGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_pass(self, passer, receiver):
        if passer.id == receiver.id:
            return
        if self.graph.has_edge(passer.id, receiver.id):
            self.graph[passer.id][receiver.id]['weight'] += 1
        else:
            self.graph.add_edge(passer.id, receiver.id, weight=1)

    def get_edges(self):
        return list(self.graph.edges(data=True))

    def draw_graph(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.graph)
        weights = [self.graph[u][v]['weight'] for u, v in self.graph.edges()]
        nx.draw(self.graph, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color=weights,
                edge_cmap=plt.cm.Blues, width=3)
        edge_labels = {(u, v): d['weight'] for u, v, d in self.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        plt.title("Pass Network Graph")
        plt.show()


# -- Görselleştirici --
class Visualizer:
    def __init__(self):
        self.colors = {}

    def get_color(self, player_id):
        if player_id not in self.colors:
            self.colors[player_id] = tuple(random.randint(0, 255) for _ in range(3))
        return self.colors[player_id]

    def draw(self, frame, players, ball_pos, passer, receiver, pass_graph, ball_radius=None):
        # Oyuncuları kutu ve ID ile çiz
        for player in players:
            x1, y1, x2, y2 = player.bbox
            color = self.get_color(player.id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Player {player.id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # Centroid noktası
            cv2.circle(frame, player.centroid, 5, color, -1)

        # Topu çiz
        if ball_pos is not None:
            cv2.circle(frame, ball_pos, int(ball_radius) if ball_radius else 10, (0, 140, 255), 3)
            cv2.putText(frame, "Ball", (ball_pos[0] + 10, ball_pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

        # Pası çiz
        if passer and receiver:
            px, py = passer.centroid
            rx, ry = receiver.centroid
            cv2.arrowedLine(frame, (px, py), (rx, ry), (0, 255, 255), 3)
            cv2.putText(frame, f"Pass: {passer.id} -> {receiver.id}",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Pas sayısını yaz
        passes = sum([d['weight'] for _, _, d in pass_graph.graph.edges(data=True)])
        cv2.putText(frame, f"Total Passes: {passes}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        return frame


# -- Ana Akış --
def main(video_path="ba.mp4"):
    cap = cv2.VideoCapture(video_path)

    detector = YOLODetector(model_path="yolov8n.pt")
    ball_tracker = BallTracker()
    tracker = PlayerTracker()
    pass_analyzer = PassAnalyzer()
    pass_graph = PassGraph()
    visualizer = Visualizer()

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Her frame'de oyuncuları algıla
        detections = detector.detect_players(frame)

        # Oyuncuları takip et
        players = tracker.update(detections)

        # Topu algıla
        ball_pos, ball_radius = ball_tracker.detect_ball(frame)
        ball_tracker.draw_ball_trajectory(frame)

        # Pas analizini yap
        passer, receiver = pass_analyzer.analyze(players, ball_pos)
        if passer and receiver:
            pass_graph.add_pass(passer, receiver)

        # Görselleştir
        output_frame = visualizer.draw(frame, players, ball_pos, passer, receiver, pass_graph, ball_radius)

        cv2.imshow("Basketball Pass Analysis", output_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g'):  # 'g' tuşuna basınca pas grafiğini göster
            pass_graph.draw_graph()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()