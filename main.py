import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------
# Simple centroid-based tracker (no 'lap' package needed)
# ---------------------------------------------------------
class CentroidTracker:
    def __init__(self, max_distance=80, max_missed=25, min_hits=3):
        self.next_id = 0
        self.objects = {}       # id -> (cx, cy)
        self.missed = {}        # id -> frames missed in a row
        self.hits = {}          # id -> consecutive frames successfully matched
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.min_hits = min_hits  # a track must be seen this many times before it's "confirmed"

    def update(self, detections):
        """
        detections: list of (cx, cy, x1, y1, x2, y2, name, conf)
        returns: list of (track_id, x1, y1, x2, y2, name, conf, confirmed)
        """
        results = []

        if len(self.objects) == 0:
            for det in detections:
                cx, cy, x1, y1, x2, y2, name, conf = det
                self.objects[self.next_id] = (cx, cy)
                self.missed[self.next_id] = 0
                self.hits[self.next_id] = 1
                confirmed = self.hits[self.next_id] >= self.min_hits
                results.append((self.next_id, x1, y1, x2, y2, name, conf, confirmed))
                self.next_id += 1
            return results

        existing_ids = list(self.objects.keys())
        existing_centers = np.array([self.objects[i] for i in existing_ids])

        assigned_ids = set()
        unmatched_detections = []

        if len(detections) > 0:
            det_centers = np.array([(d[0], d[1]) for d in detections])

            dists = np.linalg.norm(
                existing_centers[:, None, :] - det_centers[None, :, :], axis=2
            )

            used_rows = set()
            used_cols = set()
            pairs = []
            flat = [(dists[r, c], r, c) for r in range(dists.shape[0]) for c in range(dists.shape[1])]
            flat.sort(key=lambda x: x[0])

            for dist, r, c in flat:
                if r in used_rows or c in used_cols:
                    continue
                if dist > self.max_distance:
                    continue
                used_rows.add(r)
                used_cols.add(c)
                pairs.append((r, c))

            for r, c in pairs:
                track_id = existing_ids[r]
                cx, cy, x1, y1, x2, y2, name, conf = detections[c]
                self.objects[track_id] = (cx, cy)
                self.missed[track_id] = 0
                self.hits[track_id] = self.hits.get(track_id, 0) + 1
                assigned_ids.add(track_id)
                confirmed = self.hits[track_id] >= self.min_hits
                results.append((track_id, x1, y1, x2, y2, name, conf, confirmed))

            for c in range(len(detections)):
                if c not in used_cols:
                    unmatched_detections.append(detections[c])
        else:
            unmatched_detections = []

        # Register unmatched detections as new tracks
        for det in unmatched_detections:
            cx, cy, x1, y1, x2, y2, name, conf = det
            self.objects[self.next_id] = (cx, cy)
            self.missed[self.next_id] = 0
            self.hits[self.next_id] = 1
            confirmed = self.hits[self.next_id] >= self.min_hits
            results.append((self.next_id, x1, y1, x2, y2, name, conf, confirmed))
            self.next_id += 1

        # Age out tracks that went unmatched for too long
        for track_id in existing_ids:
            if track_id not in assigned_ids:
                self.missed[track_id] = self.missed.get(track_id, 0) + 1
                if self.missed[track_id] > self.max_missed:
                    del self.objects[track_id]
                    del self.missed[track_id]
                    del self.hits[track_id]

        return results


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------
model = YOLO("yolov8n.pt")
video = cv2.VideoCapture("videos/traffic.mp4")

print("Video opened:", video.isOpened())

vehicle_classes = ["car", "bus", "truck", "motorcycle"]

# Tuned for fast highway traffic:
# - max_distance raised: cars move further between frames on a highway
# - max_missed raised: tolerate brief missed detections without losing the ID
# - min_hits: a track must be confirmed for 3 frames before it can be counted,
#             which filters out flickery/noisy detections
tracker = CentroidTracker(max_distance=45, max_missed=20, min_hits=3)

# Counting line (adjust to fit your video resolution/footage)
line_y = 400
offset = 6

counted_ids = set()
vehicle_count = 0
class_counts = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0}

while True:
    ret, frame = video.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            name = model.names[cls]

            if name in vehicle_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                detections.append((cx, cy, x1, y1, x2, y2, name, conf))

    tracked = tracker.update(detections)

    for track_id, x1, y1, x2, y2, name, conf, confirmed in tracked:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Draw all tracked boxes, but dim unconfirmed ones so it's visually clear
        color = (0, 255, 0) if confirmed else (0, 180, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name} ID:{track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # Only count confirmed tracks (seen for several consecutive frames)
        if confirmed and line_y - offset < cy < line_y + offset and track_id not in counted_ids:
            counted_ids.add(track_id)
            vehicle_count += 1
            if name in class_counts:
                class_counts[name] += 1

    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (255, 0, 0), 2)
    cv2.putText(frame, f"Count: {vehicle_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Small breakdown by vehicle type in the top-right corner
    y_offset = 40
    for cls_name, count in class_counts.items():
        text = f"{cls_name}: {count}"
        cv2.putText(frame, text, (frame.shape[1] - 180, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += 25

    cv2.imshow("Vehicle Tracking and Counting", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

print("\n--- Final Results ---")
print(f"Total vehicles counted: {vehicle_count}")
for cls_name, count in class_counts.items():
    print(f"{cls_name}: {count}")