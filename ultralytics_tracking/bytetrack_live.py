import cv2
import time
from ultralytics import YOLO

# ----------------------------
# Load YOLO model
# ----------------------------
model = YOLO("weights/yolov8m.pt")  # <-- your trained model

# ----------------------------
# Open webcam (0 = default)
# ----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

print("[INFO] Press 'q' to quit")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ----------------------------
    # YOLO + ByteTrack
    # ----------------------------
    results = model.track(
        frame,
        tracker="byte_track.yaml",
        persist=True,
        device="cuda",
        conf=0.25,
        iou=0.5
    )

    # ----------------------------
    # FPS calculation
    # ----------------------------
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    # ----------------------------
    # Draw results + FPS
    # ----------------------------
    annotated_frame = results[0].plot()

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    cv2.imshow("YOLO + ByteTrack (LIVE)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------
# Cleanup
# ----------------------------
cap.release()
cv2.destroyAllWindows()

