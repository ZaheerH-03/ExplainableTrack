import cv2
from ultralytics import YOLO

# ----------------------------
# Load YOLO model
# ----------------------------
model = YOLO("weights/yolo11m.pt")  # change to your trained model

# ----------------------------
# Open webcam (0 = default)
# ----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

print("[INFO] Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ----------------------------
    # YOLO + DeepSORT tracking
    # ----------------------------
    results = model.track(
        frame,
        tracker="deep_sort.yaml",
        persist=True,
        device="cuda",
        conf=0.4,
        iou=0.5
    )

    # ----------------------------
    # Draw results
    # ----------------------------
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO + DeepSORT (LIVE)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------
# Cleanup
# ----------------------------
cap.release()
cv2.destroyAllWindows()

