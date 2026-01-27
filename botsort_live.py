import cv2
from ultralytics import YOLO

model = YOLO("weights/yolov8m.pt")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
    frame,
    tracker="bot_sort.yaml",
    persist=True,
    conf=0.25,
    device="cuda"
)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv11 + BoT-SORT (LIVE)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

