import os
import cv2
from ultralytics import YOLO

os.environ["ULTRALYTICS_OFFLINE"] = "1"
os.environ["YOLO_OFFLINE"] = "1"

# Explicitly define task
model = YOLO("weights/yolo11m.pt", task='detect')

cap = cv2.VideoCapture("input3.mp4")
cap.set(cv2.CAP_PROP_POS_MSEC, 40 * 1000)

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
    
    result = results[0]
    
    if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        
        for box, track_id, conf in zip(boxes, ids, confs):
            x1, y1, x2, y2 = map(int, box)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw ID and confidence
            label = f"ID {track_id} ({conf:.2f})"
            cv2.putText(
                frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
    
    cv2.imshow("Tracking with ReID", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
