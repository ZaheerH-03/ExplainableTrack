import cv2
import time

cap = cv2.VideoCapture(0)

num_frames = 100
start = time.time()

for _ in range(num_frames):
    ret, frame = cap.read()
    if not ret:
        break

end = time.time()
cap.release()

fps = num_frames / (end - start)
print(f"Actual Webcam FPS: {fps:.2f}")

