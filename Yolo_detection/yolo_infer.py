import os
from ultralytics import YOLO

# Define paths
WEIGHTS_PATH = "/media/aid-pc/My1TB/Zaheer/Explainable_Object_tacking/weights/yolov11_best.pt"
IMAGE_PATH = "/media/aid-pc/My1TB/Zaheer/Explainable_Object_tacking/media/input_images_xai/tank2.jpg"
OUTPUT_DIR = "/media/aid-pc/My1TB/Zaheer/Explainable_Object_tacking/media/yolo_outputs"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_inference():
    # Check if files exist
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Weights file not found at {WEIGHTS_PATH}")
        return
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image file not found at {IMAGE_PATH}")
        return

    print("Loading model...")
    model = YOLO(WEIGHTS_PATH)

    print(f"Running inference on {IMAGE_PATH}...")
    # Run inference
    # save=True will save the result in 'runs/detect/predict', we can move it or specify project/name
    # visualize=True is optional
    results = model.predict(IMAGE_PATH, save=True, project=OUTPUT_DIR, exist_ok=True)

    print("Inference complete.")
    print(f"Results saved to: {results[0].save_dir}")

if __name__ == "__main__":
    run_inference()
