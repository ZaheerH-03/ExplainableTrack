import cv2
import time
import argparse
import collections
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO

# Pipeline Utils
from pipeline_utils.image_enhancer import ImageEnhancer
from pipeline_utils.explainer import Explainer
from pipeline_utils.tracker import ObjectTracker

# --- Configuration ---
LAPLACIAN_VAR_THRESHOLD = 50.0  # Lower = Blurry
# SRGAN not used by default on full frame due to OOM risk, only if flag enabled
DO_SRGAN = False 

def get_variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def main():
    parser = argparse.ArgumentParser(description="Unified Explainable Tracking Pipeline")
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam, or path)")
    
    # Model Args
    parser.add_argument("--model_type", type=str, default="rtdetr", choices=["yolo", "rtdetr"], help="Model type: yolo or rtdetr")
    parser.add_argument("--weights", type=str, default="weights/best_stg2.pth", help="Path to model weights")
    parser.add_argument("--config", type=str, default="RT-DETRv4/configs/rtv4/rtv4_x_custom.yml", help="Path to config (RT-DETR only)")
    parser.add_argument("--deblur_weights", type=str, default="weights/generator_deblur.pth", help="Path to DeblurGAN weights")
    
    # Flags
    parser.add_argument("--enable_deblur", action="store_true", help="Enable DeblurGAN on blurry frames")
    parser.add_argument("--enable_xai", action="store_true", help="Enable EigenCAM for new tracks")
    parser.add_argument("--enable_sr", action="store_true", help="Enable SRGAN (Warning: High Memory usage!)")
    
    args = parser.parse_args()

    # 1. Initialize Models
    print("--- Initializing Pipeline ---")
    
    # Tracker (Detection + Tracking)
    print(f"Loading Tracker: {args.model_type} | Weights: {args.weights}")
    tracker = ObjectTracker(
        model_type=args.model_type, 
        weights_path=args.weights, 
        config_path=args.config
    )
    
    # Image Enhancer
    enhancer = None
    if args.enable_deblur or args.enable_sr:
        print("Loading Image Enhancer...")
        sr_weights = "weights/generator_sr.pth" if args.enable_sr else None
        deblur_weights = args.deblur_weights if args.enable_deblur else None
        enhancer = ImageEnhancer(deblur_weights=deblur_weights, sr_weights=sr_weights)

    # Explainer
    explainer = None
    if args.enable_xai:
        print("Loading Explainer (EigenCAM)...")
        # Get underlying model for XAI
        core_model = tracker.get_model_object()
        explainer = Explainer(core_model, model_type=args.model_type)

    # 2. Open Video
    source = args.source
    if source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    # Tracking State
    # Keep track of IDs we have already explained
    explained_ids = set()
    # Store latest explanation thumbnails: {id: thumbnail_img}
    xai_thumbnails = {}
    
    fps_history = collections.deque(maxlen=30)
    
    print("--- Starting Loop (Press 'q' to quit) ---")
    
    frame_count = 0
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        
        # Save original frame before any processing (for XAI)
        original_frame = frame.copy()
        
        display_frame = frame.copy()
        processed_frame = frame # Frame to be passed to detector
        
        # --- A. Enhancement Stage ---
        is_blurry = False
        if enhancer:
            # Check Blur
            lap_var = get_variance_of_laplacian(frame)
            if lap_var < LAPLACIAN_VAR_THRESHOLD:
                is_blurry = True
                
            if args.enable_deblur and is_blurry:
                # Deblur
                try:
                    processed_frame = enhancer.deblur(processed_frame)
                    cv2.putText(display_frame, "DEBLUR APPLIED", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                except Exception as e:
                    print(f"Deblur Error: {e}")

            if args.enable_sr:
                # SRGAN triggering: Check if image is small/low-res
                # e.g. if height < 720
                h, w = processed_frame.shape[:2]
                if h < 720:
                    try:
                        processed_frame = enhancer.upscale(processed_frame)
                        # Sync display frame (naive, might mismatch boxes if not careful, keeping simple)
                        display_frame = processed_frame.copy() 
                        cv2.putText(display_frame, "SRGAN APPLIED", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    except Exception as e:
                         print(f"SRGAN Error: {e}")

        # --- B. Tracking Stage ---
        # Returns list of [x1, y1, x2, y2, id, cls_id, conf]
        try:
            detections = tracker.update(processed_frame)
        except Exception as e:
            print(f"Tracking Error: {e}")
            detections = []
        
        # --- C. XAI & Visualization Stage ---
        # Ensure output dir
        xai_out_dir = Path("media/eigen_cam_outputs_pipeline")
        xai_out_dir.mkdir(parents=True, exist_ok=True)
        
        for det in detections:
            x1, y1, x2, y2, track_id, cls_id, conf = det
            x1, y1, x2, y2, track_id, cls_id = int(x1), int(y1), int(x2), int(y2), int(track_id), int(cls_id)
            
            # --- Visualization ---
            color = (0, 255, 0) # Green for box
            
            # Draw Box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Class Name Logic
            if args.model_type == 'yolo':
                 class_name = tracker.model.names[cls_id]
            else:
                 class_name = str(cls_id)
            
            label_text = f"{track_id} {class_name} {conf:.2f}"
            
            # Draw Label with efficient background
            t_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            # Draw background rectangle for text (Inside top of box)
            cv2.rectangle(display_frame, (x1, y1), (x1 + t_size[0], y1 + t_size[1] + 5), color, -1)
            # Draw text in Black (contrasts well with Green box)
            cv2.putText(display_frame, label_text, (x1, y1 + t_size[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Check XAI Trigger
            if args.enable_xai and explainer and track_id not in explained_ids:
                print(f"Generating Explanation for New Track ID: {track_id}")
                # Mark as explained immediately to prevent infinite retries on failure
                explained_ids.add(track_id)
                
                try:
                    # Use original unprocessed frame for XAI analysis
                    heatmap = explainer.generate_eigencam(original_frame)
                    if heatmap is not None:
                        # Save Full Visualization
                        save_name = xai_out_dir / f"track_{track_id}_class_{class_name}.jpg"
                        cv2.imwrite(str(save_name), heatmap)
                        print(f"Saved XAI to {save_name}")

                        # Ensure ID is marked as explained to prevent infinite retries if crop fails
                        # (Already marked above)


                        
                except Exception as e:
                    print(f"XAI Error: {e}")
                    # Optionally mark as explained even on error to stop retrying?
                    # explained_ids.add(track_id)



        # FPS Calc
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        fps_history.append(fps)
        avg_fps = sum(fps_history) / len(fps_history)
        
        cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Unified Pipeline", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
