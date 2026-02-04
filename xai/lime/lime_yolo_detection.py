"""
LIME Explanation for YOLO Object Detection

This script produces Local Interpretable Model-agnostic Explanations (LIME) for YOLO 
object detection predictions. It perturbs the image (superpixels) and observes 
how the detection confidence changes to identify which regions are most important 
for a specific detection.

Key Concepts:
- **LIME Image Explainer**: Generates perturbed samples by masking superpixels.
- **Predict Function**: Custom wrapper that feeds perturbed images to YOLO and checks if the original object is still detected.
- **Visualization**: Shows the original detection, LIME superpixel boundaries, and a heatmap of importance.
"""

import numpy as np
import cv2
from lime import lime_image
from skimage.segmentation import mark_boundaries
from ultralytics import YOLO
import matplotlib.pyplot as plt

# COCO class names matching the YOLO trained classes
# COCO_CLASSES = [
#     "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
#     "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
#     "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
#     "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
#     "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
#     "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
#     "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
#     "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
#     "hair drier", "toothbrush"
# ]
CUSTOM_CLASSES = ["apc",
    "army_truck",
    "artillery_gun",
    "bmp",
    "camouflage_soldier",
    "civilian",
    "civilian_vehicle",
    "command_vehicle",
    "engineer_vehicle",
    "fkill",
    "imv",
    "kkill",
    "military_aircraft",
    "military_artillery",
    "military_truck",
    "military_vehicle",
    "military_warship",
    "missile",
    "mkill",
    "mt_lb",
    "reconnaissance_vehicle",
    "rocket",
    "rocket_artillery",
    "soldier",
    "tank",
    "trench",
    "weapon"]

def lime_explain_detection(
    image_path,
    model_path='weights/yolo11m.pt',
    target_box_idx=0,
    num_samples=1000,
    num_features=10,
    output_path='lime_explanation.jpg'
):
    """
    Explain a specific YOLO detection using LIME.
    
    Args:
        image_path (str): Path to input image.
        model_path (str): Path to YOLO `.pt` model weights.
        target_box_idx (int): Index of the detection to explain (sorted by confidence).
                               0 = highest confidence.
        num_samples (int): Number of perturbed image samples to generate. 
                           Higher values = more stable explanations but slower.
        num_features (int): Number of top superpixels (regions) to highlight in visualization.
        output_path (str): File path to save the resulting visualization.
    """
    
    # 1. Load model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # 2. Load and prepare image
    print(f"Loading image from {image_path}...")
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not find image at {image_path}")
    
    # LIME expects RGB
    image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 3. Run initial detection to find target
    print("Running initial detection...")
    results = model.predict(image_rgb, conf=0.25, verbose=False)[0]
    
    if len(results.boxes) == 0:
        print("No objects detected!")
        return
    
    # Sort detections by confidence to make indexing valid
    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    
    sorted_idx = np.argsort(scores)[::-1]
    
    # Handle index out of bounds
    if target_box_idx >= len(sorted_idx):
        print(f"Warning: Only {len(sorted_idx)} boxes detected, using box 0")
        target_box_idx = 0
    
    target_idx = sorted_idx[target_box_idx]
    target_box = boxes[target_idx]
    target_class = int(classes[target_idx])
    target_score = scores[target_idx]
    
    class_name = CUSTOM_CLASSES[target_class] if target_class < len(CUSTOM_CLASSES) else f"Class_{target_class}"
    
    print(f"\nExplaining detection #{target_box_idx}:")
    print(f"  Class: {class_name} (ID: {target_class})")
    print(f"  Confidence: {target_score:.3f}")
    print(f"  Box: [{target_box[0]:.0f}, {target_box[1]:.0f}, {target_box[2]:.0f}, {target_box[3]:.0f}]")
    
    # Helper: Calculate IoU to match perturbations to original box
    def calculate_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    # 4. Define Prediction function for LIME
    # The Black Box: LIME gives us images, we return detection probabilities
    def predict_fn(images):
        """
        Custom prediction callback for LIME.
        
        Args:
            images (list or np.ndarray): Batch of perturbed images.
            
        Returns:
            probs (np.ndarray): Array of shape (N, 2) where the first column is 
                                the score of the target class in the target box.
        """
        results_list = []
        
        for img in images:
            # Run detection on perturbed image with low confidence threshold
            # to catch weak detections caused by occlusion (superpixels off)
            res = model.predict(img, conf=0.1, verbose=False)[0]
            
            if len(res.boxes) == 0:
                # No detection found -> probability 0 for target
                results_list.append([0.0, 1.0])  # [target_prob, background_prob]
                continue
            
            # Find best detection matching the original target box
            pred_boxes = res.boxes.xyxy.cpu().numpy()
            pred_classes = res.boxes.cls.cpu().numpy()
            pred_scores = res.boxes.conf.cpu().numpy()
            
            best_score = 0.0
            
            for i, (box, cls, score) in enumerate(zip(pred_boxes, pred_classes, pred_scores)):
                # Must check Class consistency
                if int(cls) == target_class:
                    iou = calculate_iou(box, target_box)
                    
                    # We weight the score by IoU to ensure we are tracking the *same* object
                    # in the *same* location.
                    combined_score = iou * score
                    best_score = max(best_score, combined_score)
            
            # Return pseudo-probability distribution [target_score, inverse]
            results_list.append([best_score, 1.0 - best_score])
        
        return np.array(results_list)
    
    # 5. Run LIME explanation
    print(f"\nGenerating LIME explanation with {num_samples} samples...")
    print("This may take a few minutes...")
    
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_rgb,
        predict_fn,
        top_labels=2,    # We only care about target vs background
        hide_color=0,    # Fill occluded superpixels with black
        num_samples=num_samples,
        batch_size=10    # Process perturbations in batches
    )
    
    # 6. Visualize explanation
    print("Creating visualization...")
    
    # Get the explanation overlay (label 0 = our target presence)
    temp, mask = explanation.get_image_and_mask(
        0,  
        positive_only=True,
        num_features=num_features,
        hide_rest=False
    )
    
    # Setup PyPlot figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # View 1: Original Detection
    axes[0].imshow(image_rgb)
    x1, y1, x2, y2 = target_box
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=3)
    axes[0].add_patch(rect)
    axes[0].text(x1, y1-10, f"{class_name}: {target_score:.2f}", 
                 color='red', fontsize=12, weight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[0].set_title('Original Detection', fontsize=14, weight='bold')
    axes[0].axis('off')
    
    # View 2: LIME Superpixel Overlay
    axes[1].imshow(mark_boundaries(temp, mask))
    axes[1].set_title(f'LIME Explanation\n(Top {num_features} influential regions)', 
                     fontsize=14, weight='bold')
    axes[1].axis('off')
    
    # View 3: Importance Heatmap
    from skimage.color import gray2rgb
    segments = explanation.segments
    weights = np.zeros(image_rgb.shape[:2])
    
    # Map weights to superpixels
    explanation_map = dict(explanation.local_exp[0])
    for segment_id, weight in explanation_map.items():
        if weight > 0:  # Only positive contributions
            weights[segments == segment_id] = weight
    
    # Normalize heatmap
    if weights.max() > 0:
        weights = weights / weights.max()
    
    axes[2].imshow(image_rgb)
    axes[2].imshow(weights, cmap='jet', alpha=0.5)
    axes[2].set_title('Importance Heatmap\n(Warmer = More Important)', 
                     fontsize=14, weight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nExplanation saved to: {output_path}")
    
    plt.show() # Display interactively if possible
    
    # Print numerical feature importance
    print("\nTop contributing regions (superpixel weights):")
    sorted_features = sorted(explanation.local_exp[0], key=lambda x: abs(x[1]), reverse=True)
    for i, (feature, weight) in enumerate(sorted_features[:10]):
        print(f"  {i+1}. Superpixel {feature}: {weight:+.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LIME Explanation for YOLO")
    parser.add_argument("--image", type=str, default="media/input_images_xai/tank.jpg", help="Path to input image")
    parser.add_argument("--model", type=str, default="weights/yolov11_best.pt", help="Path to YOLO model")
    parser.add_argument("--box_idx", type=int, default=0, help="Index of detection to explain")
    parser.add_argument("--samples", type=int, default=1000, help="Number of LIME samples")
    parser.add_argument("--output", type=str, default="media/lime_outputs/tank_yolo.jpg", help="Output path")
    
    args = parser.parse_args()
    
    lime_explain_detection(
        image_path=args.image,
        model_path=args.model,
        target_box_idx=args.box_idx,
        num_samples=args.samples,
        output_path=args.output
    )

