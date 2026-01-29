import numpy as np
import cv2
from lime import lime_image
from skimage.segmentation import mark_boundaries
from ultralytics import YOLO
import matplotlib.pyplot as plt

# COCO class names
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

def lime_explain_detection(
    image_path,
    model_path='weights/yolo11m.pt',
    target_box_idx=0,
    num_samples=1000,
    num_features=10,
    output_path='lime_explanation.jpg'
):
    """
    Explain a YOLO detection using LIME.
    
    Args:
        image_path: Path to input image
        model_path: Path to YOLO model weights
        target_box_idx: Which detected box to explain (0 = highest confidence)
        num_samples: Number of perturbed samples for LIME (higher = better but slower)
        num_features: Number of superpixels to highlight
        output_path: Where to save the explanation
    """
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Load and prepare image
    print(f"Loading image from {image_path}...")
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not find image at {image_path}")
    
    image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Run initial detection to find target box
    print("Running initial detection...")
    results = model.predict(image_rgb, conf=0.25, verbose=False)[0]
    
    if len(results.boxes) == 0:
        print("No objects detected!")
        return
    
    # Sort by confidence and select target
    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    
    sorted_idx = np.argsort(scores)[::-1]
    
    if target_box_idx >= len(sorted_idx):
        print(f"Warning: Only {len(sorted_idx)} boxes detected, using box 0")
        target_box_idx = 0
    
    target_idx = sorted_idx[target_box_idx]
    target_box = boxes[target_idx]
    target_class = int(classes[target_idx])
    target_score = scores[target_idx]
    
    class_name = COCO_CLASSES[target_class] if target_class < len(COCO_CLASSES) else f"Class_{target_class}"
    
    print(f"\nExplaining detection #{target_box_idx}:")
    print(f"  Class: {class_name} (ID: {target_class})")
    print(f"  Confidence: {target_score:.3f}")
    print(f"  Box: [{target_box[0]:.0f}, {target_box[1]:.0f}, {target_box[2]:.0f}, {target_box[3]:.0f}]")
    
    # Calculate IoU between a box and target box
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
    
    # Prediction function for LIME
    def predict_fn(images):
        """
        For each perturbed image, return a score indicating how well
        the target object is still detected.
        """
        results_list = []
        
        for img in images:
            # Run detection on perturbed image
            res = model.predict(img, conf=0.1, verbose=False)[0]
            
            if len(res.boxes) == 0:
                # No detection = 0 probability
                results_list.append([0.0, 1.0])  # [target_prob, background_prob]
                continue
            
            # Find best matching box (highest IoU with target)
            pred_boxes = res.boxes.xyxy.cpu().numpy()
            pred_classes = res.boxes.cls.cpu().numpy()
            pred_scores = res.boxes.conf.cpu().numpy()
            
            best_score = 0.0
            for i, (box, cls, score) in enumerate(zip(pred_boxes, pred_classes, pred_scores)):
                # Only consider boxes of the same class
                if int(cls) == target_class:
                    iou = calculate_iou(box, target_box)
                    # Score is combination of IoU and confidence
                    combined_score = iou * score
                    best_score = max(best_score, combined_score)
            
            # Return as probability distribution [target, background]
            results_list.append([best_score, 1.0 - best_score])
        
        return np.array(results_list)
    
    # Run LIME explanation
    print(f"\nGenerating LIME explanation with {num_samples} samples...")
    print("This may take a few minutes...")
    
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_rgb,
        predict_fn,
        top_labels=2,
        hide_color=0,
        num_samples=num_samples,
        batch_size=10  # Process in batches for efficiency
    )
    
    # Visualize explanation
    print("Creating visualization...")
    
    # Get the explanation for the target class (label 0 = target present)
    temp, mask = explanation.get_image_and_mask(
        0,  # Label 0 represents "target detected"
        positive_only=True,
        num_features=num_features,
        hide_rest=False
    )
    
    # Create figure with multiple views
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image with detection
    axes[0].imshow(image_rgb)
    x1, y1, x2, y2 = target_box
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=3)
    axes[0].add_patch(rect)
    axes[0].text(x1, y1-10, f"{class_name}: {target_score:.2f}", 
                 color='red', fontsize=12, weight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[0].set_title('Original Detection', fontsize=14, weight='bold')
    axes[0].axis('off')
    
    # LIME explanation overlay
    axes[1].imshow(mark_boundaries(temp, mask))
    axes[1].set_title(f'LIME Explanation\n(Top {num_features} influential regions)', 
                     fontsize=14, weight='bold')
    axes[1].axis('off')
    
    # Heatmap view
    from skimage.color import gray2rgb
    # Get weights for all superpixels
    segments = explanation.segments
    weights = np.zeros(image_rgb.shape[:2])
    
    # Get feature weights
    explanation_map = dict(explanation.local_exp[0])
    for segment_id, weight in explanation_map.items():
        if weight > 0:  # Only positive contributions
            weights[segments == segment_id] = weight
    
    # Normalize and create heatmap
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
    
    plt.show()
    
    # Print feature importance
    print("\nTop contributing regions (superpixel weights):")
    sorted_features = sorted(explanation.local_exp[0], key=lambda x: abs(x[1]), reverse=True)
    for i, (feature, weight) in enumerate(sorted_features[:10]):
        print(f"  {i+1}. Superpixel {feature}: {weight:+.4f}")


if __name__ == "__main__":
    # Example usage
    lime_explain_detection(
        image_path="./media/devendar_background.png",
        model_path="weights/yolo11m.pt",
        target_box_idx=0,  # Explain the highest confidence detection
        num_samples=2000,   # Increase for better quality (but slower)
        num_features=20,    # Number of superpixels to highlight
        output_path="media/lime_detection_explanation.jpg"
    )
