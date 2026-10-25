# %%
import cv2
import os
import matplotlib.pyplot as plt
import shutil
import random
import yaml
import pandas as pd
from collections import Counter
from ultralytics import YOLO
from IPython.display import Image, display


# %%
DATASET_ROOT = './dataset'
TRAIN_PATH = os.path.join(DATASET_ROOT, 'train')
TEST_PATH = os.path.join(DATASET_ROOT, 'test')
SAMPLE15_PATH = os.path.join(DATASET_ROOT, 'sample15')

TRAIN_IMAGES = os.path.join(TRAIN_PATH, 'images')
TRAIN_LABELS = os.path.join(TRAIN_PATH, 'labels')
TEST_IMAGES = os.path.join(TEST_PATH, 'images')
TEST_LABELS = os.path.join(TEST_PATH, 'labels')

DATA_YAML_PATH = os.path.join(DATASET_ROOT, 'data.yaml')


# %%
with open(DATA_YAML_PATH, 'r') as f:
    data_yaml = yaml.safe_load(f)
    class_names = data_yaml['names']

print("Classes:", class_names)
print(f"Number of classes: {len(class_names)}")


# %%
def get_image_files(directory, limit=None):
    """Get image files from directory with optional limit."""
    files = [f for f in os.listdir(directory) if f.endswith(('.jpg', '.png'))]
    return files[:limit] if limit else files


# %%
def yolo_to_coords(yolo_box, img_w, img_h):
    """Convert YOLO format to pixel coordinates."""
    cls, x_c, y_c, w_n, h_n = yolo_box
    x1 = int((x_c - w_n / 2) * img_w)
    y1 = int((y_c - h_n / 2) * img_h)
    x2 = int((x_c + w_n / 2) * img_w)
    y2 = int((y_c + h_n / 2) * img_h)
    return x1, y1, x2, y2

# %%
def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    intersection_area = inter_width * inter_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0



# %%
def draw_yolo_labels(img, label_file_path, class_names, color=(0, 255, 0)):
    """Draw YOLO bounding boxes on image."""
    if not os.path.exists(label_file_path):
        return img
        
    h, w = img.shape[:2]
    with open(label_file_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            cls, x, y, bw, bh = map(float, parts[:5])
            
            x1, y1, x2, y2 = yolo_to_coords((cls, x, y, bw, bh), w, h)
            # Draw rectangle and label
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label_text = class_names[int(cls)] if class_names else f'Class {int(cls)}'
            cv2.putText(img, label_text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

# %%
def load_boxes_from_file(file_path):
    """Load bounding boxes from YOLO format file."""
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r') as f:
        return [list(map(float, line.strip().split())) for line in f.readlines()]


# %%
def count_class_instances(label_dir):
    """Count instances of each class in label directory."""
    class_counts = Counter()
    for fname in os.listdir(label_dir):
        if not fname.endswith('.txt'):
            continue
        with open(os.path.join(label_dir, fname), 'r') as f:
            for line in f:
                cls_id = int(line.strip().split()[0])
                class_counts[cls_id] += 1
    return class_counts

# %%
def get_class_ratios(label_dir, class_names):
    """Calculate class distribution ratios."""
    total_count = 0
    class_counts = Counter()

    for fname in os.listdir(label_dir):
        if not fname.endswith('.txt'):
            continue
        with open(os.path.join(label_dir, fname), 'r') as f:
            for line in f:
                cls_id = int(line.strip().split()[0])
                class_counts[cls_id] += 1
                total_count += 1

    class_ratios = {class_names[cls_id]: count / total_count for cls_id, count in class_counts.items()}
    
    # Ensure all classes appear (even if 0%)
    for cls_id, name in enumerate(class_names):
        class_ratios.setdefault(name, 0.0)
    
    return class_ratios



# %%
def compute_overall_iou(img_dir, gt_dir, pred_dir):
    """Compute overall IoU across all images."""
    image_files = get_image_files(img_dir)
    total_ious = []

    for img_name in image_files:
        base = os.path.splitext(img_name)[0]
        gt_label_path = os.path.join(gt_dir, base + '.txt')
        pred_label_path = os.path.join(pred_dir, base + '.txt')
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(gt_label_path):
            continue

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        gt_boxes = load_boxes_from_file(gt_label_path)
        pred_boxes = load_boxes_from_file(pred_label_path)

        for gt_box in gt_boxes:
            best_iou = 0
            for pred_box in pred_boxes:
                if int(gt_box[0]) == int(pred_box[0]):  # Class match
                    iou = calculate_iou(
                        yolo_to_coords(gt_box, w, h),
                        yolo_to_coords(pred_box[:5], w, h)
                    )
                    best_iou = max(best_iou, iou)
            
            if best_iou > 0:
                total_ious.append(best_iou)

    return sum(total_ious) / len(total_ious) if total_ious else 0.0

# %%
print("\nClass distribution in training set:")
for cls_id, count in train_class_counts.items():
    class_name = class_names[cls_id]
    percentage = train_class_ratios[class_name] * 100
    print(f"  {class_name}: {count} instances ({percentage:.1f}%)")

# %%
def create_sample15_dataset():
    """Create sample dataset with 15 random test images."""
    # Get random sample of 15 images
    all_images = get_image_files(TEST_IMAGES)
    sample_images = random.sample(all_images, 15)
    print("Selected images:", sample_images)
    
    # Create directories
    os.makedirs(os.path.join(SAMPLE15_PATH, 'images'), exist_ok=True)
    os.makedirs(os.path.join(SAMPLE15_PATH, 'labels'), exist_ok=True)
    
    # Copy images
    print(f"Copying {len(sample_images)} images...")
    for img in sample_images:
        src_img = os.path.join(TEST_IMAGES, img)
        dst_img = os.path.join(SAMPLE15_PATH, 'images', img)
        shutil.copy(src_img, dst_img)
    
    # Copy labels
    print("Copying corresponding labels...")
    for img in sample_images:
        label_file = img.rsplit('.', 1)[0] + '.txt'
        src_label = os.path.join(TEST_LABELS, label_file)
        dst_label = os.path.join(SAMPLE15_PATH, 'labels', label_file)
        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)
        else:
            print(f"Warning: No label file found for {img}")
    
    print(f"Random sample15 dataset created successfully!")
    return sample_images


sample_images = create_sample15_dataset()

# %%
SAMPLE_IMAGES = './dataset/sample15/images'
SAMPLE_LABELS = './dataset/sample15/labels'

def get_image_files(folder, limit=None):
    valid_exts = ['.jpg', '.png']
    files = [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in valid_exts]
    return files[:limit] if limit else files

# Get up to 15 sample images
image_files = get_image_files(SAMPLE_IMAGES, limit=15)

for fname in image_files:
    img_path = os.path.join(SAMPLE_IMAGES, fname)
    label_path = os.path.join(SAMPLE_LABELS, 
                              fname.replace('.jpg', '.txt').replace('.png', '.txt'))

    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image {img_path}")
        continue

    img_with_labels = draw_yolo_labels(img.copy(), label_path, class_names)
    
    img_rgb = cv2.cvtColor(img_with_labels, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.title(fname)
    plt.axis('off')
    plt.show()

# %%
model = YOLO('best_car_detector.pt')


# %%
sample_results = model.predict(
    source=os.path.join(SAMPLE15_PATH, 'images'),
    save=True,
    save_txt=True,
    project='runs/detect',
    name='predict_sample15'
)


# %%
full_results = model.predict(
    source=TEST_IMAGES,
    save=True,
    save_txt=True,
    project='runs/detect',
    name='predict_fulltest'
)

# %%
sample_val = model.val(data="/Users/rokiialaa/Downloads/CarsDetection/dataset/sample15.yaml", split='test')


# %%
full_val = model.val(data=DATA_YAML_PATH, split='test')


# %%
def compare_gt_vs_predictions(img_dir, gt_dir, pred_dir, limit=15):
    """Compare ground truth vs predictions side by side."""
    image_files = sorted(os.listdir(img_dir))[:limit]

    for img_name in image_files:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        base_name = os.path.splitext(img_name)[0] + '.txt'
        gt_label = os.path.join(gt_dir, base_name)
        pred_label = os.path.join(pred_dir, base_name)

        gt_img = draw_yolo_labels(img.copy(), gt_label, class_names, color=(0,255,0))
        pred_img = draw_yolo_labels(img.copy(), pred_label, class_names, color=(255,0,0))

        # Side by side comparison
        fig, axs = plt.subplots(1, 2, figsize=(12,6))
        axs[0].imshow(gt_img)
        axs[0].set_title('Ground Truth')
        axs[0].axis('off')
        axs[1].imshow(pred_img)
        axs[1].set_title('Prediction')
        axs[1].axis('off')
        plt.tight_layout()
        plt.show()

# %%
# Compare GT vs Predictions for sample15
compare_gt_vs_predictions(
    os.path.join(SAMPLE15_PATH, 'images'),
    TEST_LABELS,
    './runs/detect/predict_sample15/labels'
)


# %%
def compare_gt_vs_predictions_overlay(img_dir, gt_dir, pred_dir, limit=15):
    """Compare ground truth vs predictions on same image."""
    image_files = sorted(os.listdir(img_dir))[:limit]
    
    for img_name in image_files:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        base = os.path.splitext(img_name)[0] + '.txt'
        gt_label = os.path.join(gt_dir, base)
        pred_label = os.path.join(pred_dir, base)

        combined = img.copy()
        combined = draw_yolo_labels(combined, gt_label, class_names, color=(0, 255, 0))  # Green = GT
        combined = draw_yolo_labels(combined, pred_label, class_names, color=(255, 0, 0))  # Red = Pred

        plt.figure(figsize=(10, 8))
        plt.imshow(combined)
        plt.title(f"GT (Green) vs Pred (Red): {img_name}")
        plt.axis('off')
        plt.show()

# %%
# Overlay comparison for sample15
compare_gt_vs_predictions_overlay(
    os.path.join(SAMPLE15_PATH, 'images'),
    TEST_LABELS,
    './runs/detect/predict_sample15/labels'
)

# %%
sample_overall_iou = compute_overall_iou(
        os.path.join(SAMPLE15_PATH, 'images'),
        os.path.join(SAMPLE15_PATH, 'labels'),
        './runs/detect/predict_sample15/labels'
    )
    
full_test_iou = compute_overall_iou(
        TEST_IMAGES,
        TEST_LABELS,
        './runs/detect/predict_fulltest/labels'
    )

# %%
sample_map50 = sample_val.results_dict['metrics/mAP50(B)']
sample_map5095 = sample_val.results_dict['metrics/mAP50-95(B)']

full_map50 = full_val.results_dict['metrics/mAP50(B)']
full_map5095 = full_val.results_dict['metrics/mAP50-95(B)']


# %%
print("\n Sample15 Evaluation:")
print(f"  mAP@0.5:     {sample_map50:.3f}")
print(f"  mAP@0.5:0.95 {sample_map5095:.3f}")
print(f"  Overall IoU: {sample_overall_iou:.3f}")

print("\n Full Test Evaluation:")
print(f"  mAP@0.5:     {full_map50:.3f}")
print(f"  mAP@0.5:0.95 {full_map5095:.3f}")
print(f"  Overall IoU: {full_test_iou:.3f}")

# %%
class_names = sample_val.names


# %%
for i, class_name in sample_val.names.items():
    precision, recall, map50, map95 = sample_val.box.class_result(i)
    print(f"\n Class: {class_name}")
    print(f"   Precision:  {precision:.3f}")
    print(f"   Recall:     {recall:.3f}")
    print(f"   mAP@0.5:    {map50:.3f}")
    print(f"   mAP@0.5:95  {map95:.3f}")


# %%
for i, class_name in full_val.names.items():
    precision, recall, map50, map95 = full_val.box.class_result(i)
    print(f"\n Class: {class_name}")
    print(f"   Precision:  {precision:.3f}")
    print(f"   Recall:     {recall:.3f}")
    print(f"   mAP@0.5:    {map50:.3f}")
    print(f"   mAP@0.5:95  {map95:.3f}")

# %%

import numpy as np

# Collect per-class metrics for sample and full
metrics = {
    "Precision": [],
    "Recall": [],
    "mAP@0.5": [],
    "mAP@0.5:0.95": []
}

full_metrics = {
    "Precision": [],
    "Recall": [],
    "mAP@0.5": [],
    "mAP@0.5:0.95": []
}

class_labels = []

for i, class_name in sample_val.names.items():
    sp, sr, sm50, sm95 = sample_val.box.class_result(i)
    fp, fr, fm50, fm95 = full_val.box.class_result(i)

    class_labels.append(class_name)

    metrics["Precision"].append(sp)
    metrics["Recall"].append(sr)
    metrics["mAP@0.5"].append(sm50)
    metrics["mAP@0.5:0.95"].append(sm95)

    full_metrics["Precision"].append(fp)
    full_metrics["Recall"].append(fr)
    full_metrics["mAP@0.5"].append(fm50)
    full_metrics["mAP@0.5:0.95"].append(fm95)

# Plot grouped bar charts
x = np.arange(len(class_labels))  # Class indices
width = 0.35  # Bar width

fig, axs = plt.subplots(2, 2, figsize=(18, 12))
axs = axs.flatten()

for idx, metric in enumerate(metrics.keys()):
    ax = axs[idx]
    ax.bar(x - width/2, metrics[metric], width, label='Sample15')
    ax.bar(x + width/2, full_metrics[metric], width, label='Full Test')

    ax.set_title(f'{metric} per Class')
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()



