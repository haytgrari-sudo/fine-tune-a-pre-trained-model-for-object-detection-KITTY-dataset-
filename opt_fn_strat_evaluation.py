import os
import xml.etree.ElementTree as ET
from math import cos, sin
from typing import List, Optional

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from torchmetrics.detection.mean_ap import MeanAveragePrecision

BATCH_SIZE = 1 
EPOCHS = 5
DEFAULT_LR = 0.005

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_PATH = 'DL Assignment Dataset'
TRAIN_SEQUENCES = ['Video11', 'Video12', 'Video13', 'Video14', 'Video16', 'Video9', 'VideoFour', 'VideoSix', 'VideoThree']
VAL_SEQUENCES = ['Video17']
TEST_SEQUENCES = ['Video15', 'Video18', 'VideoFive', 'VideoSeven']

CLASS_MAP = {
    'Person_sitting': 'Pedestrian',
}

print(f"Executing Task 2 on device: {DEVICE}")

# Dataset Loader: Builds image and annotation pairs for training, validation, and test splits.
class KittiCustomDataset(Dataset):
    def __init__(self, root_dir, sequences, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['Background', 'Car', 'Pedestrian', 'Cyclist']
        self.class_to_id = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.image_paths = []
        self.targets = []

        for seq in sequences:
            seq_root = os.path.join(root_dir, seq)
            image_dir = self._find_image_dir(seq_root)
            tracklet_file = self._find_file(seq_root, 'tracklet_labels.xml')
            calib_file = self._find_file(seq_root, 'calib_cam_to_cam.txt')

            if image_dir is None:
                continue

            projection_matrix = None
            if calib_file is not None:
                projection_matrix = self._load_projection_matrix(calib_file)

            tracklets = []
            if tracklet_file is not None and projection_matrix is not None:
                tracklets = self._parse_tracklets(tracklet_file)

            image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
            for idx, filename in enumerate(image_files):
                full_image_path = os.path.join(image_dir, filename)
                self.image_paths.append(full_image_path)
                self.targets.append(self._build_target_for_frame(idx, tracklets, projection_matrix, full_image_path))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, target

    def _find_file(self, root_dir, filename):
        for dirpath, _, filenames in os.walk(root_dir):
            if filename in filenames:
                return os.path.join(dirpath, filename)
        return None

    def _find_image_dir(self, root_dir):
        for dirpath, _, _ in os.walk(root_dir):
            normalized = dirpath.replace('\\', '/').lower()
            if normalized.endswith('image_02/data'):
                return dirpath
        return None

    def _load_projection_matrix(self, calib_path):
        with open(calib_path, 'r') as f:
            for line in f:
                if line.startswith('P2:') or line.startswith('P_rect_02:'):
                    values = [float(x) for x in line.split()[1:]]
                    return torch.tensor(values, dtype=torch.float32).view(3, 4)
        return None

    def _parse_tracklets(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        tracklets_node = root.find('tracklets')
        if tracklets_node is None:
            tracklets_node = root
            
        tracklets = []

        for item in tracklets_node.findall('item'):
            object_type = item.find('objectType').text
            object_type = CLASS_MAP.get(object_type, object_type)
            first_frame = int(item.find('first_frame').text)
            h, w, l = float(item.find('h').text), float(item.find('w').text), float(item.find('l').text)
            poses = item.find('poses')
            pose_items = poses.findall('item') if poses is not None else []

            if object_type in self.class_to_id:
                tracklets.append({
                    'class': object_type, 'first_frame': first_frame,
                    'h': h, 'w': w, 'l': l, 'poses': pose_items,
                })
        return tracklets

    def _build_target_for_frame(self, frame_idx, tracklets, projection_matrix, image_path):
        boxes, labels = [], []
        image = Image.open(image_path)
        image_width, image_height = image.size
        image.close()

        for tracklet in tracklets:
            local_frame = frame_idx - tracklet['first_frame']
            if local_frame < 0 or local_frame >= len(tracklet['poses']):
                continue

            pose = tracklet['poses'][local_frame]
            try:
                if float(pose.find('truncation').text) >= 99.0: continue
            except: pass

            tx_velo = float(pose.find('tx').text)
            ty_velo = float(pose.find('ty').text)
            tz_velo = float(pose.find('tz').text)
            rz_velo = float(pose.find('rz').text)

            projected = self._project_3d_box(-ty_velo, -tz_velo, tx_velo, tracklet['l'], tracklet['w'], tracklet['h'], -rz_velo, projection_matrix)
            if projected is None: continue

            x_min = max(0.0, min(image_width, projected[0]))
            y_min = max(0.0, min(image_height, projected[1]))
            x_max = max(0.0, min(image_width, projected[2]))
            y_max = max(0.0, min(image_height, projected[3]))

            if x_max > x_min and y_max > y_min:
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(self.class_to_id[tracklet['class']])

        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64)

        return {"boxes": boxes, "labels": labels, "image_id": torch.tensor([frame_idx])}

    def _project_3d_box(self, tx, ty, tz, l, w, h, ry, projection_matrix):
        x_corners = torch.tensor([w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2], dtype=torch.float32)
        y_corners = torch.tensor([0.0, 0.0, 0.0, 0.0, -h, -h, -h, -h], dtype=torch.float32)
        z_corners = torch.tensor([l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2], dtype=torch.float32)
        R = torch.tensor([[cos(ry), 0.0, sin(ry)], [0.0, 1.0, 0.0], [-sin(ry), 0.0, cos(ry)]], dtype=torch.float32)
        
        corners = (R @ torch.stack([x_corners, y_corners, z_corners], dim=0)) + torch.tensor([[tx], [ty], [tz]], dtype=torch.float32)
        corners_h = torch.cat([corners, torch.ones((1, corners.shape[1]), dtype=torch.float32)], dim=0)
        proj = projection_matrix @ corners_h

        valid_mask = proj[2] > 0
        if not valid_mask.any(): return None

        xs, ys = proj[0][valid_mask] / proj[2][valid_mask], proj[1][valid_mask] / proj[2][valid_mask]
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def get_object_detection_model(num_classes, freeze_mode='full_freeze'):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    if freeze_mode == 'unfreeze_layer4':
        for param in model.backbone.body.layer4.parameters():
            param.requires_grad = True

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def train_one_epoch(model, optimizer, data_loader, epoch):
    model.train()
    epoch_loss = 0
    total_batches = len(data_loader)
    
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        epoch_loss += losses.item()
        
        if (i + 1) % 10 == 0 or (i + 1) == total_batches:
            print(f"  -> Batch {i + 1}/{total_batches} complete. Current loss: {losses.item():.4f}")
            
    return epoch_loss

def validate_one_epoch(model, data_loader):
    model.train() 
    epoch_loss = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()
    return epoch_loss

def evaluate_model(model, data_loader, exp_name):
    model.eval() 
    metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True)
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(DEVICE) for image in images)
            outputs = model(images)
            formatted_targets = [{"boxes": t["boxes"].to("cpu"), "labels": t["labels"].to("cpu")} for t in targets]
            formatted_outputs = [{"boxes": o["boxes"].to("cpu"), "scores": o["scores"].to("cpu"), "labels": o["labels"].to("cpu")} for o in outputs]
            metric.update(formatted_outputs, formatted_targets)
    
    results = metric.compute()
    
    # Save a detailed text report for easy copy-pasting into document's tables
    report = (
        f"--- Detailed Metrics for {exp_name} ---\n"
        f"mAP (IoU=0.50:0.95): {results.get('map', 0):.4f}\n"
        f"mAP (IoU=0.50)     : {results.get('map_50', 0):.4f}\n"
        f"mAP (IoU=0.75)     : {results.get('map_75', 0):.4f}\n"
        f"mAP (Small Objects): {results.get('map_small', 0):.4f}\n"
        f"mAP (Large Objects): {results.get('map_large', 0):.4f}\n\n"
    )
    
    with open('visualizations/task2_detailed_metrics.txt', 'a') as f:
        f.write(report)
        
    return results['map_50'].item()

def collate_fn(batch):
    return tuple(zip(*batch))

def plot_loss_curves(train_losses, val_losses, exp_names, save_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    all_loss_values = []

    for i, name in enumerate(exp_names):
        epochs = range(1, len(train_losses[i]) + 1)
        ax.plot(epochs, train_losses[i], color=colors[i], label=f'{name} Train')
        ax.plot(epochs, val_losses[i], color=colors[i], linestyle='--', label=f'{name} Val')
        
        all_loss_values.extend(train_losses[i])
        all_loss_values.extend(val_losses[i])

    max_loss = max(all_loss_values)
    ax.set_ylim(0, max_loss * 1.1)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss Comparison (All Experiments)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_map_comparison(exp_names, map50_scores, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(exp_names, map50_scores, color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
    ax.set_ylabel('mAP @ IoU=0.50')
    ax.set_title('mAP Comparison Between Architectures & Strategies')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    os.makedirs('visualizations', exist_ok=True)
    
    # Metrics Output: Clears the previous summary file so each run starts with a fresh report.
    if os.path.exists('visualizations/task2_detailed_metrics.txt'):
        os.remove('visualizations/task2_detailed_metrics.txt')
    
    standard_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    augmented_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading Baseline Validation and Test Datasets...")
    val_dataset = KittiCustomDataset(DATASET_PATH, sequences=VAL_SEQUENCES, transform=standard_transform)
    test_dataset = KittiCustomDataset(DATASET_PATH, sequences=TEST_SEQUENCES, transform=standard_transform)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    experiments = [
        {'name': '1_Baseline_Freeze',  'freeze_mode': 'full_freeze',     'lr': DEFAULT_LR, 'augment': False},
        {'name': '2_Unfreeze_L4',      'freeze_mode': 'unfreeze_layer4', 'lr': DEFAULT_LR, 'augment': False},
        {'name': '3_Low_Learn_Rate',   'freeze_mode': 'full_freeze',     'lr': 0.001,      'augment': False},
        {'name': '4_Data_Augmented',   'freeze_mode': 'full_freeze',     'lr': DEFAULT_LR, 'augment': True}
    ]

    all_train_losses, all_val_losses, all_maps, exp_names = [], [], [], []

    for exp in experiments:
        exp_name = exp['name']
        exp_names.append(exp_name)
        print(f"\n{'='*50}")
        print(f"Starting Experiment: {exp_name}")
        print(f"Details: Mode: {exp['freeze_mode']} | LR: {exp['lr']} | Augment: {exp['augment']}")
        print(f"{'='*50}")
        
        current_transform = augmented_transform if exp['augment'] else standard_transform
        train_dataset = KittiCustomDataset(DATASET_PATH, sequences=TRAIN_SEQUENCES, transform=current_transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

        model = get_object_detection_model(num_classes=4, freeze_mode=exp['freeze_mode']).to(DEVICE)
        
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=exp['lr'], momentum=0.9, weight_decay=0.0005)

        train_losses, val_losses = [], []
        
        for epoch in range(1, EPOCHS + 1):
            print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
            t_loss = train_one_epoch(model, optimizer, train_loader, epoch)
            v_loss = validate_one_epoch(model, val_loader)
            
            train_losses.append(t_loss)
            val_losses.append(v_loss)
            print(f"> Epoch {epoch} Summary | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")

        print(f"\nEvaluating final model for {exp_name} on test set...")
        map_score = evaluate_model(model, test_loader, exp_name)
        print(f"Final Test mAP@50 for {exp_name}: {map_score:.4f}")

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_maps.append(map_score)
        
        torch.save(model.state_dict(), f'kitti_model_{exp_name}.pth')

    print("\nGenerating report graphs...")
    plot_loss_curves(all_train_losses, all_val_losses, exp_names, 'visualizations/loss_comparison_expanded.png')
    plot_map_comparison(exp_names, all_maps, 'visualizations/map_comparison_expanded.png')
    print("Complete! Check the 'visualizations' folder for your comprehensive graphs and your 'task2_detailed_metrics.txt' file.")