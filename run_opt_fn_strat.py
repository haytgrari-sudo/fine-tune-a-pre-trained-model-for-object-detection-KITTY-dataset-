import os
import random
import re
import xml.etree.ElementTree as ET
from math import cos, sin
from typing import List, Optional

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle
from torch.utils.data import Dataset
from PIL import Image

# Configuration: Stores the dataset location and class mapping settings for GT visualization.
DATASET_PATH = 'DL Assignment Dataset'

TRAIN_SEQUENCES = ['Video11', 'Video12', 'Video13', 'Video14', 'Video16', 'Video9', 'VideoFour', 'VideoSix', 'VideoThree']

# Class Mapping: Treats sitting people as pedestrians
CLASS_MAP = {
    'Person_sitting': 'Pedestrian',
}

# Class Colors: Assigns a consistent display color to each supported object class.
CLASS_COLORS = {
    'Background': 'black',
    'Car': '#1f77b4',       # Car Color: Uses blue for car boxes.
    'Van': '#17becf',       # Van Color: Uses cyan for van boxes.
    'Truck': '#ff7f0e',     # Truck Color: Uses orange for truck boxes.
    'Tram': '#9467bd',      # Tram Color: Uses purple for tram boxes.
    'Pedestrian': '#d62728',# Pedestrian Color: Uses red for pedestrian boxes.
    'Cyclist': '#2ca02c'    # Cyclist Color: Uses green for cyclist boxes.
}

# Custom Dataset: Loads KITTI images and tracklets for ground-truth visualization only.
class KittiCustomDataset(Dataset):
    def __init__(self, root_dir, sequences, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Class List: Defines the seven supported classes used for IDs and plotting.
        self.classes = ['Background', 'Car', 'Van', 'Truck', 'Tram', 'Pedestrian', 'Cyclist']
        self.class_to_id = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_paths = []
        self.targets = []

        # Sequence Loading: Iterates through each selected video sequence and gathers its images and annotations.
        for seq in sequences:
            seq_root = os.path.join(root_dir, seq)
            image_dir = self._find_image_dir(seq_root)
            tracklet_file = self._find_file(seq_root, 'tracklet_labels.xml')
            calib_file = self._find_file(seq_root, 'calib_cam_to_cam.txt')

            if image_dir is None:
                continue

            # Calibration Data: Loads camera projection information so 3D annotations can be projected into image coordinates.
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

        # Tracklet Parsing: Reads each annotation entry and converts it into a normalized structure for later use.
        for item in tracklets_node.findall('item'):
            object_type = item.find('objectType').text
            object_type = CLASS_MAP.get(object_type, object_type)
            first_frame = int(item.find('first_frame').text)
            h = float(item.find('h').text)
            w = float(item.find('w').text)
            l = float(item.find('l').text)
            poses = item.find('poses')
            pose_items = poses.findall('item') if poses is not None else []

            if object_type not in self.class_to_id:
                continue

            tracklets.append({
                'class': object_type,
                'first_frame': first_frame,
                'h': h,
                'w': w,
                'l': l,
                'poses': pose_items,
            })

        return tracklets

    def _build_target_for_frame(self, frame_idx, tracklets, projection_matrix, image_path):
        boxes = []
        labels = []

        image = Image.open(image_path)
        image_width, image_height = image.size
        image.close()

        # Frame Projection: Projects each tracklet into the current frame using its pose and the camera matrix.
        for tracklet in tracklets:
            local_frame = frame_idx - tracklet['first_frame']
            if local_frame < 0 or local_frame >= len(tracklet['poses']):
                continue

            pose = tracklet['poses'][local_frame]
            trunc_node = pose.find('truncation')
            if trunc_node is not None:
                try:
                    trunc_val = float(trunc_node.text)
                    if trunc_val >= 99.0:
                        continue
                except Exception:
                    pass
            
            tx_velo = float(pose.find('tx').text)
            ty_velo = float(pose.find('ty').text)
            tz_velo = float(pose.find('tz').text)
            rz_velo = float(pose.find('rz').text)

            cam_x = -ty_velo
            cam_y = -tz_velo
            cam_z = tx_velo
            ry = -rz_velo 

            projected = self._project_3d_box(cam_x, cam_y, cam_z, tracklet['l'], tracklet['w'], tracklet['h'], ry, projection_matrix)
            if projected is None:
                continue

            x_min, y_min, x_max, y_max = projected
            x_min = max(0.0, min(image_width, x_min))
            x_max = max(0.0, min(image_width, x_max))
            y_min = max(0.0, min(image_height, y_min))
            y_max = max(0.0, min(image_height, y_max))

            if x_max <= x_min or y_max <= y_min:
                continue

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(self.class_to_id[tracklet['class']])

        # Box Assembly: Converts the collected bounding boxes and labels into tensors for visualization.
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)

        return {"boxes": boxes, "labels": labels, "image_id": torch.tensor([frame_idx])}

    def _project_3d_box(self, tx, ty, tz, l, w, h, ry, projection_matrix):
        corners = self._compute_3d_box_corners(l, w, h, ry)
        corners = corners + torch.tensor([[tx], [ty], [tz]], dtype=torch.float32)
        corners_h = torch.cat([corners, torch.ones((1, corners.shape[1]), dtype=torch.float32)], dim=0)
        proj = projection_matrix @ corners_h

        valid_mask = proj[2] > 0
        if not valid_mask.any():
            return None

        xs = proj[0] / proj[2]
        ys = proj[1] / proj[2]
        xs = xs[valid_mask]
        ys = ys[valid_mask]

        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

    def _compute_3d_box_corners(self, l, w, h, ry):
        x_corners = torch.tensor([w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2], dtype=torch.float32)
        y_corners = torch.tensor([0.0, 0.0, 0.0, 0.0, -h, -h, -h, -h], dtype=torch.float32)
        z_corners = torch.tensor([l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2], dtype=torch.float32)
        R = torch.tensor([
            [cos(ry), 0.0, sin(ry)],
            [0.0, 1.0, 0.0],
            [-sin(ry), 0.0, cos(ry)]
        ], dtype=torch.float32)
        corners = torch.stack([x_corners, y_corners, z_corners], dim=0)
        return R @ corners

# Visualization Helpers: Draws boxes on images and saves annotated sample outputs.
def visualize_boxes(image: Image.Image, boxes: torch.Tensor, labels: torch.Tensor, class_names: List[str], title: str, save_path: Optional[str] = None):
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    # Box Drawing: Draws each predicted or ground-truth box and adds its class label to the image.
    for box, label in zip(boxes, labels):
        x_min, y_min, x_max, y_max = box.tolist()
        
        class_name = class_names[label]
        color = CLASS_COLORS.get(class_name, 'red') 
        
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                         linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        ax.text(x_min, y_min - 5, class_name, color=color, fontsize=10, weight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                
    ax.axis('off')
    ax.set_title(title)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def visualize_sample(dataset: Dataset, index: int, title: str, save_path: Optional[str] = None):
    image = Image.open(dataset.image_paths[index]).convert('RGB')
    target = dataset.targets[index]
    
    visualize_boxes(image, target['boxes'], target['labels'], 
                    ['Background', 'Car', 'Van', 'Truck', 'Tram', 'Pedestrian', 'Cyclist'], 
                    title, save_path)


if __name__ == '__main__':
    print("Loading KITTI dataset for visualization...")
    train_dataset = KittiCustomDataset(DATASET_PATH, sequences=TRAIN_SEQUENCES)
    print(f"Loaded {len(train_dataset)} total images.")

    # Visualizations Folder: Creates the directory where generated sample images will be stored.
    VIS_DIR = 'visualizations'
    os.makedirs(VIS_DIR, exist_ok=True)
    
    # File Naming: Finds the next available sample number so new outputs do not overwrite existing ones.
    highest_n = 0
    existing_files = os.listdir(VIS_DIR)
    
    for file in existing_files:
        # Looking for files that match the pattern gt_sample_X_(idx_Y).png
        match = re.search(r'gt_sample_(\d+)_', file)
        if match:
            current_n = int(match.group(1))
            if current_n > highest_n:
                highest_n = current_n

    start_n = highest_n + 1
    NUM_VISUALIZATIONS = 5 

    if len(train_dataset) > 0:
        print(f"\nExisting images found up to n={highest_n}. Starting new generation at n={start_n}.")
        print(f"Generating {NUM_VISUALIZATIONS} random Ground Truth samples directly into '{VIS_DIR}/'...")
        
        # Random Sampling: Selects a few frames from the dataset so the generated visualizations are varied.
        gt_indices = random.sample(range(len(train_dataset)), min(NUM_VISUALIZATIONS, len(train_dataset)))
        
        for i, idx in enumerate(gt_indices):
            current_n = start_n + i
            save_name = os.path.join(VIS_DIR, f'gt_sample_{current_n}_(idx_{idx}).png')
            
            visualize_sample(train_dataset, idx, f'Ground Truth Sample (Index {idx})', save_path=save_name)
            
        print(f"\nSuccess! Check the '{VIS_DIR}' folder for your {NUM_VISUALIZATIONS} new images.")