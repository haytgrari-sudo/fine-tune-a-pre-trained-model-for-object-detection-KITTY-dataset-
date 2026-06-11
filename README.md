# Object Detection with Faster R-CNN: KITTI Benchmark

## Project Overview
This repository transitions from controlled image classification to the complexity of real-world autonomous vehicle footage. The objective is full object detection: determining not only what an object is, but precisely where it is located within a scene.

Using the KITTI Vision Benchmark Suite, this project fine-tunes a pre-trained Faster R-CNN model to detect and localize targets on the road. A rigorous four-experiment ablation study is conducted to evaluate the effects of layer freezing strategies, learning rate selection, and data augmentation on detection performance and training stability.

## Repository Structure
* **`run_opt_fn_strat.py` (Data Pipeline & Visualization):** This script handles data engineering and verification. It features a custom `KittiCustomDataset` class that parses 3D tracklet XML files, applies camera calibration matrices, and projects 3D coordinates into 2D axis-aligned bounding boxes. It generates color-coded ground truth samples for 7 distinct vehicle/pedestrian classes to verify the integrity of the data pipeline before training begins.
* **`opt_fn_strat_evaluation.py` (Training & Ablation Study):** The core experimentation script. It consolidates the vehicle subtypes into primary classes (Cars, Pedestrians, Cyclists) for robust detection and executes the training loop across four distinct configurations. Post-training, it evaluates the model on unseen sequences, computes comprehensive Mean Average Precision (mAP) metrics (including breakdowns for small/large objects and varying IoU thresholds), and generates visual comparative graphs.

## How It Works (Network Architecture)
Object detection requires balancing computational speed with bounding-box accuracy. This project utilizes a two-stage detector architecture:

1. **Feature Extraction (ResNet-50 FPN):** Processes the raw image to create a rich feature map. The backbone network applies deep residual layers to the input image. Pre-trained on ImageNet, it already possesses a robust vocabulary of visual patterns (edges, textures) that transfer perfectly to complex road scenes, ensuring stable feature extraction.
2. **Region Proposal Network (RPN):** Scans the feature map to generate candidate object regions. Rather than sliding a window across the entire image, this dedicated network efficiently proposes regions of interest that have a high probability of containing targets like cars or pedestrians.
3. **Detection Head:** Classifies and spatially refines each proposed region. Each proposed region is individually classified and adjusted to output the exact bounding box coordinates and class probability, making two-stage detectors highly accurate for cluttered, real-world driving scenes.

## Experimentation & Results

To determine the optimal fine-tuning approach, four distinct experimental configurations were tested over 5 epochs. Performance was measured using Mean Average Precision (mAP) at an IoU threshold of 0.50.

| Experiment Strategy | Final mAP | Training Behavior & Stability |
| :--- | :--- | :--- |
| **1. Baseline Full Freeze** | **0.077** | **The optimal pipeline.** Steady, healthy drop in loss. The frozen pre-trained backbone acted as a reliable foundation. |
| **2. Low Learning Rate** | 0.065 | Stable training, but the reduced learning rate meant the model simply ran out of time to fully learn targets within the 5-epoch limit. |
| **3. Data Augmentation** | 0.061 | Color jittering (shifting brightness/contrast) was absorbed by the frozen backbone and did not meaningfully improve generalization to outdoor lighting. |
| **4. Unfreeze Layer 4** | 0.000 | Catastrophic failure. The training gradients exploded (returning NaN errors), leaving the model completely broken. |

### The Impact of Catastrophic Forgetting
The ablation study emphatically proves the necessity of controlled fine-tuning. The "Baseline Full Freeze" succeeded because locking the ResNet-50 backbone completely protected its vast library of pre-trained visual features, allowing the new detection head to accurately locate cars, pedestrians, and cyclists.

Conversely, the "Unfreeze Layer 4" experiment failed completely. The standard learning rate (0.005) was far too aggressive for delicately tuned pre-trained weights. The model aggressively overwrote and destroyed its foundational ImageNet knowledge—a textbook case of catastrophic forgetting—resulting in an inability to predict a single valid bounding box.

<img width="1460" height="800" alt="Capture d&#39;écran 2026-06-11 152532" src="https://github.com/user-attachments/assets/b00cbe08-87b4-4b07-8cd4-13004b9d969d" />

<img width="1460" height="800" alt="Capture d&#39;écran 2026-06-11 152554" src="https://github.com/user-attachments/assets/079510ff-4f76-4bbe-b829-26927a19e1f8" />

<img width="950" height="323" alt="gt_sample_6_(idx_1745)" src="https://github.com/user-attachments/assets/e0c6f1b9-8883-49f4-9483-0ee6e1c9bfca" />

<img width="950" height="323" alt="gt_sample_2_(idx_1483)" src="https://github.com/user-attachments/assets/b4c060f7-c247-4173-89de-d38f5c24632a" />

<img width="950" height="323" alt="gt_sample_8_(idx_1986)" src="https://github.com/user-attachments/assets/0e0c70bd-dd83-4a7d-a03b-b64d1953fb0e" />

<img width="950" height="323" alt="gt_sample_13_(idx_215)" src="https://github.com/user-attachments/assets/9260e087-151b-42ca-9e3b-65c61fa1037b" />

## Setup & Requirements

To run this project locally, first download the kitty dataset: https://drive.google.com/drive/folders/1KcCtqmh89UQrbv1BehyNLI8TZ2ZN9SYY?usp=drive_link

Then ensure you have Python installed, and the following requirements: 

torch,
torchvision,
torchmetrics,
matplotlib,
Pillow

```
