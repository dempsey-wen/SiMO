# SiMO: Single-Modal-Operable Multimodal Collaborative Perception

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue)](https://openreview.net/forum?id=h0iRgjTmVs)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-black?logo=github)](https://github.com/dempsey-wen/SiMO)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Official PyTorch Implementation**

This repository contains the official implementation of the paper "Single-Modal-Operable Multimodal Collaborative Perception" (ICLR 2026).

---

## Abstract

Multimodal collaborative perception promises robust 3D object detection by fusing complementary sensor data from multiple connected vehicles. However, existing methods suffer from catastrophic performance degradation when one modality becomes unavailable during deployment, a common scenario in real-world autonomous driving. SiMO addresses this critical limitation through two key innovations:

1. **LAMMA (Length-Adaptive Multi-Modal Fusion)**: A novel fusion module that adaptively handles variable numbers of input modalities, operating like a parallel circuit rather than series fusion.

2. **PAFR Training Strategy**: A four-stage training paradigm (Pretrain-Align-Fuse-Random Drop) that prevents modality competition and enables seamless single-modal operation.

| Modality | AP@30 | AP@50 | AP@70 |
|----------|--------|--------|--------|
| LiDAR + Camera | **98.38** | **98.05** | **94.89** |
| LiDAR-only | 97.32 | 97.07 | 94.06 |
| Camera-only | 80.81 | 69.63 | 44.82 |

**Key Result**: SiMO achieves state-of-the-art performance on OPV2V-H with graceful degradation when modalities fail.

---

## Key Features

- **Single-Modal Operability**: First multimodal collaborative perception framework that maintains functional performance with any subset of modalities
- **Adaptive Fusion**: LAMMA module dynamically adjusts to available modalities without architecture changes
- **No Modality Competition**: PAFR training prevents feature suppression between modalities
- **Drop-in Replacement**: Compatible with existing fusion frameworks like HEAL's Pyramid Fusion
- **Multi-Dataset Support**: Evaluated on OPV2V-H, V2XSet, and DAIR-V2X-C

---

## Architecture Overview

### LAMMA (Length-Adaptive Multi-Modal Fusion)

LAMMA is the core fusion module that enables SiMO's single-modal operability:

```
Input: Camera Features (B, N, C, H, W) + LiDAR Features (B, N, C, H, W)
         ↓
    [Positional Encoding]
         ↓
    [Feature Projection] → Downsampling (2x)
         ↓
    [Modality-Aware Masking] ← Single-mode or Random Drop
         ↓
    [Cross-Attention] × 2 (Camera branch + LiDAR branch)
         ↓
    [Parallel Fusion] → Sum of attended features
         ↓
    [Feature Recovery] → Upsampling (2x)
         ↓
Output: Fused Features + Single-Modal Features
```

**Key Design Principles**:
- **Parallel Processing**: Unlike sequential fusion, LAMMA processes modalities in parallel and sums their contributions
- **Adaptive Masking**: During training, random modality dropout forces the network to learn robust single-modal representations
- **Cross-Attention**: Each modality attends to the concatenated features of all available modalities

### Integration with Pyramid Fusion

SiMO works seamlessly with HEAL's Pyramid Fusion framework:

```
Stage 1: Single-Modal Encoders (PointPillar for LiDAR, Lift-Splat-Shoot for Camera)
         ↓
Stage 2: Single-Modal Backbones (ResNet-based BEV feature extraction)
         ↓
Stage 3: Modality Alignment (ConvNeXt-based feature alignment)
         ↓
Stage 4: LAMMA Fusion (Adaptive multimodal fusion)
         ↓
Stage 5: Pyramid Fusion Backbone (Multi-scale collaborative aggregation)
         ↓
Stage 6: Detection Head (Anchor-based 3D object detection)
```

---

## PAFR Training Strategy

The PAFR (Pretrain-Align-Fuse-Random Drop) strategy consists of four stages:

### Stage 1: Pretrain (P)
**Goal**: Train single-modal feature extractors independently

```bash
# Pretrain LiDAR branch
python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/opv2v/LiDAROnly/lidar_pyramid.yaml

# Pretrain Camera branch
python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/opv2v/CameraOnly/camera_pyramid.yaml
```

**Configuration**: Set `freeze: true` for all pretrained components in subsequent stages.

### Stage 2: Align (A)
**Goal**: Align multi-modal features to a common representation space using ConvNeXt

**Key Configuration**:
```yaml
aligner_args:
  core_method: convnext
  freeze: true
  spatial_align: false
  args:
    num_of_blocks: 3
    dim: 64
```

**Training**: Train with both modalities, allowing the aligner to learn cross-modal feature correspondence.

### Stage 3: Fuse (F)
**Goal**: Train LAMMA fusion module with full multimodal inputs

**Key Configuration**:
```yaml
mm_fusion_method: 'lamma3'
lamma:
  freeze: false
  feature_stride: 2
  feat_dim: 64
  dim: 128
  heads: 2
  single_mode: false
  random_drop: false
```

**Important**: Keep `random_drop: false` and `single_mode: false` during this stage.

### Stage 4: Random Drop (RD)
**Goal**: Fine-tune with random modality dropout to enable single-modal operation

**Key Configuration**:
```yaml
lamma:
  random_drop: true
  lidar_drop_ratio: 0.5  # 50% chance to drop LiDAR when dropping
```

**Training**: With 50% probability, randomly drop one modality during training. This forces the network to maintain functional performance with either modality alone.

---

## Installation

This project is implemented based on [HEAL](https://github.com/yifanlu0227/HEAL) and adopts the same environment setup. Please refer to the HEAL repository for detailed installation instructions and troubleshooting.

### Prerequisites

- Python >= 3.8
- PyTorch >= 1.12.0
- CUDA >= 11.3
- spconv >= 2.0

### Step 1: Clone Repository

```bash
git clone https://github.com/dempsey-wen/SiMO.git
cd SiMO
```

### Step 2: Install Dependencies

```bash
# Install PyTorch (adjust CUDA version as needed)
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install spconv (for LiDAR feature extraction)
pip install spconv-cu113

# Install other requirements
pip install -r requirements.txt
```

**Key Dependencies**:
- `easydict~=1.9`
- `opencv-python-headless~=4.5.1.48`
- `timm`
- `einops`
- `shapely==2.0.0`
- `efficientnet_pytorch==0.7.0`

### Step 3: Install OpenCOOD

```bash
pip install -e .
```

### Step 4: Compile CUDA Extensions

```bash
cd opencood/pcdet_utils/pointnet2
python setup.py install
cd ../iou3d_nms
python setup.py install
cd ../../..
```

---

## Data Preparation

### Supported Datasets

SiMO supports the following collaborative perception datasets:

| Dataset | Scenarios | Modalities | Download |
|---------|-----------|------------|----------|
| **OPV2V-H** | Highway, Urban | LiDAR, Camera | [Link](https://mobility-lab.seas.ucla.edu/opv2v/) |
| **V2XSet** | Highway, Urban | LiDAR, Camera | [Link](https://mobility-lab.seas.ucla.edu/v2xset/) |
| **DAIR-V2X-C** | Real-world | LiDAR, Camera | [Link](https://thudair.baai.ac.cn/index) |

### Directory Structure

```
data/
├── OPV2V/
│   ├── train/
│   ├── validate/
│   └── test/
├── V2XSet/
│   ├── train/
│   ├── validate/
│   └── test/
└── DAIR-V2X/
    └── ...
```

---

## Training Commands

### Complete PAFR Pipeline

#### Step 1: Pretrain Single-Modal Branches

```bash
# LiDAR-only pretraining (20 epochs)
python opencood/tools/train.py \
    --hypes_yaml opencood/hypes_yaml/opv2v/LiDAROnly/lidar_pyramid.yaml

# Camera-only pretraining (50 epochs)
python opencood/tools/train.py \
    --hypes_yaml opencood/hypes_yaml/opv2v/CameraOnly/camera_pyramid.yaml
```

**Output**: Model checkpoints saved to `saved_models/opv2v_lidar_pyramid/` and `saved_models/opv2v_camera_pyramid/`

#### Step 2: Train with LAMMA Fusion

```bash
# Train with pretrained branches frozen
python opencood/tools/train.py \
    --hypes_yaml opencood/hypes_yaml/opv2v/MoreModality/lidar_camera_lamma3_pyramid_fusion.yaml
```

**Configuration Notes**:
- Set `model_dir` for each modality to load pretrained weights
- Set `freeze: true` for pretrained components
- Set `lamma.random_drop: false` for initial fusion training

#### Step 3: Random Drop Fine-tuning

Modify the config to enable random dropout:

```yaml
lamma:
  random_drop: true
  lidar_drop_ratio: 0.5
```

Then resume training:

```bash
python opencood/tools/train.py \
    --hypes_yaml opencood/hypes_yaml/opv2v/MoreModality/lidar_camera_lamma3_pyramid_fusion.yaml \
    --model_dir saved_models/opv2v_lidarcamera_lamma3_pyramid_fusion/
```

---

## Testing Commands

### Multimodal Testing (LiDAR + Camera)

```bash
python opencood/tools/inference.py \
    --model_dir saved_models/opv2v_lidarcamera_lamma3_pyramid_fusion/ \
    --fusion_method intermediate
```

### Single-Modal Testing

#### LiDAR-Only Inference

Modify the config to set `single_modality: lidar`:

```yaml
model:
  args:
    single_modality: lidar
```

Then run inference:

```bash
python opencood/tools/inference.py \
    --model_dir saved_models/opv2v_lidarcamera_lamma3_pyramid_fusion/ \
    --fusion_method intermediate
```

#### Camera-Only Inference

```yaml
model:
  args:
    single_modality: camera
```

```bash
python opencood/tools/inference.py \
    --model_dir saved_models/opv2v_lidarcamera_lamma3_pyramid_fusion/ \
    --fusion_method intermediate
```

### Evaluation with Different Ranges

```bash
python opencood/tools/inference.py \
    --model_dir saved_models/opv2v_lidarcamera_lamma3_pyramid_fusion/ \
    --fusion_method intermediate \
    --range 51.2,51.2
```

### Save Visualization

```bash
python opencood/tools/inference.py \
    --model_dir saved_models/opv2v_lidarcamera_lamma3_pyramid_fusion/ \
    --fusion_method intermediate \
    --save_vis_interval 10
```

---

## Benchmark Results

### OPV2V-H Test Set

#### SiMO-PF (Pyramid Fusion + LAMMA)

| Method | Modality | AP@30 | AP@50 | AP@70 | Modality Drop? |
|--------|----------|-------|-------|-------|----------------|
| SiMO-PF | LiDAR + Camera | **98.38** | **98.05** | **94.89** | No |
| SiMO-PF | LiDAR only | 97.32 | 97.07 | 94.06 | Yes |
| SiMO-PF | Camera only | 80.81 | 69.63 | 44.82 | Yes |

**Key Observations**:
- SiMO maintains >97% AP@50 even when operating with LiDAR alone
- Camera-only performance is competitive for near-field detection (AP@30 = 80.81)
- Graceful degradation pattern enables safe fallback strategies

#### Comparison with Baselines

| Method | LiDAR+Camera AP@50 | LiDAR-Only AP@50 | Camera-Only AP@50 |
|--------|-------------------|------------------|-------------------|
| BM2CP (Zhao et al., 2023) | 91.45 | 91.31 | 0.00 |
| BEVFusion (Liu et al., 2023) | 94.21 | 91.99 | 0.00 |
| UniBEV (Wang et al., 2024a) | 91.71 | 91.73 | 0.00 |
| AttFusion (Xu et al., 2022c) | - | 95.09 | 52.91 |
| HEAL (Lu et al., 2024) | - | 98.00 | 60.48 |
| SiMO (AttFusion w/o RD) | 95.26 | 94.02 | 49.69 |
| **SiMO (Pyramid Fusion w/o RD)** (Ours) | **98.05** | **97.07** | **69.63** |

### V2XSet Test Set

| Method | LiDAR+Camera AP@50 | LiDAR-Only AP@50 | Camera-Only AP@50 |
|--------|-------------------|------------------|-------------------|
| SiMO-PF | 92.66 | 90.44 | 56.42 |

### DAIR-V2X-C Test Set

| Method | LiDAR+Camera AP@50 | LiDAR-Only AP@50 | Camera-Only AP@50 |
|--------|-------------------|------------------|-------------------|
| SiMO-PF | 64.51 | 52.33 | 2.24 |
---

## Model Zoo

Pretrained models will be released soon.

| Model | Dataset | Config | Checkpoint |
|-------|---------|--------|------------|
| SiMO-PF | OPV2V-H | [Config](opencood/hypes_yaml/opv2v/MoreModality/lidar_camera_lamma3_pyramid_fusion.yaml) | Coming soon |
| SiMO-AttFuse | OPV2V-H | [Config](opencood/hypes_yaml/opv2v/MoreModality/lidar_camera_lamma3_attfuse.yaml) | Coming soon |

---

## Project Structure

```
SiMO/
├── opencood/
│   ├── models/
│   │   ├── fuse_modules/
│   │   │   ├── lamma.py              # LAMMA implementation
│   │   │   └── pyramid_fuse.py       # Pyramid Fusion
│   │   └── heter_pyramid_collab.py   # Main model
│   ├── tools/
│   │   ├── train.py                  # Training script
│   │   ├── train_ddp.py              # Distributed training
│   │   └── inference.py              # Testing script
│   ├── hypes_yaml/
│   │   └── opv2v/
│   │       ├── LiDAROnly/            # Single-modal configs
│   │       ├── CameraOnly/
│   │       └── MoreModality/         # Multimodal configs
│   └── data_utils/
│       └── datasets/                 # Dataset loaders
├── requirements.txt
├── setup.py
└── README.md
```

---

## Citation

If you find this work useful for your research, please cite:

```bibtex
@inproceedings{wen2026simo,
  title={Single-Modal-Operable Multimodal Collaborative Perception},
  author={Wen, Dempsey and Lu, Yifan and others},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

If you use the OpenCOOD framework, please also cite:

```bibtex
@inproceedings{xu2022opencood,
  title={OpenCOOD: An Open Cooperative Perception Framework for Autonomous Driving},
  author={Xu, Runsheng and Lu, Yifan and others},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2023}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The code is based on [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) and [HEAL](https://github.com/yifanlu0227/HEAL).

---

## Acknowledgements

We thank the authors of OpenCOOD and HEAL for their excellent open-source frameworks. This work builds upon their contributions to collaborative perception research.

---

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.


