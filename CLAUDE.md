# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DMNet-Denoising is an adaptation of DMNet (ICCV 2023) for point cloud denoising. The original DMNet was a classification network for occupancy prediction; this version predicts (dx, dy, dz) offsets to denoise 3D point clouds using a Graph Convolutional Network with Delaunay triangulation for connectivity.

## Build Commands

**C++ Data Preprocessing Tool:**
```bash
cd create_gt
mkdir build && cd build
cmake ../
make
# Produces: igl_gt, igl_gt_denoise executables
```

**C++ Mesh Generation Tool:**
```bash
cd create_mesh
mkdir build && cd build
cmake ../
make
# Produces: DelaunayMeshing executable
```

**Dependencies:**
- Python 3.9, PyTorch 1.12.1, Open3D 0.16, PyYAML
- C++: CGAL, Boost, libgmp-dev

## Workflow Commands

**Training:**
```bash
cd network
python cache_dataset.py    # Pre-cache data for 10x faster training (run once)
python train_pro.py        # Start training
```

**Testing/Inference:**
```bash
cd network
python test_pro.py         # Outputs to output_denoised/
```

**Data Preprocessing (for new data):**
```bash
cd create_gt
python denoise_data_process.py --noisy <path> --clean <path> --output <path>
python data_process.py     # Build adjacency matrices
```

**Mesh Generation:**
```bash
cd create_mesh
python run.py
```

## Architecture

### Pipeline Flow
```
Noisy Point Cloud + Clean Point Cloud
  ↓ (denoise_data_process.py)
C++ Executable (igl_gt_denoise) [Delaunay Triangulation + Normalization]
  ↓
Graph Structure (cells, adjacency, features)
  ↓ (R-GCN Network in train_pro.py/test_pro.py)
Point Cloud Offsets (dx, dy, dz per vertex)
  ↓ (create_mesh/run.py)
Denoised Point Cloud Output
```

### Key Components

| Directory | Purpose |
|-----------|---------|
| `create_gt/` | Data preprocessing - Delaunay triangulation (C++) and normalization |
| `network/` | PyTorch R-GCN model, training/testing scripts, data loaders |
| `create_mesh/` | Mesh generation from network output |

### Network Architecture (network/R_GCN_model.py)
- **Input**: Point features (6D: xyz + normals)
- **Encoder**: ResBlocks (64→256→64)
- **Graph Convolutions**: Two cycles of Facet↔Node message passing
- **Output Head**: `node_offset_head` predicts (dx, dy, dz) per vertex
- **Loss**: MSE between predicted and ground truth point offsets

### Key Files
- `network/train_pro.py` - Training entry point
- `network/test_pro.py` - Inference entry point
- `network/R_GCN_model.py` - Main model (R_GCN class)
- `network/DT_data.py` - Graph data structures (ScanData, DT_Data)
- `network/losses.py` - Loss functions (`compute_denoising_loss`)
- `network/train_cfg.yaml` - Training configuration
- `create_gt/meshing_denoise.cc` - C++ Delaunay triangulation engine

## Important Implementation Notes

1. **Normalization Alignment**: Both noisy and clean point clouds MUST be normalized using the same transform (calculated from noisy cloud and saved to `transform.txt`). This is critical for correct offset learning.

2. **Dummy Labels**: The data loader expects `ref_point_label.txt` files (legacy from classification). These contain dummy values; the denoising loss ignores them.

3. **Multi-GPU**: Uses custom `DTParallel` wrapper with `DataListLoader` for multi-GPU training.

4. **Memory**: Use `cache_dataset.py` to pre-load data into GPU memory before training iterations.

## Configuration

Training config in `network/train_cfg.yaml`:
- `weight_ratio: [0.9, 0.1, 1.0]` - Loss weights
- `ref_num: 3` - Reference clean points per noisy point
- `k_n: 32` - KNN neighbors for normal estimation
- `data_root_dir` - Path to processed data
