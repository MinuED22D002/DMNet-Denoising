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
- Python 3.9, PyTorch 1.12.1+, Open3D 0.16, PyYAML
- C++: CGAL, Boost, libgmp-dev, libmpfr-dev
- Docker: Dockerfile provided (Ubuntu 20.04 base, builds Open3D from source)

## Workflow Commands

**Training:**
```bash
cd network
python cache_dataset.py    # Pre-cache data to cached_train_data.pt / cached_val_data.pt (run once, 10x speedup)
python train_pro.py        # Start training
```

**Testing/Inference:**
```bash
cd network
python test_pro.py         # Outputs denoised PLY files to experiment_dir/output_denoised/
```

**Data Preprocessing (for new data):**
```bash
cd create_gt
python denoise_data_process.py --noisy <dir> --clean <dir> --out <dir>
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

### Network Architecture (network/R_GCN_model.py)
- **Input**: Point features (6D: xyz + normals)
- **Encoder**: ResBlocks (64->256->64) with LeakyReLU(0.2)
- **Graph Convolutions**: Two cycles of Facet<->Node message passing with attention pooling (8 heads), dropout 0.2 between cycles
- **Output Head**: `node_offset_head` (SharedMLP) predicts (dx, dy, dz) per vertex
- **Loss**: MSE between predicted and ground truth positions (`compute_denoising_loss` in losses.py)

### Loss Function (network/losses.py)
`compute_denoising_loss()` aggregates per-cell offsets to per-vertex offsets via `scatter_mean()`, filters out infinite cells (index == -1), then computes MSE between `noisy_pos + predicted_offsets` and `clean_pc`. Falls back to Chamfer distance if point counts differ.

### Key Files
- `network/train_pro.py` - Training entry point
- `network/test_pro.py` - Inference entry point
- `network/R_GCN_model.py` - Main model (R_GCN class)
- `network/DT_data.py` - Graph data structures (ScanData, DT_Data); denoising adds `clean_pc` and `pos` fields
- `network/losses.py` - Loss functions
- `network/cache_dataset.py` - Pre-loads all data into RAM, saves to `.pt` files in experiment_dir
- `network/train_cfg.yaml` - Training configuration
- `create_gt/meshing_denoise.cc` - C++ Delaunay triangulation engine

## Training Details

- **Optimizer**: Adam, lr=0.001
- **LR Scheduler**: MultiStepLR, decay at epochs [50, 100, 150], gamma=0.5
- **AMP**: Mixed precision training with `torch.cuda.amp.GradScaler` and `autocast()`
- **Speed flags**: `cudnn.benchmark=True`, `matmul.allow_tf32`, `cudnn.allow_tf32`
- **torch.compile()**: Disabled (commented out) — causes excessive recompilation with variable-shaped graph data
- **Checkpoints**: Best model saved on lowest val loss; extra checkpoints every 5 epochs and every 2000 steps
- **Validation**: Runs every 5 epochs (not every epoch)

## Configuration

Training config in `network/train_cfg.yaml`:
- `weight_ratio: [1.0, 0.0, 0.0]` - Loss weights [denoising_mse, loss2, loss3]; only denoising MSE is active
- `ref_num: 3` - Reference clean points per noisy point
- `k_n: 32` - KNN neighbors for normal estimation
- `device_ids: [1]` - **Must be adjusted for your GPU setup**
- `batch_size: 1` - Must be <= len(device_ids)
- `data_root_dir` / `val_data_root_dir` - Paths to processed data
- `experiment_dir` - Output directory for checkpoints, cache, and results

## Important Implementation Notes

1. **Normalization Alignment**: Both noisy and clean point clouds MUST be normalized using the same transform (calculated from noisy cloud and saved to `transform.txt`). This is critical for correct offset learning.

2. **Dummy Labels**: The data loader expects `ref_point_label.txt` files (legacy from classification). These contain dummy values; the denoising loss ignores them.

3. **Multi-GPU**: Uses custom `DTParallel` wrapper with `DataListLoader`. Requires `batch_size <= len(device_ids)`. Data scattered per GPU, loss averaged with `.mean()`.

4. **Caching**: `cache_dataset.py` loads all data into RAM and saves `cached_train_data.pt` / `cached_val_data.pt`. Training auto-detects and uses these files if present.
