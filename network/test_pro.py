import argparse
import os
import torch
import numpy as np
import open3d as o3d

import train_utils
import DTU
from DT_data import *
from DT_dataloader import *
import R_GCN_model
from losses import scatter_mean

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('--config', default="./train_cfg.yaml", type=str, help='Path to config file')
parser.add_argument('--model_path', type=str, help='Path to trained model (optional, overrides config)')

args = parser.parse_args()
cfg = train_utils.load_config(args.config)
cfg = train_utils.augment_config(cfg)
cfg = train_utils.check_config(cfg)

print(cfg)

# Load model
geo_in = 6
test_model = R_GCN_model.R_GCN(geo_in)

# Use specified model path or default from config
model_path = args.model_path if args.model_path else cfg["model_path"]

if cfg["cuda"]:
    device = torch.device("cuda:{}".format(cfg["device_ids"][0]))
    test_model = test_model.to(device)
else:
    device = torch.device("cpu")

# Load trained weights
if os.path.exists(model_path):
    test_model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from: {model_path}")
else:
    print(f"ERROR: Model not found at {model_path}")
    exit(1)

# Load test data
test_data = DTU.DTUDelDataset(cfg, "test")
test_data_loader = DataListLoader(test_data, 1, num_workers=cfg["num_workers"])

print(f"Testing on {len(test_data)} samples")

# Create output directory
output_base_dir = os.path.join(cfg["experiment_dir"], "output_denoised")
os.makedirs(output_base_dir, exist_ok=True)

# Run inference
test_model.eval()
with torch.no_grad():
    for i, test_data_list in enumerate(test_data_loader):
        # Forward pass
        node_offsets_per_cell, _, _, _ = test_model(test_data_list)
        
        # Assume batch_size=1
        data = test_data_list[0]
        
        # Aggregate cell offsets to vertex offsets
        offsets_flat = node_offsets_per_cell.reshape(-1, 3)
        indices_flat = data.cell_vertex_idx.reshape(-1).long().to(device)
        
        # Filter out invalid indices (-1)
        mask = indices_flat != -1
        offsets_flat = offsets_flat[mask]
        indices_flat = indices_flat[mask]
        
        # Scatter mean to get per-vertex offsets
        num_points = data.pos.shape[0]
        vertex_offsets = scatter_mean(offsets_flat, indices_flat, num_points)
        
        # Apply offsets to get denoised positions
        denoised_pos = data.pos.to(device) + vertex_offsets
        denoised_np = denoised_pos.detach().cpu().numpy()
        
        # Also get noisy input for comparison
        noisy_np = data.pos.cpu().numpy()
        
        # Create output directory for this sample
        sample_name = data.data_name.split("/")[0]
        output_dir = os.path.join(output_base_dir, sample_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save denoised point cloud
        pcd_denoised = o3d.geometry.PointCloud()
        pcd_denoised.points = o3d.utility.Vector3dVector(denoised_np)
        o3d.io.write_point_cloud(os.path.join(output_dir, "denoised.ply"), pcd_denoised)
        
        # Save noisy input for comparison
        pcd_noisy = o3d.geometry.PointCloud()
        pcd_noisy.points = o3d.utility.Vector3dVector(noisy_np)
        o3d.io.write_point_cloud(os.path.join(output_dir, "noisy.ply"), pcd_noisy)
        
        # If ground truth exists, save it too
        if hasattr(data, 'target_pos') and data.target_pos is not None:
            clean_np = data.target_pos.cpu().numpy()
            pcd_clean = o3d.geometry.PointCloud()
            pcd_clean.points = o3d.utility.Vector3dVector(clean_np)
            o3d.io.write_point_cloud(os.path.join(output_dir, "ground_truth.ply"), pcd_clean)
            
            # Compute error metrics
            mse = np.mean((denoised_np - clean_np) ** 2)
            rmse = np.sqrt(mse)
            print(f"[{i+1}/{len(test_data)}] {sample_name} - RMSE: {rmse:.6f} - Saved to {output_dir}")
        else:
            print(f"[{i+1}/{len(test_data)}] {sample_name} - Saved to {output_dir}")

print(f"\nTesting complete! Results saved to: {output_base_dir}")
