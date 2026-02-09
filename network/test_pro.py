import argparse
import os
import collections

import torch
import numpy as np
import open3d as o3d

import train_utils
import DTU
from DT_data import *
from DT_dataloader import *
import R_GCN_model
from losses import scatter_mean

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--config', default="./test_cfg.yaml", type=str, help='Path to config file')
args = parser.parse_args()

cfg = train_utils.load_config(args.config)
cfg = train_utils.augment_config(cfg)
cfg = train_utils.check_config(cfg)
print(cfg)

if not os.path.exists(cfg["experiment_dir"]):
    os.makedirs(cfg["experiment_dir"])

geo_in = 6
test_model = R_GCN_model.R_GCN(geo_in)

model_path = cfg["model_path"]
device = torch.device("cpu")
if cfg["cuda"] and torch.cuda.is_available():
    device = torch.device("cuda:{}".format(cfg["device_ids"][0]))

# Load checkpoint, stripping 'module.' prefix from DTParallel-saved weights
state_dict = torch.load(model_path, map_location=device)
new_state_dict = collections.OrderedDict()
for k, v in state_dict.items():
    name = k.replace("module.", "") if k.startswith("module.") else k
    new_state_dict[name] = v
test_model.load_state_dict(new_state_dict)

test_model = test_model.to(device)
test_model.eval()

# Load test data
test_data = DTU.DTUDelDataset(cfg, "test")
test_data_loader = DataListLoader(test_data, batch_size=1, shuffle=False, num_workers=cfg["num_workers"])

print(f"Loaded {len(test_data)} test samples")

with torch.no_grad():
    for test_data_list in test_data_loader:
        node_offsets_per_cell, _, _, _ = test_model(test_data_list)

        # Assume batch_size=1
        data = test_data_list[0]

        offsets_flat = node_offsets_per_cell.reshape(-1, 3)
        indices_flat = data.cell_vertex_idx.reshape(-1).long().to(device)

        mask = indices_flat != -1
        offsets_flat = offsets_flat[mask]
        indices_flat = indices_flat[mask]

        num_points = data.pos.shape[0]
        vertex_offsets = scatter_mean(offsets_flat, indices_flat, num_points)

        denoised_pos = data.pos.to(device) + vertex_offsets
        denoised_np = denoised_pos.detach().cpu().numpy()

        output_dir = os.path.join(cfg["experiment_dir"], "output_denoised", data.data_name.split("/")[0])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save as PLY
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(denoised_np)
        o3d.io.write_point_cloud(os.path.join(output_dir, "denoised.ply"), pcd)

        print("test", data.data_name, "saved to", output_dir)

print("Done.")
