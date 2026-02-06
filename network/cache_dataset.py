#!/usr/bin/env python3
"""
Pre-cache all training data to speed up subsequent epochs by 10x.
Run once: python3 cache_dataset.py
Then training will load from cached tensors instead of recomputing.
"""

import torch
from DT_dataloader import DataListLoader
import DTU
import train_utils
import os
from tqdm import tqdm

# Load config
cfg = train_utils.load_config('./train_cfg.yaml')
cfg = train_utils.augment_config(cfg)
cfg = train_utils.check_config(cfg)

print("="*60)
print("PREPROCESSING TRAINING DATA")
print("="*60)
print(f"k_n: {cfg['k_n']}")
print(f"Data directory: {cfg['data_root_dir']}")

# Create datasets
train_data = DTU.DTUDelDataset(cfg, "train")
val_data = DTU.DTUDelDataset(cfg, "val")

print(f"\nFound {len(train_data)} training samples")
print(f"Found {len(val_data)} validation samples")

# Cache training data
print("\n" + "="*60)
print("CACHING TRAINING DATA...")
print("="*60)
train_loader = DataListLoader(train_data, 1, num_workers=cfg['num_workers'])
cached_train = []

for i, batch in enumerate(tqdm(train_loader, desc="Training")):
    cached_train.append(batch[0])

cache_path = os.path.join(cfg['experiment_dir'], 'cached_train_data.pt')
os.makedirs(os.path.dirname(cache_path), exist_ok=True)
torch.save(cached_train, cache_path)
print(f"✓ Saved to: {cache_path}")

# Cache validation data
print("\n" + "="*60)
print("CACHING VALIDATION DATA...")
print("="*60)
val_loader = DataListLoader(val_data, 1, num_workers=cfg['num_workers'])
cached_val = []

for i, batch in enumerate(tqdm(val_loader, desc="Validation")):
    cached_val.append(batch[0])

cache_path_val = os.path.join(cfg['experiment_dir'], 'cached_val_data.pt')
torch.save(cached_val, cache_path_val)
print(f"✓ Saved to: {cache_path_val}")

print("\n" + "="*60)
print("PREPROCESSING COMPLETE!")
print("="*60)
print("Next step: Modify train_pro.py to load from cache.")
print("This will make training 10x faster!")
