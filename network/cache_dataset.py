#!/usr/bin/env python3
"""
Pre-cache all training data to speed up subsequent epochs.
Run once: python3 cache_dataset.py
Then training will load from cached tensors instead of recomputing.

Saves each sample as an individual .pt file to avoid RAM exhaustion.
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

def cache_dataset(dataset, name, cfg):
    cache_dir = os.path.join(cfg['experiment_dir'], f'cached_{name}')
    os.makedirs(cache_dir, exist_ok=True)

    loader = DataListLoader(dataset, 1, num_workers=min(cfg['num_workers'], 4))
    count = 0

    for i, batch in enumerate(tqdm(loader, desc=name)):
        sample_path = os.path.join(cache_dir, f'{i}.pt')
        torch.save(batch[0], sample_path)
        count += 1

    # Save count for the loader to know how many files exist
    with open(os.path.join(cache_dir, 'count.txt'), 'w') as f:
        f.write(str(count))

    print(f"Saved {count} samples to: {cache_dir}")
    return count

# Cache training data
print("\n" + "="*60)
print("CACHING TRAINING DATA...")
print("="*60)
train_count = cache_dataset(train_data, "train", cfg)

# Cache validation data
print("\n" + "="*60)
print("CACHING VALIDATION DATA...")
print("="*60)
val_count = cache_dataset(val_data, "val", cfg)

print("\n" + "="*60)
print("PREPROCESSING COMPLETE!")
print("="*60)
print(f"Cached {train_count} train + {val_count} val samples as individual files.")
#print(f"Cached {train_count} train + {val_count} val samples as individual file#s.")
