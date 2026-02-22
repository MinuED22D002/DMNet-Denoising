#!/usr/bin/env python3
import torch
import os
import numpy as np
import DTU
import train_utils
import DT_data
from tqdm import tqdm
from scipy.spatial import cKDTree
import sys

# Add create_gt to path for imports
sys_path = os.path.abspath(os.path.join(os.getcwd(), "../create_gt"))
if sys_path not in sys.path:
    sys.path.append(sys_path)

def build_tet_adj_facet_single(file_path):
    """Helper to process a single directory (modified from data_process.py)"""
    file_names = dict()
    file_names['output_tetrahedron_adj'] = (os.path.join(file_path, "output_tetrahedron_adj"))
    file_names['output_facet_nei_cell'] = (os.path.join(file_path, "output_facet_nei_cell.txt"))

    tetrahedron_adj = np.fromfile(file_names['output_tetrahedron_adj'], dtype=np.int32, sep=' ').reshape(-1, 4)
    facet_nei_cell = np.fromfile(file_names['output_facet_nei_cell'], dtype=np.int32, sep=' ').reshape(-1, 2)
    facet_nei_cell = np.sort(facet_nei_cell, axis=1)
    facet_idx = dict()
    for i in range(facet_nei_cell.shape[0]):
        string = str(facet_nei_cell[i][0]) + "--" + str(facet_nei_cell[i][1])
        facet_idx[string] = i

    tet_idx_key = np.linspace(0, tetrahedron_adj.shape[0]-1, tetrahedron_adj.shape[0]).astype(np.int32).reshape(-1,1)
    tet_neighbor = dict()
    tet_neighbor[0] = np.sort(np.hstack((tet_idx_key, tetrahedron_adj[:,0].reshape(-1,1))), axis=1)
    tet_neighbor[1] = np.sort(np.hstack((tet_idx_key, tetrahedron_adj[:,1].reshape(-1,1))), axis=1)
    tet_neighbor[2] = np.sort(np.hstack((tet_idx_key, tetrahedron_adj[:,2].reshape(-1,1))), axis=1)
    tet_neighbor[3] = np.sort(np.hstack((tet_idx_key, tetrahedron_adj[:,3].reshape(-1,1))), axis=1)

    tet_adj_facet = (np.ones((tetrahedron_adj.shape[0], 4)) * (-1)).astype(np.int32)
    for j in range(4):
        tet = tet_neighbor[j]
        for i in range(tet.shape[0]):
            string = str(tet[i][0]) + "--" + str(tet[i][1])
            idx = facet_idx.get(string)
            if idx is None:
                tet_adj_facet[i][j] = -1
            else:
                tet_adj_facet[i][j] = idx
    np.savetxt(os.path.join(file_path, 'tet_adj_facet.txt'), tet_adj_facet, fmt='%d')

# Load config
cfg = train_utils.load_config('./train_cfg.yaml')
cfg = train_utils.augment_config(cfg)
cfg = train_utils.check_config(cfg)

PATCH_SIZE = 8192
NUM_PATCHES_PER_CLOUD = 8 # (8 * 8k = 64k, gives enough overlap for 50k)

def extract_and_cache_patches(dataset, name, cfg):
    cache_dir = os.path.join(cfg['experiment_dir'], f'cached_{name}_patches')
    os.makedirs(cache_dir, exist_ok=True)
    
    count = 0
    for i in range(len(dataset)):
        # Load full scan
        scan = dataset.load_scan(i)
        
        # Get point coordinates
        points = np.asarray(scan.pc.points)
        num_points = points.shape[0]
        
        # 1. FPS to find patch centers
        points_torch = torch.from_numpy(points).float()
        if torch.cuda.is_available():
            points_torch = points_torch.cuda()
            
        # We want NUM_PATCHES_PER_CLOUD centers
        center_indices = DT_data.farthest_point_sample(points_torch, NUM_PATCHES_PER_CLOUD)
        centers = points[center_indices.cpu().numpy()]
        
        # 2. KNN to get patch points
        kdtree = cKDTree(points)
        _, patch_indices = kdtree.query(centers, k=PATCH_SIZE) # (NUM_PATCHES, PATCH_SIZE)
        
        # 3. Process each patch
        for p_idx in range(NUM_PATCHES_PER_CLOUD):
            p_indices = patch_indices[p_idx]
            
            # Create a virtual scan for this patch
            patch_scan = DT_data.ScanData()
            patch_scan.data_para = scan.data_para
            patch_scan.scan_name = f"{scan.scan_name}_patch_{p_idx}"
            
            # Crop Point Cloud
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[p_indices])
            if scan.pc.has_normals():
                pcd.normals = o3d.utility.Vector3dVector(np.asarray(scan.pc.normals)[p_indices])
            patch_scan.pc = pcd
            
            # Clean points for loss (subset)
            if scan.surface_pc is not None:
                clean_points = np.asarray(scan.surface_pc.points)
                c_kdtree = cKDTree(clean_points)
                # For each point in patch, find closest in clean
                _, c_indices = c_kdtree.query(points[p_indices], k=1)
                
                c_pcd = o3d.geometry.PointCloud()
                c_pcd.points = o3d.utility.Vector3dVector(clean_points[c_indices])
                if scan.surface_pc.has_normals():
                    c_pcd.normals = o3d.utility.Vector3dVector(np.asarray(scan.surface_pc.normals)[c_indices])
                patch_scan.surface_pc = c_pcd
            
            # IGNORE: Traditionally we would run DT on patches here
            # But create_full_data handles DT and feature extraction.
            # We need to provide DT info if we want to avoid recomputing it inside create_full_data
            # HOWEVER, create_full_data is exactly what we want to avoid running every epoch.
            # So we perform the heavy lifting here once.
            
            # We need to simulate what denoise_data_process.py does but for patches.
            # For brevity and since we want to be "fast", we will let create_full_data 
            # do the DT for the 8k patch (which is fast) and then SAVE the result.
            
            # Note: create_full_data expects scan.cell_vertex_idx etc. 
            # If they are None, it will fail or try to load them if we are not careful.
            
            # Actually, looking at DT_data.py, create_full_data loads from scan.
            # We need to mock the Delaunay results for the patch.
            
            # Let's use a simpler approach: 
            # Just create the patch-based DT_Data object and save it.
            try:
                # We need to ensure CREATE_FULL_DATA doesn't try to load files from disk for patches
                # So we manually populate what's needed or adjust DT_data.py
                
                # Mocking the load_full_scan expected files is hard.
                # Better: Run a minimal DT on the patch here.
                # Since the user asked for patches to help, let's implement the patch-level DT.
                
                # For now, let's use the full processing pipeline on the patch
                # This is only done ONCE during caching.
                
                # We need a way to run the C++ Delaunay on these patches or use a python version.
                # The codebase uses a C++ tool.
                
                # Optimization: For now, if we can't easily run C++ on patches, 
                # we'll use scipy/torch Delaunay if available, or just skip DT for patches if the user 
                # is okay with just point-based denoising (unlikely given DT symmetry).
                
                # Let's assume we can run the C++ tool via os.system on temporary patch files.
                temp_patch_path = os.path.join(cache_dir, f"temp_{count}.ply")
                o3d.io.write_point_cloud(temp_patch_path, pcd)
                
                # Command to run C++ tool (igl_gt_denoise equivalent)
                # exe_file found in create_gt/build/igl_gt_denoise
                exe_file = os.path.join(os.getcwd(), "../create_gt/build/igl_gt_denoise")
                if not os.path.exists(exe_file):
                    # fallback to relative from network
                    exe_file = "../create_gt/build/igl_gt_denoise"
                
                patch_out_dir = os.path.join(cache_dir, f"patch_{count}/")
                os.makedirs(patch_out_dir, exist_ok=True)
                
                cmd = f"{exe_file} {temp_patch_path} {patch_out_dir} 10 {PATCH_SIZE} 0.0"
                os.system(cmd)
                
                # Run adjacency building for this patch
                build_tet_adj_facet_single(patch_out_dir)
                
                # Now load it back as a ScanData
                patch_scan_loaded = DT_data.ScanData()
                patch_scan_loaded.load_full_scan(patch_out_dir, cfg)
                patch_scan_loaded.scan_name = f"patch_{count}"
                
                # Create the final cached data
                patch_sample = DT_data.create_full_data(patch_scan_loaded, cfg)
                
                # Save as .pt
                sample_path = os.path.join(cache_dir, f'{count}.pt')
                torch.save(patch_sample, sample_path)
                count += 1
                
                # Cleanup temp
                os.remove(temp_patch_path)
                import shutil
                shutil.rmtree(patch_out_dir)
                
            except Exception as e:
                print(f"Failed to process patch {count}: {e}")
                continue

    # Save count
    with open(os.path.join(cache_dir, 'count.txt'), 'w') as f:
        f.write(str(count))
    print(f"Saved {count} patches to: {cache_dir}")

if __name__ == "__main__":
    print("="*60)
    print("CACHING PATCH-BASED DATASET (8k patches)")
    print("="*60)
    
    train_data = DTU.DTUDelDataset(cfg, "train")
    val_data = DTU.DTUDelDataset(cfg, "val")
    
    print("\nCaching Train Patches...")
    extract_and_cache_patches(train_data, "train", cfg)
    
    print("\nCaching Val Patches...")
    extract_and_cache_patches(val_data, "val", cfg)
    
    print("\nProcessing Complete!")
