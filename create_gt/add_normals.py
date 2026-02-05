import os
import open3d as o3d
import argparse
import numpy as np

def add_normals_to_folder(root_dir):
    print(f"Scanning {root_dir}...")
    
    count = 0
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith("points.ply"): # sampled_points.ply or clean_points.ply
                file_path = os.path.join(subdir, file)
                try:
                    pcd = o3d.io.read_point_cloud(file_path)
                    
                    # Check if normals exist
                    if not pcd.has_normals():
                        print(f"Estimating normals for {file_path}")
                        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                        o3d.io.write_point_cloud(file_path, pcd)
                        count += 1
                    else:
                        # Optional: Force re-estimate if they look wrong (all zeros)
                        normals = np.asarray(pcd.normals)
                        if np.all(normals == 0):
                             print(f"Fixing zero normals for {file_path}")
                             pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                             o3d.io.write_point_cloud(file_path, pcd)
                             count += 1

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    print(f"Done. Updated {count} files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='Root directory of processed data (e.g. ../example/processed_data/)')
    args = parser.parse_args()
    
    add_normals_to_folder(args.dir)
