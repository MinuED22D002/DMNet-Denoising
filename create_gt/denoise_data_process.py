import os
import shutil
import sys
import argparse
import numpy as np
# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data_process import build_tet_adj_facet
except ImportError as e:
    print(f"Error importing data_process: {e}")
    print("Make sure you are in create_gt folder or it's in python path")
    # check if trimesh is installed
    try:
        import trimesh
    except ImportError:
        print("Missing dependency: trimesh. Please install with `pip install trimesh`")
    sys.exit(1)

def create_dummy_mesh(path):
    # Create a simple tetrahedron PLY file
    content = """ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
element face 4
property list uchar int vertex_indices
end_header
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
3 0 1 3
3 1 2 3
3 2 0 3
3 0 2 1
"""
    with open(path, 'w') as f:
        f.write(content)
    print(f"Created dummy mesh at {path}")

def process_denoising_data(noisy_dir, clean_dir, output_dir, dummy_mesh_path=None):
    """
    Process pairs of noisy and clean point clouds.
    """
    
    # Locate executable
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exe_file = os.path.join(script_dir, "build/igl_gt_denoise")
    
    if not os.path.exists(exe_file):
        print(f"Error: Could not find igl_gt_denoise at {exe_file}")
        # Try finding it relative to current dir
        if os.path.exists("./build/igl_gt_denoise"):
            exe_file = "./build/igl_gt_denoise"
        else:
            print("Please compile the C++ tools first: cd create_gt/build && cmake .. && make")
            return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Support both .ply and .xyz
    files = [f for f in os.listdir(noisy_dir) if f.endswith('.ply') or f.endswith('.xyz')]
    if not files:
        print(f"No .ply or .xyz files found in {noisy_dir}")
        return

    print(f"Found {len(files)} files to process.")
    
    import open3d as o3d

    def convert_and_save(src_path, dest_path):
        if src_path.endswith('.xyz'):
             pcd = o3d.io.read_point_cloud(src_path)
             o3d.io.write_point_cloud(dest_path, pcd)
        else:
             shutil.copy(src_path, dest_path)
    
    # Get List of Clean Files
    clean_files = [f for f in os.listdir(clean_dir) if f.endswith('.ply') or f.endswith('.xyz')]
    if not clean_files:
        print(f"No clean point clouds found in {clean_dir}")
        return

    def find_clean_match(noisy_filename, clean_file_list):
        # Heuristic: The clean filename (without extension) should be a substring of noisy filename
        noisy_stem = os.path.splitext(noisy_filename)[0]
        
        candidates = []
        for cf in clean_file_list:
            clean_stem = os.path.splitext(cf)[0]
            if clean_stem in noisy_stem:
                candidates.append(cf)
        
        if not candidates:
            return None
            
        candidates.sort(key=lambda x: len(os.path.splitext(x)[0]), reverse=True)
        return candidates[0]

    for f in files:
        noisy_path = os.path.join(noisy_dir, f)
        
        # Find corresponding clean file
        clean_filename = find_clean_match(f, clean_files)
        
        if not clean_filename:
            print(f"Warning: Could not find matching clean file for {f}, skipping.")
            continue
            
        clean_path = os.path.join(clean_dir, clean_filename)
        print(f"Mapping Noisy: {f} -> Clean: {clean_filename}")
            
        # Create output subfolder
        scan_name = f.split('.')[0]
        scan_out_dir = os.path.join(output_dir, scan_name)
        if not os.path.exists(scan_out_dir):
            os.makedirs(scan_out_dir)
            
        # Copy/Convert files
        # 1. Noisy -> sampled_points.ply
        dest_noisy = os.path.join(scan_out_dir, "sampled_points.ply")
        if not os.path.exists(dest_noisy):
            convert_and_save(noisy_path, dest_noisy)
        
        # 2. Clean -> clean_points.ply
        dest_clean = os.path.join(scan_out_dir, "clean_points.ply")
        if not os.path.exists(dest_clean):
             convert_and_save(clean_path, dest_clean)
        
        # 3. Run igl_gt_denoise
        # Usage: exe_file point_path output_dir sample_depth point_num noise_level
        dest_out_dir_slash = scan_out_dir + "/"
        cmd = "%s %s %s 10 10000 0.0" % (exe_file, dest_noisy, dest_out_dir_slash)
        print(f"Running igl_gt_denoise for {scan_name}...")
        ret = os.system(cmd)
        if ret != 0:
            print(f"Error running igl_gt_denoise for {f}. Return code {ret}")
            continue

        # 3.5 Normalize clean_points.ply
        transform_path = os.path.join(scan_out_dir, "transform.txt")
        if os.path.exists(transform_path):
             try:
                 with open(transform_path, 'r') as tf:
                     lines = tf.read().strip().split()
                     vals = [float(x) for x in lines]
                     
                 if len(vals) >= 4:
                     origin = np.array(vals[:3])
                     scale = vals[3]
                     
                     pcd_clean = o3d.io.read_point_cloud(dest_clean)
                     pts = np.asarray(pcd_clean.points)
                     pts_norm = (pts - origin) / scale
                     pcd_clean.points = o3d.utility.Vector3dVector(pts_norm)
                     o3d.io.write_point_cloud(dest_clean, pcd_clean)
                     print(f"Normalized {dest_clean}")
             except Exception as e:
                 print(f"Failed to normalize clean points: {e}")
            
    # 4. Build Adjacency
    print("Building Tetrahedral Adjacency...")
    try:
        build_tt_adj_facet(output_dir)
    except Exception as e:
        print(f"Error in data_process: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--noisy', type=str, required=True, help='Directory with noisy point clouds')
    parser.add_argument('--clean', type=str, required=True, help='Directory with clean point clouds')
    parser.add_argument('--out', type=str, required=True, help='Output directory for processed data')
    
    args = parser.parse_args()
    
    process_denoising_data(args.noisy, args.clean, args.out)
