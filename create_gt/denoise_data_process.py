import os
import shutil
import sys
import argparse

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
    # Create a simple tetrahedron OFF file
    content = """OFF
4 4 0
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
    exe_file = os.path.join(script_dir, "build/igl_gt")
    
    if not os.path.exists(exe_file):
        print(f"Error: Could not find igl_gt at {exe_file}")
        # Try finding it relative to current dir
        if os.path.exists("./build/igl_gt"):
            exe_file = "./build/igl_gt"
        else:
            return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Handle Dummy Mesh
    if not dummy_mesh_path or not os.path.exists(dummy_mesh_path):
        dummy_mesh_path = os.path.join(output_dir, "dummy.off")
        create_dummy_mesh(dummy_mesh_path)

    # Support both .ply and .xyz
    files = [f for f in os.listdir(noisy_dir) if f.endswith('.ply') or f.endswith('.xyz')]
    if not files:
        print(f"No .ply or .xyz files found in {noisy_dir}")
        return

    print(f"Found {len(files)} files to process.")
    
    import open3d as o3d
    # ... (rest of imports and helper functions same as before)
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
        # We pick the longest clean filename that matches to avoid ambiguity (e.g. 'chair' vs 'chair_arm')
        noisy_stem = os.path.splitext(noisy_filename)[0]
        
        candidates = []
        for cf in clean_file_list:
            clean_stem = os.path.splitext(cf)[0]
            if clean_stem in noisy_stem:
                candidates.append(cf)
        
        if not candidates:
            return None
            
        # Return the one with the longest stem (most specific match)
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
        # Use noisy filename as the unique identifier for output folder
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
        
        # 3. Run igl_gt
        # Usage: exe_file point_path surface_path output_dir sample_depth point_num noise_level ref_num
        dest_out_dir_slash = scan_out_dir + "/"
        cmd = "%s %s %s %s 10 10000 0.0 3" % (exe_file, dest_noisy, dummy_mesh_path, dest_out_dir_slash)
        print(f"Running igl_gt for {scan_name}...")
        ret = os.system(cmd)
        if ret != 0:
            print(f"Error running igl_gt for {f}. Return code {ret}")
            
    # 4. Build Adjacency
    print("Building Tetrahedral Adjacency...")
    try:
        build_tet_adj_facet(output_dir)
    except Exception as e:
        print(f"Error in data_process: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--noisy', type=str, required=True, help='Directory with noisy point clouds')
    parser.add_argument('--clean', type=str, required=True, help='Directory with clean point clouds')
    parser.add_argument('--out', type=str, required=True, help='Output directory for processed data')
    parser.add_argument('--dummy_mesh', type=str, required=False, help='Path to any valid mesh file (optional, will auto-create if missing)')
    
    args = parser.parse_args()
    
    process_denoising_data(args.noisy, args.clean, args.out, args.dummy_mesh)
