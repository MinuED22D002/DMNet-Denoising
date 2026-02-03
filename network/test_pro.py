import open3d as o3d
from losses import scatter_mean

# ... (existing imports) ...

# ... (inside loop) ...
    train_model.eval()
    with torch.no_grad():
        node_offsets_per_cell, _, _, _ = train_model(test_data_list)
        
        # Assume batch_size=1
        data = test_data_list[0]
        device = node_offsets_per_cell.device
        
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




















