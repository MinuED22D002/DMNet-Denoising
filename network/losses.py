import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def soft_cross_entropy(pred, soft_targets, reduction='mean'):

    entropy = torch.sum(- soft_targets * F.log_softmax(pred, dim=1), 1)
    if reduction == 'mean':
        return torch.mean(entropy)
    elif reduction == 'sum':
        return torch.sum(entropy)
    elif reduction == 'none':
        return entropy


def cal_loss_neighbor_consistency(pred, target):

    entropies = []
    for i in range(target.shape[1]):
        tmp_target = target[:, i, :]
        entropies.append(soft_cross_entropy(pred, tmp_target, reduction='none'))
    loss = torch.stack(entropies, dim=1)
    loss = torch.mean(loss, dim=1)
    return loss


def cal_loss_multilabel(c_pred, p_target):

    entropies = []
    for i in range(p_target.shape[1]):
        tmp_p_target = p_target[:, i]
        entropies.append(F.cross_entropy(c_pred, tmp_p_target, reduction='none'))
    loss = torch.stack(entropies, dim=1)
    loss = torch.mean(loss, 1)
    return loss


def dist_surface_to_triangle(surf_samples, gen_face_probs, tri_dists, sorted_inds):
    char_len = torch.norm((surf_samples - torch.mean(surf_samples, dim=0, keepdim=True)), dim=-1).mean()
    print("char_len", char_len)
    sorted_probs = gen_face_probs[sorted_inds]
    prob_none_closer = torch.cat((
            torch.ones_like(sorted_probs)[:, :1],
            torch.cumprod(1. - sorted_probs, dim=-1)[:, :-1]
        ), dim=-1)
    prob_is_closest = prob_none_closer * sorted_probs

    last_prob = 1.0 - torch.sum(prob_is_closest, dim=-1)
    last_dist = char_len * torch.ones(tri_dists.shape[0], dtype=tri_dists.dtype, device=tri_dists.device)
    prob_is_closest = torch.cat((prob_is_closest, last_prob.unsqueeze(-1)), dim=-1)
    prob_is_closest = torch.clamp(prob_is_closest, 0., 1.)
    tri_dists = torch.cat((tri_dists, last_dist.unsqueeze(-1)), dim=-1)
    expected_dist = torch.sum(prob_is_closest * tri_dists, dim=-1)
    result = torch.mean(expected_dist / char_len)
    print("expected_dist max", torch.max(expected_dist / char_len, dim = -1)[0])
    return result

def limit_triangle_length(gen_face_probs, triangle_length_max):
    triangle_length_max = triangle_length_max / torch.max(triangle_length_max, dim=0)[0]
    length_average = torch.mean(triangle_length_max)
    triangle_length_max = triangle_length_max / length_average
    gen_face_probs = F.softmax(gen_face_probs, dim=0)
    result = torch.sum(gen_face_probs * triangle_length_max)
    return result


def tet_to_tri_pro_1(cell_pred_soft, facet_nei_cell):
    facet_nei_cell_pro = cell_pred_soft[facet_nei_cell]
    facet_pro = facet_nei_cell_pro[:,0,0] * facet_nei_cell_pro[:,1,1] + facet_nei_cell_pro[:,0,1] * facet_nei_cell_pro[:,1,0]
    pi = torch.tensor(np.pi)
    facet_pro = (1-torch.cos(pi * facet_pro)) / 2.0
    return facet_pro


def tet_to_tri_pro_2(cell_pred_soft, facet_nei_cell):
    facet_nei_cell_pro = cell_pred_soft[facet_nei_cell]
    pro = torch.abs(facet_nei_cell_pro[:,0,0]-0.5) + torch.abs(facet_nei_cell_pro[:,1,0]-0.5)
    facet_pro = torch.pow(torch.abs(facet_nei_cell_pro[:,0,0] - facet_nei_cell_pro[:,1,0]), 2) / pro
    return facet_pro


def build_losses(cell_pred, cell_pred_soft, deepdt_data):
    loss1 = torch.mean(cal_loss_multilabel(cell_pred, deepdt_data.ref_label))
    c_c_target = cell_pred_soft[deepdt_data.adj_idx]
    loss2 = torch.mean(cal_loss_neighbor_consistency(cell_pred, c_c_target))
    gen_face_probs = tet_to_tri_pro_1(cell_pred_soft, deepdt_data.facet_nei_cell)
    loss3 = torch.mean(limit_triangle_length(gen_face_probs, deepdt_data.facet_length_max))

    return loss1, loss2, loss3


def scatter_mean(src, index, dim_size):
    # input: src (E, C), index (E), dim_size (N)
    # output: (N, C)
    out = torch.zeros(dim_size, src.shape[1], device=src.device, dtype=src.dtype)
    count = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    
    out.index_add_(0, index, src)
    count.index_add_(0, index, torch.ones_like(src[:, :1]))
    
    return out / (count + 1e-6)


def chamfer_distance(p1, p2):
    """
    p1: (B, N, 3)
    p2: (B, M, 3)
    """
    x = p1.unsqueeze(2) # (B, N, 1, 3)
    y = p2.unsqueeze(1) # (B, 1, M, 3)
    
    # We can't do full matrix for large point clouds, split?
    # For now assuming batch size 1 and tolerable point count
    dist = torch.norm(x - y, dim=-1) # (B, N, M)
    
    min_dist_x = torch.min(dist, dim=2)[0] # (B, N)
    min_dist_y = torch.min(dist, dim=1)[0] # (B, M)
    
    return torch.mean(min_dist_x) + torch.mean(min_dist_y)


def compute_denoising_loss(node_offsets_per_cell, deepdt_data):
    # node_offsets_per_cell: (N_cells, 4, 3)
    # deepdt_data.pos: (N_points, 3)
    # deepdt_data.clean_pc: (N_clean_points, 3)

    if deepdt_data.clean_pc is None:
        return torch.tensor(0.0, device=node_offsets_per_cell.device)

    offsets_flat = node_offsets_per_cell.reshape(-1, 3)
    indices_flat = deepdt_data.cell_vertex_idx.reshape(-1).long()
    
    # Filter out -1 indices (infinite cells)
    mask = indices_flat != -1
    offsets_flat = offsets_flat[mask]
    indices_flat = indices_flat[mask]
    
    num_points = deepdt_data.pos.shape[0]
    
    # Aggregate offsets to vertices
    vertex_offsets = scatter_mean(offsets_flat, indices_flat, num_points)
    
    # Denoised positions
    denoised_pos = deepdt_data.pos + vertex_offsets
    
    clean_pos = deepdt_data.clean_pc
    
    # Check if number of points match
    if clean_pos.shape[0] == num_points:
         loss = F.mse_loss(denoised_pos, clean_pos)
    else:
         # Use batch dimension for chamfer
         loss = chamfer_distance(denoised_pos.unsqueeze(0), clean_pos.unsqueeze(0))

    return loss

