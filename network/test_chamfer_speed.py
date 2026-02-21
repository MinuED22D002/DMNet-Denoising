import torch
import time
from losses import chamfer_distance as chamfer_chunked

def chamfer_naive(p1, p2):
    x = p1.unsqueeze(2)
    y = p2.unsqueeze(1)
    dist = torch.norm(x - y, dim=-1)
    min_dist_x = torch.min(dist, dim=2)[0]
    min_dist_y = torch.min(dist, dim=1)[0]
    return torch.mean(min_dist_x) + torch.mean(min_dist_y)

def benchmark():
    N = 10000 # Using 10k for baseline comparison (50k naive would OOM on most GPUs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on {device} with {N} points...")
    
    p1 = torch.rand(1, N, 3, device=device)
    p2 = torch.rand(1, N, 3, device=device)
    
    # Naive
    start = time.time()
    try:
        res_naive = chamfer_naive(p1, p2)
        end = time.time()
        print(f"Naive Chamfer: {end - start:.4f}s, Result: {res_naive.item():.6f}")
    except RuntimeError as e:
        print(f"Naive Chamfer failed: {e}")

    # Chunked
    start = time.time()
    res_chunked = chamfer_chunked(p1, p2, chunk_size=1000)
    end = time.time()
    print(f"Chunked Chamfer: {end - start:.4f}s, Result: {res_chunked.item():.6f}")

if __name__ == "__main__":
    benchmark()
