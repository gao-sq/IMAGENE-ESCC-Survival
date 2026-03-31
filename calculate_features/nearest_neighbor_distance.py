import cupy as cp
import numpy as np
from typing import Tuple

def analyze_nearest_neighbor_distance(
    points: np.ndarray, 
    types: np.ndarray, 
    type_num: int, 
    distance_threshold: float,
    chunk_size: int = 2048
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated nearest neighbor analysis function
    """
    # Convert data to GPU (using mixed precision)
    points_gpu = cp.asarray(points, dtype=cp.float32)
    types_gpu = cp.asarray(types, dtype=cp.int32) - 1  # Types start from 1, need to subtract 1
    
    n = len(points_gpu)
    neighbor_dist = cp.full((n, type_num), cp.finfo(cp.float32).max, dtype=cp.float32)
    neighbor_count = cp.zeros((n, type_num), dtype=cp.int32)

    # Pre-allocate GPU memory for chunk calculation
    block_points = cp.empty((chunk_size, 2), dtype=cp.float32)
    
    for target_type in range(type_num):
        # Get all points of target type
        type_mask = types_gpu == target_type
        type_indices = cp.where(type_mask)[0]
        type_points = points_gpu[type_indices]
        m = len(type_points)
        
        if m == 0:
            continue

        # Process target point set in chunks
        for chunk_start in range(0, m, chunk_size):
            chunk_end = min(chunk_start + chunk_size, m)
            chunk = type_points[chunk_start:chunk_end]
            chunk_len = chunk_end - chunk_start
            
            # Reuse pre-allocated memory
            block_points[:chunk_len] = chunk
            
            # Batch calculate distance matrix (n, chunk_len)
            # Use Einstein summation for efficient distance calculation
            diff = points_gpu[:, None, :] - block_points[None, :chunk_len, :]
            dist_matrix = cp.sqrt(cp.einsum('ijk,ijk->ij', diff, diff)) # (n, chunk_len)

            # Handle same-type point exclusion
            # if target_type == types_gpu[0]:
            same_type_mask = type_mask[:, None] & (type_indices[chunk_start:chunk_end] == cp.arange(n)[:, None])
            dist_matrix[same_type_mask] = cp.finfo(cp.float32).max

            # Update nearest distance
            chunk_min = cp.min(dist_matrix, axis=1) # (n,)
            neighbor_dist[:, target_type] = cp.minimum(neighbor_dist[:, target_type], chunk_min)
            
            # Count neighbors
            neighbor_count[:, target_type] += cp.sum(dist_matrix <= distance_threshold, axis=1)

    # Post-processing: convert infinity to -1 for easier subsequent processing
    neighbor_dist[neighbor_dist == cp.finfo(cp.float32).max] = -1
    
    return neighbor_dist.get(), neighbor_count.get()


    

    
