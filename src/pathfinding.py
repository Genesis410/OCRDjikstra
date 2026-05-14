import heapq
import numpy as np
from typing import List, Tuple, Dict, Optional

def find_seam(cost_map: np.ndarray, start_y: int, vertical_penalty: float = 10.0) -> List[Tuple[int, int]]:
    """
    Finds a low-cost path from left to right using Dijkstra's algorithm.
    
    Args:
        cost_map: 2D array of costs (h, w)
        start_y: Starting row in the first column (x=0)
        vertical_penalty: Extra cost for moving up or down between columns
        
    Returns:
        path: List of (y, x) coordinates representing the path
        
    Raises:
        ValueError: If start_y is out of bounds or cost_map is invalid
    """
    if cost_map.size == 0:
        raise ValueError("Cost map is empty")
    
    h, w = cost_map.shape
    
    if not (0 <= start_y < h):
        raise ValueError(f"Starting position y={start_y} is out of bounds for height {h}")

    # dist[y, x] to store the minimum cost to reach pixel (y, x)
    dist = np.full((h, w), np.inf, dtype=np.float32)
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
    
    # Priority Queue stores (cumulative_cost, current_y, current_x)
    pq: List[Tuple[float, int, int]] = [(float(cost_map[start_y, 0]), start_y, 0)]
    dist[start_y, 0] = cost_map[start_y, 0]
    parent[(start_y, 0)] = None
    
    target_node: Optional[Tuple[int, int]] = None
    
    while pq:
        curr_cost, y, x = heapq.heappop(pq)
        
        if x == w - 1:
            target_node = (y, x)
            break
            
        if curr_cost > dist[y, x]:
            continue
            
        # Explore neighbors in the next column (x + 1)
        # Allowed moves: Up-Right, Right, Down-Right
        for dy in [-1, 0, 1]:
            ny, nx = y + dy, x + 1
            if 0 <= ny < h:
                # Total weight = pixel cost + vertical penalty
                weight = float(cost_map[ny, nx]) + abs(dy) * vertical_penalty
                new_dist = curr_cost + weight
                
                if new_dist < dist[ny, nx]:
                    dist[ny, nx] = new_dist
                    parent[(ny, nx)] = (y, x)
                    heapq.heappush(pq, (new_dist, ny, nx))
                    
    # Backtrack to find path
    path: List[Tuple[int, int]] = []
    if target_node:
        curr = target_node
        while curr is not None:
            path.append(curr)
            curr = parent.get(curr)
        path.reverse()
        
    return path
