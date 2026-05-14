import numpy as np
import pytest
from src.pathfinding import find_seam

def test_dijkstra_finds_path():
    # 10x10 cost map, mostly low cost (1.0)
    cost_map = np.ones((10, 10), dtype=np.float32)
    # Create a high-cost barrier at column 5
    cost_map[:, 5] = 100.0
    # But leave a gap at (5, 5)
    cost_map[5, 5] = 1.0
    
    # Run pathfinding starting from middle left
    path = find_seam(cost_map, start_y=5)
    
    # Path should have length equal to width
    assert len(path) == 10
    # Path should go through the gap at (5, 5)
    # path is list of (y, x)
    assert (5, 5) in path

def test_vertical_penalty_affects_path():
    # 3x3 cost map
    # [1, 10, 1]
    # [1, 1, 1]
    # [1, 10, 1]
    cost_map = np.array([
        [1, 10, 1],
        [1, 1, 1],
        [1, 10, 1]
    ], dtype=np.float32)
    
    # Starting at (0, 0), it wants to get to column 2.
    # Route A (straight): (0,0) -> (0,1) -> (0,2). Cost = 1 + 10 + 1 = 12.
    # Route B (diagonal): (0,0) -> (1,1) -> (0,2). Cost = 1 + (1 + penalty) + (1 + penalty) = 3 + 2*penalty.
    
    # If penalty = 2, Route B cost = 3 + 4 = 7. Route B is better.
    path_low_penalty = find_seam(cost_map, start_y=0, vertical_penalty=2.0)
    assert (1, 1) in path_low_penalty
    
    # If penalty = 10, Route B cost = 3 + 20 = 23. Route A is better.
    path_high_penalty = find_seam(cost_map, start_y=0, vertical_penalty=10.0)
    assert (0, 1) in path_high_penalty

def test_pathfinding_out_of_bounds_start():
    cost_map = np.ones((10, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        find_seam(cost_map, start_y=15)

def test_pathfinding_empty_map():
    cost_map = np.zeros((0, 0), dtype=np.float32)
    with pytest.raises(ValueError):
        find_seam(cost_map, start_y=0)
