import numpy as np
from src.energy_map import create_cost_map

def test_create_cost_map_values():
    # Create a simple binary image with one white pixel (text)
    img = np.zeros((10, 10), dtype=np.uint8)
    img[5, 5] = 255 # One white pixel
    cost_map = create_cost_map(img)
    
    # Check if the pixel with "text" is significantly more expensive than empty background
    assert cost_map[5, 5] > cost_map[0, 0]
    
    # Check if pixels near text are more expensive than those far away (Distance Transform)
    # distance_transform_edt(img == 0) for [5,5]=255 will be 0 at [5,5]
    # dist at [5,4] is 1. dist at [0,0] is sqrt(5^2 + 5^2) = sqrt(50) ~= 7.07
    # dist_inv at [5,4] is 1/(1+0.5) = 1/1.5 = 0.66
    # dist_inv at [0,0] is 1/(7.07+0.5) = 1/7.57 ~= 0.13
    assert cost_map[5, 4] > cost_map[0, 0]
