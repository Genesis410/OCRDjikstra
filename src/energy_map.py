import numpy as np
from scipy.ndimage import distance_transform_edt
from numba import njit

@njit
def apply_penalties(intensity, dist_inv, alpha, beta):
    """
    Combine intensity and distance inverse to create the cost map.
    Optimized with Numba.
    """
    return alpha * intensity + beta * dist_inv

def create_cost_map(binary_img, alpha=1000, beta=500):
    """
    Creates a cost map where text and its vicinity have higher cost.
    
    Args:
        binary_img: Binary image (255 for text, 0 for background)
        alpha: Penalty weight for being exactly on a text pixel
        beta: Penalty weight for being near a text pixel
        
    Returns:
        cost_map: 2D array of costs
    """
    # distance_transform_edt calculates distance to the closest zero-valued pixel.
    # To find distance to text (255), we find distance to pixels where binary_img is NOT 0.
    # So we pass (binary_img == 0) to distance_transform_edt. 
    # Actually, distance_transform_edt(input) computes distance to zeros.
    # If we want distance to 255s, we should pass an array where 255s are 0s.
    # So (binary_img == 0) is correct because it's True (1) for background and False (0) for text.
    dist = distance_transform_edt(binary_img == 0)
    
    # Invert distance so that pixels closer to text have higher cost.
    # Add small epsilon to avoid division by zero.
    dist_inv = 1.0 / (dist + 0.5)
    
    # Normalize intensity to [0, 1]
    intensity = binary_img.astype(np.float32) / 255.0
    
    # Call the JIT-optimized function
    cost_map = apply_penalties(intensity, dist_inv, alpha, beta)
    
    return cost_map
