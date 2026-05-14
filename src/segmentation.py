import numpy as np
from typing import List, Tuple

def extract_ribbons(image: np.ndarray, seams: List[List[Tuple[int, int]]]) -> List[np.ndarray]:
    """
    Extract horizontal 'ribbons' of the image between consecutive seams.
    
    This function takes an image and a list of seams (paths found across the image)
    and extracts the rectangular regions that bound the space between each pair
    of consecutive seams.
    
    Args:
        image: 2D or 3D numpy array representing the image
        seams: List of paths, where each path is a List of (y, x) coordinates
        
    Returns:
        ribbons: List of image crops (numpy arrays)
    """
    ribbons: List[np.ndarray] = []
    if len(seams) < 2:
        return ribbons
        
    for i in range(len(seams) - 1):
        upper_seam = seams[i]
        lower_seam = seams[i+1]
        
        # Determine the vertical bounds for this ribbon based on seam envelopes
        y_min = min(p[0] for p in upper_seam)
        y_max = max(p[0] for p in lower_seam)
        
        # Ensure bounds are within image dimensions
        y_min = max(0, y_min)
        y_max = min(image.shape[0], y_max)
        
        if y_max > y_min:
            # We crop the full width of the image as the seams span from left to right
            ribbons.append(image[y_min:y_max, :])
            
    return ribbons
