import numpy as np
import pytest
from src.segmentation import extract_ribbons

def test_extract_ribbons_returns_correct_count():
    # Simple image 100x100
    img = np.zeros((100, 100), dtype=np.uint8)
    # Define 3 horizontal seams (as lists of (y, x))
    seam1 = [(10, x) for x in range(100)]
    seam2 = [(40, x) for x in range(100)]
    seam3 = [(70, x) for x in range(100)]
    
    ribbons = extract_ribbons(img, [seam1, seam2, seam3])
    # Between 3 seams there are 2 ribbons
    assert len(ribbons) == 2
    
    # Check shape of first ribbon (40 - 10 = 30x100)
    assert ribbons[0].shape == (30, 100)
    # Check shape of second ribbon (70 - 40 = 30x100)
    assert ribbons[1].shape == (30, 100)

def test_extract_ribbons_with_empty_seams():
    img = np.zeros((100, 100), dtype=np.uint8)
    assert extract_ribbons(img, []) == []
    assert extract_ribbons(img, [[(10, 0)]]) == []

def test_extract_ribbons_bounds_clipping():
    # Image 100x100
    img = np.zeros((100, 100), dtype=np.uint8)
    # Seams that go out of bounds
    seam1 = [(-10, x) for x in range(100)]
    seam2 = [(110, x) for x in range(100)]
    
    ribbons = extract_ribbons(img, [seam1, seam2])
    assert len(ribbons) == 1
    # Should be clipped to (0, 100)
    assert ribbons[0].shape == (100, 100)
