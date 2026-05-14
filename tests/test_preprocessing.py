import numpy as np
import pytest
import cv2
from src.preprocessing import binarize_image

def test_binarize_image_returns_binary():
    # Create dummy grayscale image
    img = np.array([[50, 200], [150, 10]], dtype=np.uint8)
    binary = binarize_image(img)
    assert binary.shape == img.shape
    assert set(np.unique(binary)).issubset({0, 255})
