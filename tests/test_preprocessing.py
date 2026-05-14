import numpy as np
import pytest
import cv2
from src.preprocessing import binarize_image, deskew_image

def test_binarize_image_returns_binary():
    # Create dummy grayscale image
    img = np.array([[50, 200], [150, 10]], dtype=np.uint8)
    binary = binarize_image(img)
    assert binary.shape == img.shape
    assert set(np.unique(binary)).issubset({0, 255})

def test_deskew_image_rotates_correctly():
    # Create a blank image and draw a horizontal line
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.line(img, (10, 20), (90, 20), 255, 1) # Horizontal line at y=20
    
    # Rotate it 10 degrees clockwise
    center = (50, 50)
    angle_to_rotate = 10
    M = cv2.getRotationMatrix2D(center, angle_to_rotate, 1.0)
    rotated = cv2.warpAffine(img, M, (100, 100))
    
    # The image is now rotated. deskew_image should detect and fix it.
    deskewed, angle = deskew_image(rotated)
    
    # Check if detected angle is close to -10 (since we rotated 10)
    assert abs(angle + 10) < 5.0 or abs(angle - 80) < 5.0
