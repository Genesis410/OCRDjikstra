import cv2
import numpy as np

def binarize_image(image):
    # Use Otsu's thresholding
    # cv2.THRESH_BINARY_INV is used because usually OCR works better with white text on black background 
    # or the input is expected to be inverted for distance transform etc.
    # The prompt specifically used THRESH_BINARY_INV.
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary

def deskew_image(binary_img):
    # Find all non-zero pixels (the text)
    coords = np.column_stack(np.where(binary_img > 0))
    # minAreaRect returns (center(x, y), size(width, height), angle)
    angle = cv2.minAreaRect(coords)[-1]
    
    # minAreaRect angle is in range [-90, 0)
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    (h, w) = binary_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(binary_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle
