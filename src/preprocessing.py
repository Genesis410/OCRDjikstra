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

def find_hpp_valleys(binary_img):
    # Sum of black pixels (which are white 255 in our inverted binary image) per row
    hpp = np.sum(binary_img, axis=1)
    valleys = []
    # Threshold to determine what is a "valley" (low intensity area)
    # Using 10% of mean as a heuristic
    threshold = np.mean(hpp) * 0.1
    
    # Pad hpp to handle boundaries in the same loop
    padded_hpp = np.pad(hpp, (1, 1), mode='constant', constant_values=hpp.max())
    
    for i in range(1, len(padded_hpp) - 1):
        idx = i - 1
        # Local minima below threshold
        if padded_hpp[i] <= threshold and padded_hpp[i] <= padded_hpp[i-1] and padded_hpp[i] <= padded_hpp[i+1]:
            # Simple check to avoid cluster of points in the same valley
            if not valleys or idx - valleys[-1] > 5:
                valleys.append(idx)
                
    # Ensure boundaries are included if the gaps are large
    if not valleys or valleys[0] > 0:
        if not valleys or valleys[0] > 5:
            valleys.insert(0, 0)
        else:
            valleys[0] = 0
            
    if valleys[-1] < len(hpp) - 1:
        if len(hpp) - 1 - valleys[-1] > 5:
            valleys.append(len(hpp) - 1)
        else:
            valleys[-1] = len(hpp) - 1
        
    return valleys
