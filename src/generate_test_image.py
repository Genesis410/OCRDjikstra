import cv2
import numpy as np

def create_synthetic_receipt(path="tests/test_receipt.png"):
    # Create white image 500x300
    img = np.ones((500, 300), dtype=np.uint8) * 255
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        "SUPERMARKET XYZ",
        "DATE: 2026-05-14",
        "----------------",
        "APPLE      1.50",
        "BANANA     0.80",
        "MILK       2.10",
        "----------------",
        "TOTAL      4.40"
    ]
    
    y = 50
    for line in lines:
        cv2.putText(img, line, (20, y), font, 0.7, 0, 2, cv2.LINE_AA)
        y += 40 # Space between lines
        
    # Add a slight skew (rotate 5 degrees)
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 5, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    cv2.imwrite(path, img)
    print(f"Synthetic receipt saved to {path}")

if __name__ == "__main__":
    create_synthetic_receipt()
