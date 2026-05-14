import cv2

def binarize_image(image):
    # Use Otsu's thresholding
    # cv2.THRESH_BINARY_INV is used because usually OCR works better with white text on black background 
    # or the input is expected to be inverted for distance transform etc.
    # The prompt specifically used THRESH_BINARY_INV.
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary
