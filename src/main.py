import cv2
import matplotlib.pyplot as plt
from src.preprocessing import binarize_image, deskew_image, find_hpp_valleys
from src.energy_map import create_cost_map
from src.pathfinding import find_seam
from src.segmentation import extract_ribbons

def process_receipt(img_path):
    # 1. Load image
    print(f"Loading image: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Image not found.")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Preprocess
    print("Binarizing and deskewing...")
    binary = binarize_image(gray)
    # Note: binarize_image returns inverted binary (white text on black)
    rotated, angle = deskew_image(binary)
    print(f"Detected skew angle: {angle:.2f} degrees")
    
    # 3. Find Valleys
    print("Finding valleys...")
    valleys = find_hpp_valleys(rotated)
    print(f"Detected {len(valleys)} valleys (line boundaries).")
    
    # 4. Create Cost Map
    print("Generating Energy Map...")
    cost_map = create_cost_map(rotated)
    
    # 5. Pathfinding
    print("Running Dijkstra for each valley...")
    seams = []
    for i, v in enumerate(valleys):
        print(f"  Processing seam {i+1}/{len(valleys)}...")
        seam = find_seam(cost_map, v)
        if seam:
            seams.append(seam)
            
    # 6. Extract Ribbons
    print("Extracting ribbons...")
    ribbons = extract_ribbons(rotated, seams)
    print(f"Extracted {len(ribbons)} text line ribbons.")
    
    # 7. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    
    # Left: Seams on rotated image
    axes[0].imshow(rotated, cmap='gray')
    for seam in seams:
        y_coords = [p[0] for p in seam]
        x_coords = [p[1] for p in seam]
        axes[0].plot(x_coords, y_coords, linewidth=2)
    axes[0].set_title("Line Segmentation Seams")
    
    # Right: Cost Map Heatmap
    im = axes[1].imshow(cost_map, cmap='hot')
    plt.colorbar(im, ax=axes[1], shrink=0.5)
    axes[1].set_title("Energy (Cost) Map")
    
    plt.tight_layout()
    plt.savefig("segmentation_result.png")
    print("Result saved to segmentation_result.png")
    # plt.show() # Commented out for non-interactive execution

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "tests/test_receipt.png"
    process_receipt(path)
