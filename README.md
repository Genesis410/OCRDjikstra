# OCRDjikstra: Greedy Pathfinding for Line Segmentation

**OCRDjikstra** is a robust line segmentation system for complex documents (specifically shopping receipts) using a graph-based pathfinding approach (Dijkstra/A*). 

Unlike traditional horizontal projection methods that fail on skewed, crumpled, or tightly-packed text, this project treats the image as a weighted graph. It calculates a "Multi-Criteria Energy Map" and finds the optimal "seam" (path of least energy) between lines of text, ensuring accurate segmentation even when characters from different lines are nearly touching.

## 🚀 Features
- **Automatic Deskewing**: Uses Hough Transform and `minAreaRect` to align slanted images automatically.
- **Energy-Based Segmentation**: Combines pixel intensity and **Distance Transform** to create a cost map that "repels" segmentation paths away from text.
- **Dijkstra Optimization**: Guaranteed optimal pathfinding using `heapq` for efficiency.
- **High Performance**: Numerical computations are accelerated using **Numba (JIT compilation)**.
- **Visualization**: Built-in tools to visualize the Energy Heatmap and the resulting segmentation seams.

## 🛠️ Tech Stack
- **Language**: Python 3.x
- **Computer Vision**: OpenCV, Scikit-Image
- **Numerical Ops**: NumPy, SciPy, Numba
- **Visualization**: Matplotlib
- **Testing**: Pytest

## 📦 Installation
```bash
# Clone the repository
git clone https://github.com/Genesis410/OCRDjikstra.git
cd OCRDjikstra

# Install dependencies
pip install -r requirements.txt
```

## 🖥️ Usage
To run the segmentation pipeline on a sample receipt:
```bash
# Generate a synthetic test image (optional)
python src/generate_test_image.py

# Run the main segmentation script
python src/main.py tests/test_receipt.png
```
The result will be saved as `segmentation_result.png` and text ribbons will be extracted as individual arrays.

## 🧪 Testing
We follow a Test-Driven Development (TDD) approach. To run the test suite:
```bash
pytest
```

---

## ⚠️ Disclaimer & Status
**This is the FIRST IMPLEMENTATION.** 
This project is currently a proof-of-concept focusing on the core pathfinding logic for line segmentation. 

**Future Roadmap:**
- [ ] **Optimizations**: Further performance tuning for mobile-sized images.
- [ ] **Advanced Testing**: Benchmarking against highly distorted or low-light receipt datasets.
- [ ] **OCR Integration**: Connecting the extracted ribbons to Tesseract or PaddleOCR for full text recognition.
- [ ] **Curved Line Support**: Implementing non-linear warping for warped/cylindrical surface receipts.

Feel free to contribute or suggest further improvements!
