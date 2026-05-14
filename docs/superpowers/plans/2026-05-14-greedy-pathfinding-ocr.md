# OCR Greedy Pathfinding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Membangun sistem segmentasi baris teks pada struk belanja menggunakan algoritma Dijkstra berbasis Energy Map.

**Architecture:** Menggunakan pipeline linear: Deskewing (Hough) -> HPP Valley Detection -> Multi-Criteria Cost Map (Distance Transform) -> Dijkstra Pathfinding -> Ribbon Extraction.

**Tech Stack:** Python 3, OpenCV, NumPy, Scikit-Image, Numba (optimasi), Matplotlib.

---

## File Structure
- `src/preprocessing.py`: Fungsi untuk binarisasi, deskewing, dan HPP.
- `src/energy_map.py`: Kalkulasi Cost Map dengan Distance Transform (dioptimasi Numba).
- `src/pathfinding.py`: Implementasi Dijkstra menggunakan `heapq`.
- `src/segmentation.py`: Ekstraksi potongan gambar (ribbons) dan visualisasi.
- `src/utils.py`: Fungsi pembantu umum.
- `tests/`: Unit tests untuk setiap modul.

---

### Task 1: Basic Preprocessing (Binarization)

**Files:**
- Create: `src/preprocessing.py`
- Test: `tests/test_preprocessing.py`

- [ ] **Step 1: Write the failing test for binarization**

```python
import numpy as np
import pytest
from src.preprocessing import binarize_image

def test_binarize_image_returns_binary():
    # Create dummy grayscale image
    img = np.array([[50, 200], [150, 10]], dtype=np.uint8)
    binary = binarize_image(img)
    assert binary.shape == img.shape
    assert set(np.unique(binary)).issubset({0, 255})
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_preprocessing.py -k binarize`

- [ ] **Step 3: Implement binarization**

```python
import cv2

def binarize_image(image):
    # Use Otsu's thresholding
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_preprocessing.py`

- [ ] **Step 5: Commit**
```bash
git add src/preprocessing.py tests/test_preprocessing.py
git commit -m "feat: add image binarization"
```

---

### Task 2: Hough Transform Deskewing

**Files:**
- Modify: `src/preprocessing.py`
- Test: `tests/test_preprocessing.py`

- [ ] **Step 1: Write failing test for deskewing**

```python
def test_deskew_image_rotates_correctly():
    # This is a conceptual test, in real impl we check if angle is near 0
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.line(img, (10, 20), (90, 20), 255, 1) # Horizontal line
    # Rotate it 10 degrees
    M = cv2.getRotationMatrix2D((50, 50), 10, 1.0)
    rotated = cv2.warpAffine(img, M, (100, 100))
    
    deskewed, angle = deskew_image(rotated)
    assert abs(angle + 10) < 2.0 # Should detect ~10 deg
```

- [ ] **Step 2: Run test and fail**

- [ ] **Step 3: Implement Hough Deskewing**

```python
def deskew_image(binary_img):
    coords = np.column_stack(np.where(binary_img > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    (h, w) = binary_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(binary_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle
```

- [ ] **Step 4: Run test and pass**

- [ ] **Step 5: Commit**
```bash
git commit -am "feat: add automatic deskewing using minAreaRect"
```

---

### Task 3: HPP Valley Detection

**Files:**
- Modify: `src/preprocessing.py`
- Test: `tests/test_preprocessing.py`

- [ ] **Step 1: Write test for HPP valleys**

```python
def test_find_hpp_valleys():
    img = np.zeros((30, 100), dtype=np.uint8)
    img[5:10, :] = 255 # Line 1
    img[20:25, :] = 255 # Line 2
    valleys = find_hpp_valleys(img)
    # Expected valleys at top (0), middle (~15), bottom (29)
    assert any(10 < v < 20 for v in valleys)
```

- [ ] **Step 2: Implement HPP Valleys**

```python
def find_hpp_valleys(binary_img):
    hpp = np.sum(binary_img, axis=1)
    valleys = []
    threshold = np.mean(hpp) * 0.1
    for i in range(1, len(hpp) - 1):
        if hpp[i] <= threshold and hpp[i] < hpp[i-1] and hpp[i] < hpp[i+1]:
            valleys.append(i)
    # Add boundary valleys if not present
    if 0 not in valleys: valleys.insert(0, 0)
    if len(hpp)-1 not in valleys: valleys.append(len(hpp)-1)
    return valleys
```

---

### Task 4: Multi-Criteria Cost Map Engine (with Numba)

**Files:**
- Create: `src/energy_map.py`
- Test: `tests/test_energy_map.py`

- [ ] **Step 1: Write test for cost map**

```python
def test_create_cost_map_values():
    img = np.zeros((10, 10), dtype=np.uint8)
    img[5, 5] = 255 # One black pixel
    cost_map = create_cost_map(img)
    assert cost_map[5, 5] > cost_map[0, 0] # Text is more expensive
```

- [ ] **Step 2: Implement Cost Map with Distance Transform**

```python
from scipy.ndimage import distance_transform_edt
from numba import njit

@njit
def apply_penalties(cost_map, intensity, dist_inv, alpha, beta):
    return alpha * intensity + beta * dist_inv

def create_cost_map(binary_img, alpha=1000, beta=500):
    dist = distance_transform_edt(binary_img == 0)
    dist_inv = 1.0 / (dist + 0.5)
    intensity = binary_img.astype(np.float32) / 255.0
    
    cost_map = apply_penalties(intensity, dist_inv, alpha, beta)
    return cost_map
```

---

### Task 5: Dijkstra Pathfinding Core (heapq)

**Files:**
- Create: `src/pathfinding.py`
- Test: `tests/test_pathfinding.py`

- [ ] **Step 1: Write test for pathfinding**

```python
def test_dijkstra_finds_path():
    cost_map = np.ones((10, 10))
    cost_map[5, :] = 100 # High cost barrier
    path = find_seam(cost_map, start_y=2)
    assert len(path) == 10
    assert all(p[0] < 5 for p in path) # Should avoid the barrier
```

- [ ] **Step 2: Implement find_seam using heapq**

```python
import heapq

def find_seam(cost_map, start_y, vertical_penalty=10):
    h, w = cost_map.shape
    pq = [(0, start_y, 0, [])] # (cost, y, x, path)
    visited = set()
    
    while pq:
        c, y, x, path = heapq.heappop(pq)
        if (y, x) in visited: continue
        visited.add((y, x))
        
        new_path = path + [(y, x)]
        if x == w - 1: return new_path
        
        for dy in [-1, 0, 1]:
            ny, nx = y + dy, x + 1
            if 0 <= ny < h:
                nc = c + cost_map[ny, nx] + abs(dy) * vertical_penalty
                heapq.heappush(pq, (nc, ny, nx, new_path))
```

---

### Task 6: Ribbon Extraction

**Files:**
- Create: `src/segmentation.py`

- [ ] **Step 1: Implement extract_ribbons**

```python
def extract_ribbons(image, seams):
    ribbons = []
    for i in range(len(seams) - 1):
        upper = seams[i]
        lower = seams[i+1]
        # Logic to crop area between two seams
        # (Simplified: take min/max y for a rect crop, or warp for precise)
        y_min = min([p[0] for p in upper])
        y_max = max([p[0] for p in lower])
        ribbons.append(image[y_min:y_max, :])
    return ribbons
```

---

### Task 7: Final Integration & Visualizer

**Files:**
- Create: `src/main.py`

- [ ] **Step 1: Implement Main CLI with Visualization**

```python
def main(img_path):
    img = cv2.imread(img_path, 0)
    binary = binarize_image(img)
    rotated, _ = deskew_image(binary)
    valleys = find_hpp_valleys(rotated)
    cost_map = create_cost_map(rotated)
    
    seams = [find_seam(cost_map, v) for v in valleys]
    
    # Visualization logic using matplotlib
    import matplotlib.pyplot as plt
    plt.imshow(rotated, cmap='gray')
    for seam in seams:
        plt.plot([p[1] for p in seam], [p[0] for p in seam])
    plt.show()
```
