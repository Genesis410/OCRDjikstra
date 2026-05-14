# Design Document: Greedy Pathfinding for Line Segmentation (OCR-Dijkstra)

**Date:** 2026-05-14
**Status:** Draft
**Author:** Gemini CLI & User

## 1. Overview
Project ini bertujuan untuk memecahkan masalah segmentasi baris teks pada dokumen yang kompleks (khususnya struk belanja) menggunakan pendekatan pathfinding berbasis graf (Dijkstra/A*). Metode ini lebih unggul dibandingkan proyeksi histogram tradisional karena mampu menangani teks yang saling bersentuhan (*touching characters*) dan baris yang tidak lurus sempurna dengan mencari jalur "energi terendah" di antara baris teks.

## 2. Problem Statement
Struk belanja seringkali memiliki kualitas cetakan rendah, baris teks yang sangat rapat, dan posisi pengambilan foto yang miring (*skewed*). Algoritma konvensional seringkali memotong bagian atas atau bawah karakter (*ascenders/descenders*) atau gagal memisahkan dua baris yang sedikit bersentuhan.

## 3. Proposed Solution
Menggunakan algoritma Dijkstra atau A* untuk mencari jalur optimal (seam) dari sisi kiri ke sisi kanan gambar. Jalur ini akan bertindak sebagai pembatas antar baris. "Keinginan" jalur untuk menghindari teks diatur melalui **Energy/Cost Map** yang kompleks.

## 4. Architecture

### Module 1: Preprocessing & Deskewing
*   **Grayscale & Adaptive Thresholding**: Mengonversi gambar ke biner untuk memisahkan tinta dari background.
*   **Hough Transform Deskewing**: Mendeteksi garis dominan pada teks untuk menghitung sudut kemiringan, kemudian melakukan rotasi otomatis agar teks sejajar horizontal.
*   **Horizontal Projection Profile (HPP)**: Menghitung jumlah pixel hitam per baris untuk mengidentifikasi "valleys" (celah antar baris) sebagai titik *Source* (awal) dan *Sink* (akhir) untuk pathfinding.

### Module 2: Energy Map Engine
Membangun matriks biaya $C(y, x)$ untuk setiap koordinat pixel:
*   **Intensity Component**: Pixel hitam (teks) memiliki biaya sangat tinggi.
*   **Distance Transform Component**: Menggunakan teknik *Euclidean Distance Transform* untuk memberikan biaya tambahan pada pixel putih yang berada dekat dengan teks. Rumus: $Cost_{dist} = \frac{1}{Distance(y, x) + \epsilon}$.
*   **Vertical Penalty**: Memberikan biaya tambahan untuk pergerakan vertikal ($\Delta y \neq 0$) agar jalur cenderung tetap horizontal dan halus.

### Module 3: Pathfinding Logic
*   **Algorithm**: Dijkstra (untuk akurasi absolut) atau A* (dengan heuristik jarak horizontal ke tepi kanan).
*   **State Space**: Setiap pixel $(y, x)$ adalah node. Edge terhubung ke 3 tetangga di kolom berikutnya: $(y-1, x+1), (y, x+1), (y+1, x+1)$.
*   **Constraint**: Mengizinkan jalur memotong pixel hitam jika benar-benar terjepit, namun dengan penalti energi yang sangat tinggi.

### Module 4: Extraction & Visualization
*   **Ribbon Extraction**: Memotong gambar asli menjadi baris-baris teks berdasarkan jalur pembatas.
*   **Heatmap Visualization**: Menampilkan Cost Map dalam bentuk warna (panas/dingin) untuk menunjukkan "rintangan" yang dihindari algoritma.
*   **Path Overlay**: Menggambar jalur berwarna-warni di atas citra asli.

## 5. Technology Stack
*   **Language**: Python 3.x
*   **Libraries**:
    *   `OpenCV`: Image processing & Hough Transform.
    *   `NumPy`: Matriks manipulasi & Cost Map calculation.
    *   `Scikit-Image`: Distance transform & advanced morphology.
    *   `Matplotlib`: Visualisasi heatmap dan grafik HPP.

## 6. Success Criteria
1.  Dapat memisahkan baris teks yang memiliki karakter yang bersentuhan secara vertikal tanpa memotong bagian utama karakter.
2.  Robust terhadap kemiringan struk hingga 15 derajat.
3.  Menghasilkan visualisasi yang menunjukkan logika di balik pemilihan jalur (Heatmap & Path Overlay).

## 7. Future Work
*   Integrasi dengan OCR Engine (Tesseract/PaddleOCR) setelah segmentasi.
*   Pengembangan menjadi "Deep Seam Carving" di mana Energy Map diprediksi menggunakan Neural Network.
