# UTS Deep Learning (Kelas B) — Prediksi Trajektori Bola Basket

Project ini dibuat untuk tugas UTS Deep Learning, dengan tujuan memprediksi trajektori bola basket berdasarkan data tracking video menggunakan YOLO, sliding window, dan model Simple RNN, LSTM, serta GRU.

## Deskripsi

Pipeline project ini mencakup:
- Ekstraksi data posisi bola dari video.
- Praproses data dan ekspor ke CSV.
- Train/test split secara kronologis.
- Pembuatan sliding window untuk data sekuens.
- Pelatihan model Simple RNN, LSTM, dan GRU.
- Evaluasi performa model dan visualisasi hasil prediksi.

## Isi Project

- `UTS_Deep_Learning_B.ipynb` — notebook utama.
- `output/` — hasil ekstraksi data, split data, dan file evaluasi.
- `figures/` — visualisasi hasil analisis dan perbandingan model.
- `models/` — model hasil training.
- `debug_frames/` — hasil deteksi sampel dari YOLO.

## Ringkasan Proses

1. Video dipotong menjadi bagian 3 menit.
2. Bola basket dideteksi dan dilacak menggunakan YOLO.
3. Koordinat bola diekstrak menjadi dataset.
4. Data dibagi menjadi train dan test secara kronologis.
5. Sliding window dibuat untuk input model.
6. Tiga model deep learning dilatih dan dibandingkan.
7. Hasil prediksi divisualisasikan dan dievaluasi.

## Hasil Akhir

Project ini menghasilkan perbandingan performa model untuk memprediksi posisi bola basket pada data uji. Evaluasi dilakukan menggunakan metrik seperti MSE, RMSE, dan MAE.

## Author

**KHAERUL HADISWARA**  
**NIM:** 25917025
