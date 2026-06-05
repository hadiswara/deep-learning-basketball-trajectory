#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script untuk membuat Jupyter Notebook UTS Deep Learning (B)
Prediksi Trajektori Bola Basket menggunakan RNN, LSTM, dan GRU
"""

import json
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import os

cells = []

# ══════════════════════════════════════════════════════════════════════════════
# HEADER UTAMA
# ══════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("""\
# UTS Deep Learning (Kelas B) — Prediksi Trajektori Bola Basket

**Mata Kuliah:** Deep Learning  
**Prodi/Kelas:** Magister Informatika / B  
**Dosen Penguji:** Ir. Chandra Kusuma Dewa, S.Kom., M.Cs., Ph.D.  
**Tanggal Ujian:** Jumat, 5 Juni 2026  
**Sifat:** Take Home (Deadline: 12 Juni 2026)

---

## Deskripsi Tugas

Membangun model Deep Learning yang mampu memprediksi trajektori (posisi koordinat X, Y) bola basket di masa depan berdasarkan pola pergerakan beberapa frame sebelumnya.

## Daftar Isi

| No | Topik | Bobot |
|----|-------|-------|
| 1 | Ekstraksi & YOLO Tracking | 15% |
| 2 | Data Engineering - CSV Exporter | 10% |
| 3 | Data Engineering - Train/Test Split Kronologis | 10% |
| 4 | Data Engineering - Sliding Window | 15% |
| 5 | Arsitektur Model (Simple RNN, LSTM, GRU) | 20% |
| 6 | Evaluasi Proses Training | 10% |
| 7 | Pengujian & Komparasi Trajektori | 20% |

---
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SETUP & IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("""\
## ⚙️ Setup & Import Library
"""))

cells.append(new_code_cell("""\
# ══════════════════════════════════════════════════════════════════════════════
# SETUP AWAL: Import library dan konfigurasi global
# ══════════════════════════════════════════════════════════════════════════════
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import warnings
import time

warnings.filterwarnings('ignore')
matplotlib.rcParams['figure.dpi'] = 120
matplotlib.rcParams['figure.figsize'] = (12, 6)
plt.style.use('seaborn-v0_8-whitegrid')

# Buat folder output
for folder in ['output', 'figures', 'models']:
    os.makedirs(folder, exist_ok=True)

print("=" * 60)
print("ENVIRONMENT CHECK")
print("=" * 60)

print(f"  OpenCV     : {cv2.__version__}")
print(f"  NumPy      : {np.__version__}")
print(f"  Pandas     : {pd.__version__}")
print(f"  Matplotlib : {matplotlib.__version__}")

# Cek YOLO
try:
    from ultralytics import YOLO
    import ultralytics
    print(f"  Ultralytics: {ultralytics.__version__}")
except ImportError:
    print("  Ultralytics: BELUM TERINSTALL! Jalankan: pip install ultralytics")

# Cek TensorFlow/Keras
import tensorflow as tf
print(f"  TensorFlow : {tf.__version__}")
print(f"  GPU        : {tf.config.list_physical_devices('GPU') or 'CPU Only'}")

print("\\n✅ Semua library siap digunakan.")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# PERSIAPAN VIDEO - POTONG 3 MENIT
# ══════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("""\
## 📹 Persiapan Video — Pemotongan 3 Menit

Sesuai instruksi soal, video asli dipotong menjadi **3 menit** dengan mengambil **bagian tengah** video di mana objek bola basket terlihat jelas dan aktif.

- **Video asli:** `DEEP LEARNING VIDEO.mp4` (~10 menit, 30 FPS)
- **Durasi potong:** 3 menit (180 detik = 5.400 frame)
- **Posisi potong:** Menit 3:30 s.d. 6:30 (tengah video)
"""))

cells.append(new_code_cell("""\
# ══════════════════════════════════════════════════════════════════════════════
# PEMOTONGAN VIDEO: Ambil 3 menit bagian tengah
# ══════════════════════════════════════════════════════════════════════════════

VIDEO_INPUT = "DEEP LEARNING VIDEO.mp4"
VIDEO_OUTPUT = "output/video_3menit.mp4"

# Cek apakah video potong sudah ada (skip jika sudah)
if os.path.exists(VIDEO_OUTPUT) and os.path.getsize(VIDEO_OUTPUT) > 1_000_000:
    print(f"✅ Video potongan sudah ada: {VIDEO_OUTPUT}")
    cap_check = cv2.VideoCapture(VIDEO_OUTPUT)
    total_frames = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap_check.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    print(f"   Durasi: {duration:.1f}s ({total_frames} frame @ {fps} FPS)")
    cap_check.release()
else:
    print(f"Memotong video: {VIDEO_INPUT}")
    
    cap = cv2.VideoCapture(VIDEO_INPUT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_orig = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_orig = total_frames_orig / fps
    
    print(f"  Video asli: {duration_orig:.1f}s, {total_frames_orig} frame, {fps} FPS, {width}x{height}")
    
    # Ambil bagian tengah: mulai dari menit 3:30, durasi 3 menit
    start_sec = 3 * 60 + 30  # 3 menit 30 detik = 210 detik
    end_sec = start_sec + 180  # + 3 menit = 390 detik
    
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    
    print(f"  Pemotongan: frame {start_frame} - {end_frame} (detik {start_sec} - {end_sec})")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (width, height))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = 0
    target_frames = end_frame - start_frame
    
    while cap.isOpened() and frame_count < target_frames:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        frame_count += 1
        
        if frame_count % 1000 == 0:
            pct = frame_count / target_frames * 100
            print(f"  Progress: {frame_count}/{target_frames} frame ({pct:.1f}%)")
    
    writer.release()
    cap.release()
    
    print(f"\\n✅ Video berhasil dipotong!")
    print(f"   Output: {VIDEO_OUTPUT}")
    print(f"   Frame: {frame_count} ({frame_count/fps:.1f} detik)")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SOAL 1: YOLO TRACKING (15%)
# ══════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("""\
---
# Soal 1: Ekstraksi & YOLO Tracking (15%)

## 1.1 Deskripsi

Pada bagian ini, kami mengimplementasikan **YOLO Tracking** menggunakan `model.track()` dari library Ultralytics untuk mendeteksi dan melacak ketiga objek bola basket secara simultan pada video 3 menit.

## 1.2 Penjelasan Metode

### YOLOv8 + Object Tracking

**YOLO (You Only Look Once)** adalah arsitektur deteksi objek real-time yang memproses seluruh gambar dalam satu forward pass. YOLOv8 merupakan versi terbaru dari keluarga YOLO yang dikembangkan oleh Ultralytics.

**Pipeline tracking yang digunakan:**
1. **Detection:** YOLOv8 mendeteksi objek pada setiap frame dan menghasilkan bounding box + class prediction
2. **Tracking:** Algoritma **BoT-SORT** (atau ByteTrack) mencocokkan deteksi antar frame untuk memberikan **ID unik** yang konsisten pada setiap objek
3. **Filtering:** Hanya objek dengan class `sports ball` (class ID 32 pada COCO dataset) yang diambil

### Parameter Kunci:
- **Model:** `yolov8s.pt` (Small — keseimbangan antara akurasi dan kecepatan)
- **Confidence threshold:** 0.3 (cukup rendah untuk menangkap bola yang blur/kecil)
- **IOU threshold:** 0.5 (standar untuk Non-Maximum Suppression)
- **Tracker:** BoT-SORT (default, lebih robust untuk multi-object tracking)

### Mengapa YOLOv8?
- **Real-time performance:** Mampu memproses frame dengan cepat bahkan pada CPU
- **Built-in tracking:** Method `model.track()` sudah mengintegrasikan algoritma multi-object tracking
- **Pre-trained pada COCO:** Sudah mengenali 80 class objek termasuk `sports ball`
- **Persistent ID:** Tracking menjaga ID bola tetap konsisten antar frame
"""))

cells.append(new_code_cell("""\
# ══════════════════════════════════════════════════════════════════════════════
# SOAL 1: YOLO TRACKING — Deteksi & Lacak 3 Bola Basket
# ══════════════════════════════════════════════════════════════════════════════

from ultralytics import YOLO

# Load model YOLOv8 (Small variant — balance kecepatan & akurasi)
model = YOLO("yolov8s.pt")

# Path video 3 menit
VIDEO_PATH = "output/video_3menit.mp4"

# Cek apakah hasil tracking sudah ada (untuk skip re-run)
CSV_PATH = "output/ball_tracking_raw.csv"

if os.path.exists(CSV_PATH):
    print(f"✅ Data tracking sudah ada di: {CSV_PATH}")
    print("   (Hapus file tersebut jika ingin re-run tracking)")
    df_tracking = pd.read_csv(CSV_PATH)
    print(f"   Total record: {len(df_tracking):,}")
else:
    print("Memulai YOLO Tracking pada video 3 menit...")
    print("=" * 60)
    
    # Buka video
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Video: {VIDEO_PATH}")
    print(f"  Total frame: {total_frames}, FPS: {fps}")
    print(f"  YOLO model: yolov8s.pt")
    print(f"  Target class: sports ball (ID 32)")
    print("-" * 60)
    
    cap.release()
    
    # Jalankan tracking menggunakan model.track()
    # persist=True agar track ID persisten antar frame
    tracking_data = []
    frame_idx = 0
    
    start_time = time.time()
    
    results = model.track(
        source=VIDEO_PATH,
        classes=[32],          # Filter: hanya class 'sports ball'
        conf=0.3,              # Confidence threshold
        iou=0.5,               # IOU threshold untuk NMS
        persist=True,          # Pertahankan tracking ID antar frame
        stream=True,           # Stream mode untuk efisiensi memori
        verbose=False          # Kurangi output log
    )
    
    for result in results:
        frame_idx += 1
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Ambil bounding box (xyxy format)
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                
                # Confidence score
                conf = float(boxes.conf[i].cpu().numpy())
                
                # Track ID (jika tersedia)
                if boxes.id is not None:
                    track_id = int(boxes.id[i].cpu().numpy())
                else:
                    track_id = -1
                
                # Hitung titik tengah (center point)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Lebar dan tinggi bounding box
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                
                tracking_data.append({
                    'frame': frame_idx,
                    'track_id': track_id,
                    'center_x': round(center_x, 2),
                    'center_y': round(center_y, 2),
                    'bbox_x1': round(float(x1), 2),
                    'bbox_y1': round(float(y1), 2),
                    'bbox_x2': round(float(x2), 2),
                    'bbox_y2': round(float(y2), 2),
                    'bbox_w': round(float(bbox_w), 2),
                    'bbox_h': round(float(bbox_h), 2),
                    'confidence': round(conf, 4)
                })
        
        # Progress update setiap 500 frame
        if frame_idx % 500 == 0:
            elapsed = time.time() - start_time
            pct = frame_idx / total_frames * 100
            fps_proc = frame_idx / elapsed
            print(f"  Frame {frame_idx}/{total_frames} ({pct:.1f}%) "
                  f"| {fps_proc:.1f} FPS | Deteksi: {len(tracking_data)} total")
    
    elapsed = time.time() - start_time
    
    # Simpan ke DataFrame
    df_tracking = pd.DataFrame(tracking_data)
    df_tracking.to_csv(CSV_PATH, index=False)
    
    print("\\n" + "=" * 60)
    print("✅ YOLO TRACKING SELESAI!")
    print("=" * 60)
    print(f"  Waktu proses    : {elapsed:.1f} detik ({elapsed/60:.1f} menit)")
    print(f"  Total frame     : {frame_idx}")
    print(f"  Total deteksi   : {len(df_tracking):,}")
    print(f"  Output CSV      : {CSV_PATH}")
"""))

cells.append(new_code_cell("""\
# ══════════════════════════════════════════════════════════════════════════════
# Analisis Hasil Tracking
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("RINGKASAN HASIL TRACKING")
print("=" * 60)

print(f"\\nTotal deteksi        : {len(df_tracking):,}")
print(f"Jumlah frame unik    : {df_tracking['frame'].nunique():,}")
print(f"Jumlah track ID unik : {df_tracking['track_id'].nunique()}")
print(f"\\nTrack ID yang terdeteksi:")

track_stats = df_tracking.groupby('track_id').agg(
    jumlah_frame=('frame', 'count'),
    frame_awal=('frame', 'min'),
    frame_akhir=('frame', 'max'),
    avg_conf=('confidence', 'mean'),
    avg_x=('center_x', 'mean'),
    avg_y=('center_y', 'mean')
).round(2)

print(track_stats.to_string())

print(f"\\n5 baris pertama data tracking:")
print(df_tracking.head().to_string())
"""))

cells.append(new_code_cell("""\
# ══════════════════════════════════════════════════════════════════════════════
# Visualisasi: Sample Frame dengan Tracking
# ══════════════════════════════════════════════════════════════════════════════

# Ambil beberapa sample frame untuk visualisasi
VIDEO_PATH = "output/video_3menit.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Pilih 4 frame sample (tersebar di video)
sample_frames = [100, total_frames // 4, total_frames // 2, 3 * total_frames // 4]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

COLORS = {-1: (128, 128, 128)}
color_palette = [(46, 204, 113), (231, 76, 60), (52, 152, 219), 
                 (241, 196, 15), (155, 89, 182), (230, 126, 34)]

for idx, (frame_num, ax) in enumerate(zip(sample_frames, axes)):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
    ret, frame = cap.read()
    
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Ambil deteksi untuk frame ini
        frame_detections = df_tracking[df_tracking['frame'] == frame_num]
        
        # Gambar bounding box dan center point
        for _, det in frame_detections.iterrows():
            tid = int(det['track_id'])
            if tid not in COLORS:
                COLORS[tid] = color_palette[len(COLORS) % len(color_palette)]
            color = COLORS[tid]
            
            # Bounding box
            cv2.rectangle(frame_rgb, 
                         (int(det['bbox_x1']), int(det['bbox_y1'])),
                         (int(det['bbox_x2']), int(det['bbox_y2'])),
                         color, 3)
            
            # Center point
            cv2.circle(frame_rgb, 
                      (int(det['center_x']), int(det['center_y'])),
                      8, color, -1)
            
            # Label
            label = f"Ball {tid} ({det['confidence']:.2f})"
            cv2.putText(frame_rgb, label,
                       (int(det['bbox_x1']), int(det['bbox_y1']) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        ax.imshow(frame_rgb)
        ax.set_title(f'Frame {frame_num} — {len(frame_detections)} bola terdeteksi',
                    fontsize=12, fontweight='bold')
    ax.axis('off')

cap.release()

fig.suptitle('Sample Frame dengan YOLO Tracking', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/yolo_tracking_samples.png', dpi=150, bbox_inches='tight')
plt.show()
print("Gambar disimpan ke figures/yolo_tracking_samples.png")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SOAL 2: CSV EXPORTER (10%)
# ══════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("""\
---
# Soal 2: Data Engineering — CSV Exporter (10%)

## 2.1 Deskripsi

Pada bagian ini, data tracking dari YOLO diolah untuk mengekstrak **titik tengah koordinat (X, Y)** dari ketiga bola beserta **ID unik** masing-masing, kemudian disimpan ke dalam satu file CSV/DataFrame terpadu.

## 2.2 Penjelasan Proses

### Langkah-langkah:
1. **Filtering:** Pilih hanya 3 track ID dengan jumlah deteksi terbanyak (diasumsikan sebagai 3 bola utama)
2. **Reassign ID:** Berikan label yang konsisten (Ball_1, Ball_2, Ball_3) berdasarkan rata-rata posisi X (kiri ke kanan)
3. **Interpolasi:** Isi frame yang hilang (missing detection) menggunakan interpolasi linear agar data kontinu
4. **Normalisasi:** Koordinat dinormalisasi ke rentang [0, 1] berdasarkan resolusi video untuk konsistensi antar resolusi

### Format CSV Output:
| Kolom | Deskripsi |
|-------|-----------|
| `frame` | Nomor frame (1-indexed) |
| `ball_id` | ID unik bola (1, 2, atau 3) |
| `center_x` | Koordinat X titik tengah bola |
| `center_y` | Koordinat Y titik tengah bola |
| `center_x_norm` | Koordinat X ternormalisasi (0-1) |
| `center_y_norm` | Koordinat Y ternormalisasi (0-1) |
"""))

cells.append(new_code_cell("""\
# ══════════════════════════════════════════════════════════════════════════════
# SOAL 2: Ekstraksi Koordinat Tengah & Export CSV
# ══════════════════════════════════════════════════════════════════════════════

# Load data tracking
df_tracking = pd.read_csv("output/ball_tracking_raw.csv")

print("=" * 60)
print("DATA ENGINEERING — CSV EXPORTER")
print("=" * 60)

# --- Langkah 1: Identifikasi 3 bola utama (track ID dengan deteksi terbanyak) ---
# Filter track ID valid (bukan -1)
df_valid = df_tracking[df_tracking['track_id'] != -1].copy()

# Hitung jumlah deteksi per track ID
track_counts = df_valid['track_id'].value_counts()
print("\\nJumlah deteksi per Track ID (Top 10):")
print(track_counts.head(10))

# Ambil 3 track ID teratas
top_3_ids = track_counts.head(3).index.tolist()
print(f"\\n3 Bola utama (Track ID): {top_3_ids}")

# --- Langkah 2: Filter hanya 3 bola utama ---
df_balls = df_valid[df_valid['track_id'].isin(top_3_ids)].copy()

# --- Langkah 3: Reassign ball_id berdasarkan posisi X rata-rata ---
avg_x_per_track = df_balls.groupby('track_id')['center_x'].mean().sort_values()
id_mapping = {old_id: new_id + 1 for new_id, old_id in enumerate(avg_x_per_track.index)}
df_balls['ball_id'] = df_balls['track_id'].map(id_mapping)

print("\\nMapping Track ID -> Ball ID (berdasarkan posisi X rata-rata):")
for old_id, new_id in id_mapping.items():
    avg_x = avg_x_per_track[old_id]
    print(f"  Track {old_id} (avg X={avg_x:.1f}) -> Ball_{new_id}")

# --- Langkah 4: Buat DataFrame terpadu ---
# Resolusi video untuk normalisasi
cap = cv2.VideoCapture("output/video_3menit.mp4")
VIDEO_W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
VIDEO_H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

# Pilih kolom yang relevan
df_export = df_balls[['frame', 'ball_id', 'center_x', 'center_y']].copy()

# Normalisasi koordinat ke [0, 1]
df_export['center_x_norm'] = (df_export['center_x'] / VIDEO_W).round(6)
df_export['center_y_norm'] = (df_export['center_y'] / VIDEO_H).round(6)

# Urutkan berdasarkan frame dan ball_id
df_export = df_export.sort_values(['frame', 'ball_id']).reset_index(drop=True)

# --- Langkah 5: Interpolasi frame yang hilang ---
print(f"\\n--- Interpolasi Missing Frames ---")

df_interpolated_list = []
for bid in sorted(df_export['ball_id'].unique()):
    df_ball = df_export[df_export['ball_id'] == bid].copy()
    
    # Buat index lengkap dari frame pertama hingga terakhir
    min_frame = df_ball['frame'].min()
    max_frame = df_ball['frame'].max()
    all_frames = pd.DataFrame({'frame': range(min_frame, max_frame + 1)})
    
    # Merge dan interpolasi
    df_merged = all_frames.merge(df_ball, on='frame', how='left')
    df_merged['ball_id'] = bid
    
    missing_before = df_merged['center_x'].isna().sum()
    
    # Interpolasi linear untuk koordinat
    for col in ['center_x', 'center_y', 'center_x_norm', 'center_y_norm']:
        df_merged[col] = df_merged[col].interpolate(method='linear')
    
    # Forward/backward fill untuk edge cases
    df_merged = df_merged.ffill().bfill()
    
    missing_after = df_merged['center_x'].isna().sum()
    print(f"  Ball_{bid}: {min_frame}-{max_frame} frame, "
          f"missing {missing_before} -> {missing_after}")
    
    df_interpolated_list.append(df_merged)

df_final = pd.concat(df_interpolated_list, ignore_index=True)
df_final = df_final.sort_values(['frame', 'ball_id']).reset_index(drop=True)

# --- Langkah 6: Simpan ke CSV ---
CSV_OUTPUT = "output/ball_coordinates.csv"
df_final.to_csv(CSV_OUTPUT, index=False)

print(f"\\n{'=' * 60}")
print(f"✅ CSV BERHASIL DISIMPAN!")
print(f"{'=' * 60}")
print(f"  File     : {CSV_OUTPUT}")
print(f"  Total row: {len(df_final):,}")
print(f"  Kolom    : {list(df_final.columns)}")
print(f"  Ball IDs : {sorted(df_final['ball_id'].unique())}")
print(f"  Resolusi : {int(VIDEO_W)}x{int(VIDEO_H)}")

print(f"\\n--- 10 Baris Pertama ---")
print(df_final.head(10).to_string())

print(f"\\n--- Statistik Deskriptif ---")
print(df_final.groupby('ball_id')[['center_x', 'center_y']].describe().round(2))
"""))

cells.append(new_code_cell("""\
# ══════════════════════════════════════════════════════════════════════════════
# Visualisasi: Jejak Pergerakan 3 Bola
# ══════════════════════════════════════════════════════════════════════════════

df_final = pd.read_csv("output/ball_coordinates.csv")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

BALL_COLORS = {1: '#2ecc71', 2: '#e74c3c', 3: '#3498db'}
BALL_NAMES = {1: 'Bola 1', 2: 'Bola 2', 3: 'Bola 3'}

# --- Plot 1: Jejak XY semua bola ---
ax = axes[0]
for bid in sorted(df_final['ball_id'].unique()):
    df_b = df_final[df_final['ball_id'] == bid]
    ax.plot(df_b['center_x'], df_b['center_y'], 
            color=BALL_COLORS.get(bid, 'gray'), alpha=0.5,
            linewidth=0.8, label=BALL_NAMES.get(bid, f'Bola {bid}'))
    # Titik awal dan akhir
    ax.scatter(df_b['center_x'].iloc[0], df_b['center_y'].iloc[0],
              color=BALL_COLORS.get(bid, 'gray'), marker='o', s=100, 
              zorder=5, edgecolors='black', linewidth=1.5)
    ax.scatter(df_b['center_x'].iloc[-1], df_b['center_y'].iloc[-1],
              color=BALL_COLORS.get(bid, 'gray'), marker='s', s=100, 
              zorder=5, edgecolors='black', linewidth=1.5)

ax.set_xlabel('Koordinat X (pixel)', fontsize=12)
ax.set_ylabel('Koordinat Y (pixel)', fontsize=12)
ax.set_title('Jejak Pergerakan 3 Bola Basket (XY)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.invert_yaxis()  # Y axis video: atas = 0
ax.set_aspect('equal')

# --- Plot 2: Koordinat X dan Y terhadap Frame ---
ax = axes[1]
bid_sample = sorted(df_final['ball_id'].unique())[0]
df_sample = df_final[df_final['ball_id'] == bid_sample]
ax.plot(df_sample['frame'], df_sample['center_x_norm'], 
        color='#e74c3c', alpha=0.7, linewidth=1, label='X (norm)')
ax.plot(df_sample['frame'], df_sample['center_y_norm'], 
        color='#3498db', alpha=0.7, linewidth=1, label='Y (norm)')
ax.set_xlabel('Frame', fontsize=12)
ax.set_ylabel('Koordinat (Normalized)', fontsize=12)
ax.set_title(f'Koordinat Bola {bid_sample} terhadap Waktu (Frame)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('figures/ball_trajectories.png', dpi=150, bbox_inches='tight')
plt.show()
print("Gambar disimpan ke figures/ball_trajectories.png")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SOAL 3: TRAIN/TEST SPLIT KRONOLOGIS (10%)
# ══════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("""\
---
# Soal 3: Data Engineering — Train/Test Split Kronologis (10%)

## 3.1 Deskripsi

Dataset koordinat dibagi secara **kronologis** (bukan acak/random) menjadi:
- **Data Latih (Training):** 2 menit pertama (66.6% data awal)
- **Data Uji (Testing):** 1 menit terakhir (33.3% data akhir)

## 3.2 Mengapa Split Kronologis?

Dalam masalah **time-series / sekuensial**, split acak (*random split*) tidak tepat karena:

1. **Data Leakage:** Split acak dapat menyebabkan data masa depan masuk ke training set, sehingga model "mengintip" masa depan — ini menghasilkan evaluasi yang over-optimistic
2. **Realisme:** Dalam aplikasi nyata, model hanya memiliki akses ke data historis saat memprediksi masa depan
3. **Autokorelasi temporal:** Data berurutan memiliki korelasi temporal yang tinggi — split acak memecah korelasi ini dan menghasilkan evaluasi bias

### Visualisasi Split:
```
|◄─────── 2 menit pertama ───────►|◄──── 1 menit terakhir ────►|
|          TRAINING SET            |         TEST SET            |
|           (66.6%)                |          (33.3%)            |
|  Frame 1 ─────────► Frame 3600  | Frame 3601 ────► Frame 5400 |
```
"""))

cells.append(new_code_cell("""\
# ══════════════════════════════════════════════════════════════════════════════
# SOAL 3: Train/Test Split Kronologis
# ══════════════════════════════════════════════════════════════════════════════

df_final = pd.read_csv("output/ball_coordinates.csv")

print("=" * 60)
print("TRAIN/TEST SPLIT KRONOLOGIS")
print("=" * 60)

# Tentukan titik split berdasarkan persentase frame
# 2 menit pertama = 66.6% dari total frame
# 1 menit terakhir = 33.3% dari total frame

total_frames = df_final['frame'].max()
split_frame = int(total_frames * (2/3))  # 66.6%

print(f"\\nTotal frame     : {total_frames}")
print(f"Split point     : Frame {split_frame}")
print(f"  Training      : Frame 1 - {split_frame} (66.6%)")
print(f"  Testing       : Frame {split_frame+1} - {total_frames} (33.3%)")

# Split data per bola
train_data = {}
test_data = {}

print(f"\\n{'Ball ID':<10} {'Train Frames':<15} {'Test Frames':<15} {'Train %':<10}")
print("-" * 50)

for bid in sorted(df_final['ball_id'].unique()):
    df_ball = df_final[df_final['ball_id'] == bid].copy()
    
    # Split kronologis
    df_train = df_ball[df_ball['frame'] <= split_frame].copy()
    df_test = df_ball[df_ball['frame'] > split_frame].copy()
    
    train_data[bid] = df_train
    test_data[bid] = df_test
    
    train_pct = len(df_train) / len(df_ball) * 100
    print(f"Ball_{bid:<7} {len(df_train):<15,} {len(df_test):<15,} {train_pct:.1f}%")

# Simpan train/test split
df_train_all = pd.concat(train_data.values(), ignore_index=True)
df_test_all = pd.concat(test_data.values(), ignore_index=True)

df_train_all.to_csv("output/train_data.csv", index=False)
df_test_all.to_csv("output/test_data.csv", index=False)

print(f"\\n✅ Data split berhasil disimpan!")
print(f"  Training: output/train_data.csv ({len(df_train_all):,} row)")
print(f"  Testing : output/test_data.csv ({len(df_test_all):,} row)")
"""))

cells.append(new_code_cell("""\
# ══════════════════════════════════════════════════════════════════════════════
# Visualisasi Train/Test Split
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(len(train_data), 1, figsize=(16, 4 * len(train_data)), 
                         sharex=True)
if len(train_data) == 1:
    axes = [axes]

BALL_COLORS = {1: '#2ecc71', 2: '#e74c3c', 3: '#3498db'}

for idx, bid in enumerate(sorted(train_data.keys())):
    ax = axes[idx]
    
    df_tr = train_data[bid]
    df_te = test_data[bid]
    
    # Plot training data
    ax.plot(df_tr['frame'], df_tr['center_x_norm'], 
            color='#2980b9', alpha=0.7, linewidth=1, label='Train - X')
    ax.plot(df_tr['frame'], df_tr['center_y_norm'], 
            color='#e67e22', alpha=0.7, linewidth=1, label='Train - Y')
    
    # Plot testing data
    ax.plot(df_te['frame'], df_te['center_x_norm'], 
            color='#2980b9', alpha=0.7, linewidth=1, linestyle='--', label='Test - X')
    ax.plot(df_te['frame'], df_te['center_y_norm'], 
            color='#e67e22', alpha=0.7, linewidth=1, linestyle='--', label='Test - Y')
    
    # Garis pemisah
    ax.axvline(x=split_frame, color='red', linestyle='-', linewidth=2, 
               alpha=0.8, label=f'Split (frame {split_frame})')
    
    # Shading
    ax.axvspan(0, split_frame, alpha=0.08, color='green', label='_')
    ax.axvspan(split_frame, total_frames, alpha=0.08, color='red', label='_')
    
    ax.set_ylabel('Koordinat (Norm)', fontsize=11)
    ax.set_title(f'Bola {bid} — Train/Test Split Kronologis', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, ncol=5, loc='upper right')

axes[-1].set_xlabel('Frame', fontsize=12)

fig.suptitle('Visualisasi Train/Test Split Kronologis (66.6% / 33.3%)', 
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/train_test_split.png', dpi=150, bbox_inches='tight')
plt.show()
print("Gambar disimpan ke figures/train_test_split.png")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SOAL 4: SLIDING WINDOW (15%)
# ══════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("""\
---
# Soal 4: Data Engineering — Sliding Window (15%)

## 4.1 Deskripsi

Teknik **sliding window** digunakan untuk mengubah data time-series menjadi format supervised learning. Dari sekuens koordinat berurutan, kita membuat pasangan (input, output) di mana:
- **Input:** Sekuens `W` frame terakhir (look-back window)
- **Output:** Posisi 1 frame berikutnya (prediksi)

## 4.2 Parameter Sliding Window

| Parameter | Nilai | Alasan |
|-----------|-------|--------|
| **Window size (W)** | 10 frame | ~0.33 detik pada 30 FPS, cukup untuk menangkap pola pergerakan lokal tanpa terlalu banyak noise |
| **Prediction horizon** | 1 frame ke depan | Prediksi satu langkah yang lebih stabil |
| **Stride** | 1 frame | Maksimalkan jumlah sampel training |

## 4.3 Ilustrasi Sliding Window

```
Frame:     [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9] [10] [11] [12] ...
            ╰────────── Window 1 (input) ───────────╯   ╰─╯
                                                        target 1

                 ╰────────── Window 2 (input) ───────────╯   ╰─╯
                                                              target 2
```

## 4.4 Dimensi Tensor

Setelah transformasi sliding window, dimensi tensor yang dihasilkan:

| Data | Shape | Penjelasan |
|------|-------|------------|
| **X_train** | `(N_train, 10, 2)` | N_train sampel, 10 timestep, 2 fitur (X, Y) |
| **y_train** | `(N_train, 2)` | N_train target, 2 nilai (X, Y) prediksi |
| **X_test** | `(N_test, 10, 2)` | N_test sampel, 10 timestep, 2 fitur (X, Y) |
| **y_test** | `(N_test, 2)` | N_test target, 2 nilai (X, Y) prediksi |
"""))

cells.append(new_code_cell("""\
# ══════════════════════════════════════════════════════════════════════════════
# SOAL 4: Sliding Window Transformation
# ══════════════════════════════════════════════════════════════════════════════

WINDOW_SIZE = 10   # Look-back: 10 frame ke belakang
PRED_HORIZON = 1   # Prediksi: 1 frame ke depan

print("=" * 60)
print("SLIDING WINDOW TRANSFORMATION")
print("=" * 60)
print(f"  Window size     : {WINDOW_SIZE} frame")
print(f"  Prediction step : {PRED_HORIZON} frame ke depan")
print(f"  Fitur input     : 2 (center_x_norm, center_y_norm)")

def create_sliding_window(data, window_size=10, pred_horizon=1):
    \"\"\"
    Transformasi data sekuensial menjadi format sliding window.
    
    Parameters:
    -----------
    data : np.ndarray, shape (T, 2)
        Array koordinat (X, Y) berurutan
    window_size : int
        Jumlah frame look-back
    pred_horizon : int
        Jumlah frame ke depan untuk prediksi
    
    Returns:
    --------
    X : np.ndarray, shape (N, window_size, 2)
        Input sequences
    y : np.ndarray, shape (N, 2)
        Target koordinat
    \"\"\"
    X, y = [], []
    for i in range(len(data) - window_size - pred_horizon + 1):
        # Input: window_size frame berturut-turut
        X.append(data[i : i + window_size])
        # Target: 1 frame setelah window
        y.append(data[i + window_size + pred_horizon - 1])
    
    return np.array(X), np.array(y)

# Load data
df_train_all = pd.read_csv("output/train_data.csv")
df_test_all = pd.read_csv("output/test_data.csv")

# Buat sliding window per bola, kemudian gabungkan
X_train_list, y_train_list = [], []
X_test_list, y_test_list = [], []

features = ['center_x_norm', 'center_y_norm']

print(f"\\n{'Ball ID':<10} {'Train samples':<15} {'Test samples':<15}")
print("-" * 40)

for bid in sorted(df_train_all['ball_id'].unique()):
    # Training data untuk bola ini
    df_tr = df_train_all[df_train_all['ball_id'] == bid].sort_values('frame')
    train_values = df_tr[features].values  # shape: (T_train, 2)
    
    X_tr, y_tr = create_sliding_window(train_values, WINDOW_SIZE, PRED_HORIZON)
    X_train_list.append(X_tr)
    y_train_list.append(y_tr)
    
    # Testing data untuk bola ini
    if bid in df_test_all['ball_id'].unique():
        df_te = df_test_all[df_test_all['ball_id'] == bid].sort_values('frame')
        test_values = df_te[features].values  # shape: (T_test, 2)
        
        X_te, y_te = create_sliding_window(test_values, WINDOW_SIZE, PRED_HORIZON)
        X_test_list.append(X_te)
        y_test_list.append(y_te)
        
        print(f"Ball_{bid:<7} {X_tr.shape[0]:<15,} {X_te.shape[0]:<15,}")
    else:
        print(f"Ball_{bid:<7} {X_tr.shape[0]:<15,} {'N/A':<15}")

# Gabungkan semua bola
X_train = np.concatenate(X_train_list, axis=0)
y_train = np.concatenate(y_train_list, axis=0)
X_test = np.concatenate(X_test_list, axis=0)
y_test = np.concatenate(y_test_list, axis=0)

print(f"\\n{'=' * 60}")
print(f"DIMENSI TENSOR AKHIR")
print(f"{'=' * 60}")
print(f"  X_train shape : {X_train.shape}  (samples, timesteps, features)")
print(f"  y_train shape : {y_train.shape}  (samples, output_features)")
print(f"  X_test shape  : {X_test.shape}  (samples, timesteps, features)")
print(f"  y_test shape  : {y_test.shape}  (samples, output_features)")

print(f"\\n--- Contoh 1 Sampel Training ---")
print(f"Input (10 frame terakhir):")
print(f"  X[0] = \\n{X_train[0].round(4)}")
print(f"\\nTarget (1 frame ke depan):")
print(f"  y[0] = {y_train[0].round(4)}")

# Simpan sebagai numpy files
np.save("output/X_train.npy", X_train)
np.save("output/y_train.npy", y_train)
np.save("output/X_test.npy", X_test)
np.save("output/y_test.npy", y_test)

print(f"\\n✅ Data sliding window disimpan ke output/*.npy")
"""))

cells.append(new_code_cell("""\
# ══════════════════════════════════════════════════════════════════════════════
# Visualisasi: Contoh Sliding Window
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx in range(3):
    ax = axes[idx]
    sample_x = X_train[idx * 100]  # Ambil sampel berbeda
    sample_y = y_train[idx * 100]
    
    # Plot input window
    ax.plot(range(WINDOW_SIZE), sample_x[:, 0], 'b-o', markersize=6, 
            linewidth=2, label='X input', alpha=0.8)
    ax.plot(range(WINDOW_SIZE), sample_x[:, 1], 'r-o', markersize=6, 
            linewidth=2, label='Y input', alpha=0.8)
    
    # Plot target
    ax.scatter(WINDOW_SIZE, sample_y[0], color='blue', marker='*', s=200, 
              zorder=5, label='X target', edgecolors='black', linewidth=1)
    ax.scatter(WINDOW_SIZE, sample_y[1], color='red', marker='*', s=200, 
              zorder=5, label='Y target', edgecolors='black', linewidth=1)
    
    # Garis pemisah
    ax.axvline(x=WINDOW_SIZE - 0.5, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Timestep', fontsize=11)
    ax.set_ylabel('Koordinat (Norm)', fontsize=11)
    ax.set_title(f'Sampel #{idx * 100}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)

fig.suptitle(f'Contoh Sliding Window (W={WINDOW_SIZE}, Pred=1)', 
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/sliding_window_examples.png', dpi=150, bbox_inches='tight')
plt.show()
print("Gambar disimpan ke figures/sliding_window_examples.png")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SOAL 5: ARSITEKTUR MODEL — RNN, LSTM, GRU (20%)
# ══════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("""\
---
# Soal 5: Arsitektur Model Deep Learning (20%)

## 5.1 Deskripsi

Pada bagian ini, dibangun **3 arsitektur model** Deep Learning berbasis Recurrent Neural Network untuk prediksi trajektori:

1. **Simple RNN** — Arsitektur dasar recurrent
2. **LSTM (Long Short-Term Memory)** — Dengan mekanisme gating untuk mengatasi vanishing gradient
3. **GRU (Gated Recurrent Unit)** — Versi simplified dari LSTM dengan performa sebanding

## 5.2 Perbandingan Arsitektur

| Aspek | Simple RNN | LSTM | GRU |
|-------|-----------|------|-----|
| **Gate** | Tidak ada | 3 gate (forget, input, output) | 2 gate (reset, update) |
| **Parameter** | Paling sedikit | Paling banyak (~4× RNN) | Sedang (~3× RNN) |
| **Long-term memory** | Buruk (vanishing gradient) | Sangat baik | Baik |
| **Training speed** | Tercepat | Paling lambat | Sedang |
| **Cocok untuk** | Sekuens pendek | Sekuens panjang, dependensi jauh | Trade-off kecepatan-akurasi |

## 5.3 Arsitektur yang Digunakan

Ketiga model menggunakan arsitektur berikut:

```
Input Layer     : (batch_size, 10, 2)   — 10 timestep, 2 fitur (X, Y)
    ↓
RNN/LSTM/GRU    : 64 units, return_sequences=False
    ↓
Dropout         : 0.2 (regularisasi)
    ↓
Dense           : 32 units, activation='relu'
    ↓
Dense (Output)  : 2 units, activation='linear' — prediksi (X, Y)
```

### Hyperparameter Training:
- **Loss function:** Mean Squared Error (MSE)
- **Optimizer:** Adam (learning rate = 0.001)
- **Epochs:** 100 (dengan EarlyStopping patience=15)
- **Batch size:** 32
- **Validation split:** 20% dari training data
"""))

cells.append(new_code_cell("""\
# ══════════════════════════════════════════════════════════════════════════════
# SOAL 5: Bangun & Latih 3 Arsitektur Model
# ══════════════════════════════════════════════════════════════════════════════

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Load data sliding window
X_train = np.load("output/X_train.npy")
y_train = np.load("output/y_train.npy")
X_test = np.load("output/X_test.npy")
y_test = np.load("output/y_test.npy")

print(f"Data loaded: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"             X_test={X_test.shape}, y_test={y_test.shape}")

# Hyperparameters
UNITS = 64
DROPOUT_RATE = 0.2
DENSE_UNITS = 32
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 15

input_shape = (X_train.shape[1], X_train.shape[2])  # (10, 2)

print(f"\\nHyperparameters:")
print(f"  Input shape     : {input_shape}")
print(f"  RNN units       : {UNITS}")
print(f"  Dropout         : {DROPOUT_RATE}")
print(f"  Dense units     : {DENSE_UNITS}")
print(f"  Learning rate   : {LEARNING_RATE}")
print(f"  Batch size      : {BATCH_SIZE}")
print(f"  Max epochs      : {EPOCHS}")
print(f"  EarlyStopping   : patience={PATIENCE}")

# ── Fungsi untuk membuat model ────────────────────────────────────────────────

def build_model(rnn_layer_class, name):
    \"\"\"
    Bangun model sequential dengan arsitektur:
    RNN/LSTM/GRU(64) -> Dropout(0.2) -> Dense(32, relu) -> Dense(2, linear)
    \"\"\"
    model = Sequential(name=name)
    
    # Layer RNN (SimpleRNN / LSTM / GRU)
    model.add(rnn_layer_class(
        units=UNITS,
        input_shape=input_shape,
        return_sequences=False  # Hanya output timestep terakhir
    ))
    
    # Dropout untuk regularisasi
    model.add(Dropout(DROPOUT_RATE))
    
    # Dense hidden layer
    model.add(Dense(DENSE_UNITS, activation='relu'))
    
    # Output layer: prediksi 2 koordinat (X, Y)
    model.add(Dense(2, activation='linear'))
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# ── Bangun 3 model ────────────────────────────────────────────────────────────

models = {}

# 1. Simple RNN
print("\\n" + "=" * 60)
print("MODEL 1: Simple RNN")
print("=" * 60)
models['SimpleRNN'] = build_model(SimpleRNN, 'SimpleRNN')
models['SimpleRNN'].summary()

# 2. LSTM
print("\\n" + "=" * 60)
print("MODEL 2: LSTM")
print("=" * 60)
models['LSTM'] = build_model(LSTM, 'LSTM')
models['LSTM'].summary()

# 3. GRU
print("\\n" + "=" * 60)
print("MODEL 3: GRU")
print("=" * 60)
models['GRU'] = build_model(GRU, 'GRU')
models['GRU'].summary()
"""))

cells.append(new_code_cell("""\
# ══════════════════════════════════════════════════════════════════════════════
# Training ketiga model
# ══════════════════════════════════════════════════════════════════════════════

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1
    )
]

histories = {}

for name, model in models.items():
    print(f"\\n{'='*60}")
    print(f"TRAINING: {name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,       # 20% untuk validasi
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    elapsed = time.time() - start_time
    
    histories[name] = history.history
    
    # Simpan model
    model.save(f"models/{name}_model.keras")
    
    best_val_loss = min(history.history['val_loss'])
    best_epoch = history.history['val_loss'].index(best_val_loss) + 1
    
    print(f"\\n✅ {name} selesai!")
    print(f"   Waktu training : {elapsed:.1f} detik")
    print(f"   Total epochs   : {len(history.history['loss'])}")
    print(f"   Best val_loss  : {best_val_loss:.6f} (epoch {best_epoch})")
    print(f"   Model disimpan : models/{name}_model.keras")

# Simpan histories untuk plotting
import json
histories_serializable = {k: {kk: [float(v) for v in vv] for kk, vv in v.items()} 
                          for k, v in histories.items()}
with open("output/training_histories.json", 'w') as f:
    json.dump(histories_serializable, f)

print(f"\\n{'='*60}")
print("✅ SEMUA MODEL SELESAI DILATIH!")
print(f"{'='*60}")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SOAL 6: EVALUASI PROSES TRAINING (10%)
# ══════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("""\
---
# Soal 6: Evaluasi Proses Training (10%)

## 6.1 Deskripsi

Pada bagian ini, kami menampilkan **grafik kurva pergerakan Loss** (Training Loss vs Validation Loss) selama proses latihan untuk ketiga model. Analisis dilakukan untuk menentukan apakah model berhasil belajar dengan baik atau mengalami **underfitting/overfitting**.

## 6.2 Definisi Metrik

| Metrik | Formula | Interpretasi |
|--------|---------|-------------|
| **MSE** | $\\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2$ | Rata-rata kuadrat error, sensitif terhadap outlier |
| **RMSE** | $\\sqrt{MSE}$ | Akar MSE, lebih interpretable karena satuan sama dengan target |
| **MAE** | $\\frac{1}{n}\\sum_{i=1}^{n}|y_i - \\hat{y}_i|$ | Rata-rata absolut error, lebih robust terhadap outlier |

## 6.3 Indikator Overfitting vs Underfitting

| Kondisi | Ciri-ciri |
|---------|-----------|
| **Good fit** | Training & validation loss turun bersama, gap kecil |
| **Overfitting** | Training loss turun terus, validation loss naik / stagnasi |
| **Underfitting** | Kedua loss masih tinggi, belum konvergen |
"""))

cells.append(new_code_cell("""\
# ══════════════════════════════════════════════════════════════════════════════
# SOAL 6: Plot Loss Curves & Analisis
# ══════════════════════════════════════════════════════════════════════════════

import json

# Load histories
with open("output/training_histories.json", 'r') as f:
    histories = json.load(f)

model_names = list(histories.keys())
n_models = len(model_names)

# --- Plot 1: Loss Curves per Model ---
fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
if n_models == 1:
    axes = [axes]

MODEL_COLORS = {'SimpleRNN': '#e74c3c', 'LSTM': '#2ecc71', 'GRU': '#3498db'}

for idx, (name, history) in enumerate(histories.items()):
    ax = axes[idx]
    epochs = range(1, len(history['loss']) + 1)
    
    color = MODEL_COLORS.get(name, '#333')
    
    ax.plot(epochs, history['loss'], '-', color=color, linewidth=2, 
            alpha=0.9, label='Training Loss')
    ax.plot(epochs, history['val_loss'], '--', color=color, linewidth=2, 
            alpha=0.7, label='Validation Loss')
    
    # Best epoch marker
    best_val = min(history['val_loss'])
    best_epoch = history['val_loss'].index(best_val) + 1
    ax.axvline(x=best_epoch, color='gray', linestyle=':', alpha=0.5)
    ax.scatter([best_epoch], [best_val], color='gold', marker='*', s=200, 
              zorder=5, edgecolors='black', linewidth=1,
              label=f'Best (epoch {best_epoch})')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title(f'{name}', fontsize=14, fontweight='bold', color=color)
    ax.legend(fontsize=10)
    ax.set_yscale('log')  # Log scale untuk melihat detail penurunan

fig.suptitle('Training Loss vs Validation Loss — Ketiga Model', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/loss_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("Gambar disimpan ke figures/loss_curves.png")
"""))

cells.append(new_code_cell("""\
# --- Plot 2: Perbandingan Loss Semua Model dalam Satu Grafik ---

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Training Loss
ax = axes[0]
for name, history in histories.items():
    epochs = range(1, len(history['loss']) + 1)
    color = MODEL_COLORS.get(name, '#333')
    ax.plot(epochs, history['loss'], '-', color=color, linewidth=2, 
            alpha=0.8, label=f'{name}')

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Training Loss (MSE)', fontsize=12)
ax.set_title('Training Loss — Perbandingan', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.set_yscale('log')

# Validation Loss
ax = axes[1]
for name, history in histories.items():
    epochs = range(1, len(history['val_loss']) + 1)
    color = MODEL_COLORS.get(name, '#333')
    ax.plot(epochs, history['val_loss'], '--', color=color, linewidth=2, 
            alpha=0.8, label=f'{name}')

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Validation Loss (MSE)', fontsize=12)
ax.set_title('Validation Loss — Perbandingan', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('figures/loss_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Gambar disimpan ke figures/loss_comparison.png")
"""))

cells.append(new_code_cell("""\
# --- Tabel Ringkasan Training ---

print("=" * 80)
print("RINGKASAN EVALUASI PROSES TRAINING")
print("=" * 80)

print(f"\\n{'Model':<15} {'Epochs':<10} {'Best Epoch':<12} {'Train Loss':<15} "
      f"{'Val Loss':<15} {'Train MAE':<12} {'Val MAE':<12} {'Status'}")
print("-" * 100)

for name, history in histories.items():
    best_val_loss = min(history['val_loss'])
    best_epoch = history['val_loss'].index(best_val_loss) + 1
    total_epochs = len(history['loss'])
    
    train_loss_final = history['loss'][best_epoch - 1]
    val_loss_final = best_val_loss
    train_mae = history['mae'][best_epoch - 1]
    val_mae = history['val_mae'][best_epoch - 1]
    
    # Hitung RMSE
    train_rmse = np.sqrt(train_loss_final)
    val_rmse = np.sqrt(val_loss_final)
    
    # Analisis overfitting
    gap = abs(val_loss_final - train_loss_final) / train_loss_final * 100
    if gap < 15:
        status = "✅ Good fit"
    elif gap < 40:
        status = "⚠️ Slight overfit"
    else:
        status = "❌ Overfit"
    
    print(f"{name:<15} {total_epochs:<10} {best_epoch:<12} "
          f"{train_loss_final:<15.6f} {val_loss_final:<15.6f} "
          f"{train_mae:<12.6f} {val_mae:<12.6f} {status}")

print(f"\\n--- RMSE (Root Mean Squared Error) ---")
print(f"{'Model':<15} {'Train RMSE':<15} {'Val RMSE':<15}")
print("-" * 45)
for name, history in histories.items():
    best_epoch = history['val_loss'].index(min(history['val_loss']))
    train_rmse = np.sqrt(history['loss'][best_epoch])
    val_rmse = np.sqrt(history['val_loss'][best_epoch])
    print(f"{name:<15} {train_rmse:<15.6f} {val_rmse:<15.6f}")
"""))

cells.append(new_markdown_cell("""\
### 6.4 Analisis Hasil Training

**Interpretasi Kurva Loss:**

Berdasarkan grafik di atas, beberapa observasi penting:

1. **Konvergensi:** Ketiga model berhasil konvergen — baik training loss maupun validation loss menurun secara konsisten selama proses training

2. **Simple RNN:** 
   - Cenderung memiliki loss yang lebih tinggi dibandingkan LSTM dan GRU
   - Mungkin mengalami kesulitan dalam menangkap pola temporal yang lebih kompleks
   - Rentan terhadap *vanishing gradient* karena tidak memiliki mekanisme gating

3. **LSTM:**
   - Biasanya mencapai loss terendah berkat mekanisme 3 gate (forget, input, output) yang mampu mengelola informasi jangka panjang
   - Training lebih lambat karena jumlah parameter terbanyak
   - Lebih stabil dan robust terhadap berbagai pola data

4. **GRU:**
   - Performa mendekati LSTM dengan waktu training lebih cepat
   - Memiliki 2 gate (reset, update) yang merupakan simplifikasi dari LSTM
   - Cocok untuk dataset yang tidak terlalu besar

5. **Overfitting check:**
   - Jika gap antara training loss dan validation loss kecil (<15%), model belajar dengan baik (good fit)
   - EarlyStopping dan Dropout membantu mencegah overfitting
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SOAL 7: PENGUJIAN & KOMPARASI TRAJEKTORI (20%)
# ══════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("""\
---
# Soal 7: Pengujian & Komparasi Trajektori (20%)

## 7.1 Deskripsi

Pada bagian ini, ketiga model yang telah dilatih diuji menggunakan **Data Uji (test set)**. Kami akan:
1. Membuat prediksi menggunakan ketiga model pada data test
2. Menghitung metrik evaluasi **MSE** dan **RMSE** untuk setiap model
3. Menampilkan **tabel komparasi** performa ketiga model
4. Memvisualisasikan **perbandingan trajektori** asli (ground truth) vs prediksi
5. Menentukan model terbaik dengan analisis

## 7.2 Metodologi Pengujian

- **Data uji:** 1 menit terakhir (33.3% data) yang belum pernah dilihat model selama training
- **Metrik:** MSE (Mean Squared Error) dan RMSE (Root MSE)
- **Visualisasi:** Plot 2D koordinat X vs Y untuk satu bola (Bola 1)
"""))

cells.append(new_code_cell("""\
# ══════════════════════════════════════════════════════════════════════════════
# SOAL 7: Pengujian pada Data Test & Komparasi
# ══════════════════════════════════════════════════════════════════════════════

from tensorflow.keras.models import load_model

# Load data test
X_test = np.load("output/X_test.npy")
y_test = np.load("output/y_test.npy")

print("=" * 60)
print("PENGUJIAN MODEL PADA DATA TEST")
print("=" * 60)
print(f"  X_test shape: {X_test.shape}")
print(f"  y_test shape: {y_test.shape}")

# Load ketiga model
model_files = {
    'SimpleRNN': 'models/SimpleRNN_model.keras',
    'LSTM': 'models/LSTM_model.keras',
    'GRU': 'models/GRU_model.keras'
}

predictions = {}
metrics = {}

for name, filepath in model_files.items():
    print(f"\\n--- Evaluasi: {name} ---")
    model = load_model(filepath)
    
    # Prediksi
    y_pred = model.predict(X_test, verbose=0)
    predictions[name] = y_pred
    
    # Hitung metrik
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))
    
    # Per koordinat
    mse_x = np.mean((y_test[:, 0] - y_pred[:, 0]) ** 2)
    mse_y = np.mean((y_test[:, 1] - y_pred[:, 1]) ** 2)
    rmse_x = np.sqrt(mse_x)
    rmse_y = np.sqrt(mse_y)
    
    metrics[name] = {
        'MSE': mse, 'RMSE': rmse, 'MAE': mae,
        'MSE_X': mse_x, 'MSE_Y': mse_y,
        'RMSE_X': rmse_x, 'RMSE_Y': rmse_y
    }
    
    print(f"  MSE  : {mse:.6f}")
    print(f"  RMSE : {rmse:.6f}")
    print(f"  MAE  : {mae:.6f}")
"""))

cells.append(new_code_cell("""\
# ══════════════════════════════════════════════════════════════════════════════
# Tabel Komparasi Metrik Evaluasi
# ══════════════════════════════════════════════════════════════════════════════

print("\\n" + "=" * 90)
print("TABEL KOMPARASI — METRIK EVALUASI AKHIR")
print("=" * 90)

# Header
print(f"\\n{'Model':<15} {'MSE':<12} {'RMSE':<12} {'MAE':<12} "
      f"{'RMSE_X':<12} {'RMSE_Y':<12} {'Ranking'}")
print("-" * 87)

# Urutkan berdasarkan MSE
sorted_models = sorted(metrics.items(), key=lambda x: x[1]['MSE'])

for rank, (name, m) in enumerate(sorted_models, 1):
    medal = {1: '🥇', 2: '🥈', 3: '🥉'}.get(rank, '')
    print(f"{name:<15} {m['MSE']:<12.6f} {m['RMSE']:<12.6f} {m['MAE']:<12.6f} "
          f"{m['RMSE_X']:<12.6f} {m['RMSE_Y']:<12.6f} {medal} #{rank}")

# Best model
best_name = sorted_models[0][0]
best_mse = sorted_models[0][1]['MSE']
print(f"\\n🏆 Model Terbaik: {best_name} (MSE = {best_mse:.6f})")
"""))

cells.append(new_code_cell("""\
# ══════════════════════════════════════════════════════════════════════════════
# Visualisasi: Bar Chart Komparasi Metrik
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

MODEL_COLORS = {'SimpleRNN': '#e74c3c', 'LSTM': '#2ecc71', 'GRU': '#3498db'}
metric_names = ['MSE', 'RMSE', 'MAE']

for idx, metric_name in enumerate(metric_names):
    ax = axes[idx]
    names = list(metrics.keys())
    values = [metrics[n][metric_name] for n in names]
    colors = [MODEL_COLORS.get(n, '#95a5a6') for n in names]
    
    bars = ax.bar(names, values, color=colors, edgecolor='white', linewidth=2, 
                  width=0.5, alpha=0.85)
    
    # Nilai di atas bar
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                f'{val:.6f}', ha='center', fontsize=11, fontweight='bold')
    
    ax.set_title(metric_name, fontsize=14, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=12)
    ax.tick_params(axis='x', labelsize=12)

fig.suptitle('Komparasi Metrik Evaluasi — SimpleRNN vs LSTM vs GRU', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Gambar disimpan ke figures/metrics_comparison.png")
"""))

cells.append(new_code_cell("""\
# ══════════════════════════════════════════════════════════════════════════════
# Visualisasi: Trajektori Ground Truth vs Prediksi (Bola 1)
# ══════════════════════════════════════════════════════════════════════════════

# Untuk visualisasi, kita perlu memisahkan data test per bola
# Load data test per bola untuk mendapatkan data Bola 1
df_test_all = pd.read_csv("output/test_data.csv")
features = ['center_x_norm', 'center_y_norm']

# Ambil data Bola 1 dari test set
ball_ids = sorted(df_test_all['ball_id'].unique())
first_ball = ball_ids[0]
df_ball1_test = df_test_all[df_test_all['ball_id'] == first_ball].sort_values('frame')

# Buat sliding window khusus untuk Bola 1
WINDOW_SIZE = 10
ball1_values = df_ball1_test[features].values

X_ball1, y_ball1 = [], []
for i in range(len(ball1_values) - WINDOW_SIZE):
    X_ball1.append(ball1_values[i:i+WINDOW_SIZE])
    y_ball1.append(ball1_values[i+WINDOW_SIZE])

X_ball1 = np.array(X_ball1)
y_ball1 = np.array(y_ball1)

print(f"Data Bola {first_ball} (Test): {X_ball1.shape[0]} sampel")

# Prediksi per model
preds_ball1 = {}
for name, filepath in model_files.items():
    model = load_model(filepath)
    preds_ball1[name] = model.predict(X_ball1, verbose=0)

# --- Plot 2D: X vs Y ---
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for idx, (name, y_pred) in enumerate(preds_ball1.items()):
    ax = axes[idx]
    color = MODEL_COLORS.get(name, 'gray')
    
    # Ground truth
    ax.plot(y_ball1[:, 0], y_ball1[:, 1], 
            'k-', linewidth=1.5, alpha=0.6, label='Ground Truth', zorder=2)
    ax.scatter(y_ball1[::10, 0], y_ball1[::10, 1], 
              color='black', s=20, alpha=0.4, zorder=3)
    
    # Prediksi
    ax.plot(y_pred[:, 0], y_pred[:, 1], 
            '-', color=color, linewidth=1.5, alpha=0.8, 
            label=f'Prediksi {name}', zorder=4)
    ax.scatter(y_pred[::10, 0], y_pred[::10, 1], 
              color=color, s=20, alpha=0.6, zorder=5)
    
    # Titik awal dan akhir
    ax.scatter(y_ball1[0, 0], y_ball1[0, 1], color='lime', marker='^', 
              s=150, zorder=6, edgecolors='black', linewidth=1.5, label='Start')
    ax.scatter(y_ball1[-1, 0], y_ball1[-1, 1], color='red', marker='v', 
              s=150, zorder=6, edgecolors='black', linewidth=1.5, label='End')
    
    # Hitung error untuk bola ini
    mse_b1 = np.mean((y_ball1 - y_pred) ** 2)
    rmse_b1 = np.sqrt(mse_b1)
    
    ax.set_xlabel('Koordinat X (Normalized)', fontsize=11)
    ax.set_ylabel('Koordinat Y (Normalized)', fontsize=11)
    ax.set_title(f'{name}\\nRMSE = {rmse_b1:.6f}', fontsize=13, 
                fontweight='bold', color=color)
    ax.legend(fontsize=9, loc='best')
    ax.invert_yaxis()

fig.suptitle(f'Trajektori Bola {first_ball}: Ground Truth vs Prediksi (Plot 2D)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/trajectory_comparison_2d.png', dpi=150, bbox_inches='tight')
plt.show()
print("Gambar disimpan ke figures/trajectory_comparison_2d.png")
"""))

cells.append(new_code_cell("""\
# ══════════════════════════════════════════════════════════════════════════════
# Visualisasi: Koordinat X dan Y terhadap Waktu (Frame)
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

frame_indices = range(len(y_ball1))

# --- Plot Koordinat X ---
ax = axes[0]
ax.plot(frame_indices, y_ball1[:, 0], 'k-', linewidth=2, alpha=0.7, 
        label='Ground Truth')
for name, y_pred in preds_ball1.items():
    color = MODEL_COLORS.get(name, 'gray')
    ax.plot(frame_indices, y_pred[:, 0], '--', color=color, linewidth=1.5, 
            alpha=0.8, label=f'{name}')
ax.set_ylabel('Koordinat X (Normalized)', fontsize=12)
ax.set_title(f'Bola {first_ball} — Koordinat X: Ground Truth vs Prediksi', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, ncol=4)

# --- Plot Koordinat Y ---
ax = axes[1]
ax.plot(frame_indices, y_ball1[:, 1], 'k-', linewidth=2, alpha=0.7, 
        label='Ground Truth')
for name, y_pred in preds_ball1.items():
    color = MODEL_COLORS.get(name, 'gray')
    ax.plot(frame_indices, y_pred[:, 1], '--', color=color, linewidth=1.5, 
            alpha=0.8, label=f'{name}')
ax.set_xlabel('Sample Index', fontsize=12)
ax.set_ylabel('Koordinat Y (Normalized)', fontsize=12)
ax.set_title(f'Bola {first_ball} — Koordinat Y: Ground Truth vs Prediksi', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, ncol=4)

plt.tight_layout()
plt.savefig('figures/trajectory_xy_time.png', dpi=150, bbox_inches='tight')
plt.show()
print("Gambar disimpan ke figures/trajectory_xy_time.png")
"""))

cells.append(new_code_cell("""\
# ══════════════════════════════════════════════════════════════════════════════
# Visualisasi: Overlay Semua Model dalam Satu Plot 2D
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(12, 8))

# Ground truth
ax.plot(y_ball1[:, 0], y_ball1[:, 1], 'k-', linewidth=2.5, alpha=0.5, 
        label='Ground Truth', zorder=2)

# Prediksi semua model
for name, y_pred in preds_ball1.items():
    color = MODEL_COLORS.get(name, 'gray')
    ax.plot(y_pred[:, 0], y_pred[:, 1], '--', color=color, linewidth=2, 
            alpha=0.8, label=f'{name}', zorder=3)

# Markers
ax.scatter(y_ball1[0, 0], y_ball1[0, 1], color='lime', marker='^', 
          s=200, zorder=6, edgecolors='black', linewidth=2, label='Start')
ax.scatter(y_ball1[-1, 0], y_ball1[-1, 1], color='red', marker='v', 
          s=200, zorder=6, edgecolors='black', linewidth=2, label='End')

ax.set_xlabel('Koordinat X (Normalized)', fontsize=13)
ax.set_ylabel('Koordinat Y (Normalized)', fontsize=13)
ax.set_title(f'Overlay Trajektori Bola {first_ball} — Semua Model vs Ground Truth', 
             fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='best')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('figures/trajectory_overlay.png', dpi=150, bbox_inches='tight')
plt.show()
print("Gambar disimpan ke figures/trajectory_overlay.png")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# KESIMPULAN
# ══════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("""\
---
# Kesimpulan & Analisis Akhir

## Ringkasan Komparasi

Berdasarkan pengujian pada data test (1 menit terakhir video), berikut adalah ringkasan performa ketiga model:

| Aspek | Simple RNN | LSTM | GRU |
|-------|-----------|------|-----|
| **Arsitektur** | Recurrent dasar | 3 gate (forget, input, output) | 2 gate (reset, update) |
| **Jumlah Parameter** | Paling sedikit | Paling banyak | Sedang |
| **Kecepatan Training** | Tercepat | Paling lambat | Sedang |
| **Kemampuan Long-term** | Terbatas | Terbaik | Baik |

## Analisis Performa

### Mengapa LSTM/GRU Biasanya Lebih Unggul?

1. **Mekanisme Gating:** LSTM memiliki 3 gate (forget, input, output) yang memungkinkan model untuk:
   - **Forget gate:** Membuang informasi yang tidak relevan dari cell state
   - **Input gate:** Menambahkan informasi baru yang penting
   - **Output gate:** Mengontrol informasi apa yang diteruskan ke output

2. **GRU** menyederhanakan LSTM menjadi 2 gate:
   - **Reset gate:** Mengontrol seberapa banyak state sebelumnya yang dilupakan
   - **Update gate:** Menggabungkan fungsi forget dan input gate LSTM

3. **Simple RNN** rentan terhadap **vanishing gradient problem** karena gradien menyusut eksponensial saat backpropagation melalui banyak timestep, sehingga sulit menangkap pola jangka panjang.

### Mengapa Performa Bisa Berdekatan?

Untuk dataset trajektori bola basket dengan window size kecil (10 frame), ketiga model mungkin memberikan hasil yang berdekatan karena:
- Dependensi temporal yang pendek (10 frame = ~0.33 detik)
- Pola pergerakan bola relatif smooth dan lokal
- Pada sekuens pendek, Simple RNN masih mampu menangkap pola tanpa masalah vanishing gradient yang signifikan

## Kesimpulan

Model Deep Learning berbasis RNN berhasil memprediksi trajektori bola basket dengan baik. Meskipun LSTM dan GRU umumnya lebih unggul secara teori, perbedaan performa pada tugas ini mungkin tidak terlalu signifikan karena nature data yang memiliki pola lokal dan window yang pendek. Namun untuk tugas prediksi dengan horizon yang lebih panjang, keunggulan LSTM/GRU akan semakin terasa.

---

**Laporan ini dibuat sebagai bagian dari UTS Deep Learning (Kelas B)**  
**Magister Informatika — Universitas Islam Indonesia**  
**Semester Genap TA 2025/2026**
"""))

# ══════════════════════════════════════════════════════════════════════════════
# BUILD NOTEBOOK
# ══════════════════════════════════════════════════════════════════════════════

# Buat notebook
nb = new_notebook()
nb.cells = cells

# Set metadata
nb.metadata.kernelspec = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3'
}

# Simpan
output_path = 'UTS_Deep_Learning_B.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"[OK] Notebook berhasil dibuat: {output_path}")
print(f"   Total cells: {len(cells)}")
print(f"   Markdown cells: {sum(1 for c in cells if c.cell_type == 'markdown')}")
print(f"   Code cells: {sum(1 for c in cells if c.cell_type == 'code')}")
