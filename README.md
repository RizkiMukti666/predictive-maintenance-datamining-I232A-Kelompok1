# Prediksi Kegagalan Mesin — Proyek Akhir Praktikum Data Mining 2026

Aplikasi web interaktif untuk memprediksi risiko kegagalan mesin industri (predictive
maintenance) berdasarkan data sensor, menggunakan algoritma **Decision Tree**.

## Struktur File

```
maintenance_app/
├── app.py              # Aplikasi utama Streamlit
├── requirements.txt    # Daftar dependency Python
├── ai4i2020.csv         # Dataset contoh (10.000 baris)
└── README.md
```

## Cara Menjalankan

1. Install dependency:
   ```
   pip install -r requirements.txt
   ```

2. Jalankan aplikasi:
   ```
   streamlit run app.py
   ```

3. Aplikasi akan terbuka otomatis di browser pada `http://localhost:8501`

## Fitur Aplikasi

- **Upload Dataset (.csv)** — pengguna bisa upload dataset sendiri (format sama) atau
  memakai dataset contoh bawaan
- **Ringkasan Data** — pratinjau tabel, jumlah baris/kolom, missing values, distribusi kelas
  (termasuk peringatan otomatis soal ketidakseimbangan kelas)
- **Eksplorasi Data (EDA)** — distribusi tiap fitur sensor berdasarkan status mesin,
  matriks korelasi antar fitur
- **Model & Evaluasi** — akurasi train/test dibandingkan baseline naif, confusion matrix,
  feature importance, laporan klasifikasi lengkap (precision/recall/F1), opsi
  `class_weight='balanced'`
- **Prediksi Manual** — slider interaktif untuk memasukkan nilai sensor dan memprediksi
  status mesin (Normal/Gagal) secara real-time, lengkap dengan probabilitas

## Penanganan Data Penting

- **5 kolom leakage dibuang** (TWF, HDF, PWF, OSF, RNF) — kolom ini adalah sub-jenis
  kegagalan yang baru diketahui *setelah* mesin gagal, sehingga tidak sah dipakai sebagai
  fitur prediksi (data leakage)
- **Kolom ID dibuang** (UDI, Product ID) — bukan karakteristik mesin
- **Kolom Type** (L/M/H) di-encode menjadi angka

## Hasil Model (dataset contoh, max_depth=10)

| Metrik | Nilai |
|---|---|
| Akurasi Train | 99,45% |
| Akurasi Test | 98,60% |
| Baseline (tebak "Normal" terus) | 96,60% |
| Recall kelas "Gagal" | 78% |
| Precision kelas "Gagal" | 80% |

Memenuhi kriteria "Sangat Baik" (train ≥80%, test ≥85%). **Catatan analisis:** dataset
tidak seimbang (hanya 3,4% kasus gagal), sehingga precision/recall kelas "Gagal" adalah
metrik yang lebih bermakna dibanding akurasi semata untuk menilai kualitas model.

## Sumber Dataset

Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset*. UCI Machine Learning
Repository. https://doi.org/10.24432/C5HS5C

Dataset sintetis yang merepresentasikan data pemeliharaan mesin industri nyata, dipublikasikan
dalam: S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance
Applications," 2020 Third International Conference on Artificial Intelligence for
Industries (AI4I), 2020, pp. 69-74.
