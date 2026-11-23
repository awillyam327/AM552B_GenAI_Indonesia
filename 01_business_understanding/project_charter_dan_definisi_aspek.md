# Definisi Masalah & Taksonomi Aspek Layanan (Bab 1: Business Understanding)

- **Judul Proyek**: Analisis Komparatif Sentimen Berbasis Aspek (ABSA) dan Pemodelan Topik Pengalaman Pengguna terhadap Ekosistem Aplikasi Mobile Generative AI di Indonesia
- **Mata Kuliah**: AM552 - Data Science of IS (S1 Sistem Informasi, FTI UKSW)
- **Dataset**: 71.221 ulasan pengguna Google Play Store Indonesia pada 8 aplikasi Generative AI (Mei 2023 – November 2026)

---

## 1. Rumusan Masalah

1. Bagaimana distribusi polaritas sentimen dan tingkat kepuasan pengguna pada 4 aspek layanan utama (**A1 Akurasi**, **A2 Harga/Kuota**, **A3 Performa/UI/Server**, dan **A4 Bahasa Indonesia**) di 8 aplikasi Generative AI?
2. Seberapa besar proporsi ulasan yang mengalami **Rating-Text Discrepancy** (ketidaksesuaian antara rating bintang dengan isi teks ulasan) dan faktor apa yang memicunya?
3. Bagaimana perbandingan performa model baseline (**Logistic Regression**) dengan model alternatif (**Complement Naive Bayes** dan **Calibrated LinearSVC**) pada data teks ulasan yang memiliki ketimpangan kelas?
4. Bagaimana mengimplementasikan model terbaik ke dalam aplikasi web berbasis **Streamlit** untuk menganalisis ulasan tunggal maupun file CSV secara massal?

---

## 2. Definisi 4 Dimensi Aspek Layanan (ABSA)

Setiap ulasan pengguna dipetakan ke dalam satu atau lebih kategori aspek berikut berdasarkan kamus kata kunci setelah normalisasi slang:

| Kode | Kategori Aspek | Deskripsi | Contoh Kata Kunci Pemicu |
| :---: | :--- | :--- | :--- |
| **A1** | **Akurasi & Kecerdasan AI** | Kemampuan menjawab pertanyaan, membantu tugas/koding, atau masalah halusinasi/jawaban tidak nyambung. | `pintar`, `cerdas`, `akurat`, `tepat`, `paham`, `jawaban`, `solusi`, `membantu`, `tugas`, `skripsi`, `coding`, `halusinasi`, `ngawur`, `salah`, `lupa`, `sensor`, `filter` |
| **A2** | **Harga & Limitasi Kuota** | Batasan kuota chat/gambar harian pada versi gratis serta biaya berlangganan paket Plus/Pro. | `gratis`, `free`, `bayar`, `berbayar`, `mahal`, `murah`, `harga`, `langganan`, `plus`, `pro`, `kuota`, `limit`, `batas`, `habis`, `token`, `iklan`, `premium` |
| **A3** | **Performa, UI/UX & Server** | Kecepatan respons aplikasi, kendala server sibuk/down, gagal login, fitur suara live, dan bug antarmuka. | `cepat`, `lambat`, `loading`, `server`, `sibuk`, `down`, `error`, `bug`, `macet`, `crash`, `login`, `masuk`, `akun`, `update`, `tampilan`, `ui`, `suara`, `live`, `koneksi` |
| **A4** | **Bahasa Indonesia** | Keluwesan penggunaan bahasa Indonesia, pemahaman konteks lokal, dan kualitas terjemahan. | `bahasa`, `indonesia`, `terjemahan`, `translate`, `logat`, `aksen`, `baku`, `gaul`, `lokal`, `kaku`, `inggris`, `pelafalan`, `bicara` |

Ulasan pendek yang hanya berisi ekspresi umum tanpa menyebut aspek spesifik (seperti *"mantap"*, *"bagus"*, *"jelek"*) dikelompokkan sebagai kesan umum (**A0: General Impression**).

---

## 3. Aturan Deteksi Rating-Text Discrepancy

Ulasan ditandai mengalami *discrepancy* apabila:
1. **Tipe I (Rating Tinggi, Teks Mengeluh)**: Pengguna memberikan Bintang 4 atau 5, tetapi model mendeteksi sentimen negatif atau ditemukan kata keluhan teknis (`error`, `macet`, `lambat`, `limit`, `mahal`, `gagal`, `kecewa`, `rusak`, `ngawur`, `bug`, `down`).
2. **Tipe II (Rating Rendah, Teks Memuji)**: Pengguna memberikan Bintang 1 atau 2, tetapi isi teks murni pujian positif tanpa kata negasi atau keluhan.
