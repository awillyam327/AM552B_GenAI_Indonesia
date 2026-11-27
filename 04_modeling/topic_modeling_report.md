# LAPORAN PEMODELAN TOPIK (LDA) & EVALUASI KOHERENSI PENGALAMAN PENGGUNA
> **Metode**: Unsupervised Latent Dirichlet Allocation (LDA) ($K=5$ topik optimal)  
> **Korpus Substantif**: Keluhan Bintang 1–2 (`14,317` ulasan) · Pujian Bintang 4–5 (`15,703` ulasan)  
> **Kamus Normalisasi Slang**: `8,618` entri bahasa Indonesia

---

## 1. Evaluasi Pemilihan Jumlah Topik Optimal ($K \in \{3, 4, 5, 6, 8\}$)

| Jumlah Topik ($K$) | Perplexity (Keluhan) | $C_{\text{UMass}}$ Coherence (Keluhan) | Perplexity (Pujian) | $C_{\text{UMass}}$ Coherence (Pujian) |
| :---: | :---: | :---: | :---: | :---: |
| **K = 3** | 789.4 | -2.5980 | 967.5 | -3.0434 |
| **K = 4** | 838.0 | -2.8873 | 934.8 | -2.9174 |
| **K = 5** | 807.1 | -2.7101 | 932.0 | -3.0445 |
| **K = 6** | 808.3 | -2.7806 | 939.8 | -3.0862 |
| **K = 8** | 795.7 | -2.9805 | 913.5 | -3.3015 |

---

## 2. Lima Klaster Topik Keluhan Utama Pengguna AI di Indonesia (Bintang 1–2)

| ID | Label Bisnis Topik | Aspek ABSA | Proporsi | Kata Kunci Dominan (LDA) |
| :---: | :--- | :--- | :---: | :--- |
| **T1** | **Masalah Server Sibuk, Jaringan & Kegagalan Login** | A3: Performa, UI/UX & Server | 11.9% (1,698) | `lama, foto, edit, akun, lambat, loading, jaringan, update` |
| **T2** | **Ketidakakuratan Jawaban, Halusinasi AI & Kendala Bahasa** | A1 & A4 (Akurasi Jawaban & Bahasa Lokal) | 21.1% (3,024) | `perbaiki, filter, sibuk, menolong, jawaban, gambar, salah, kata` |
| **T3** | **Pembatasan Kuota Harian, Limit Pesan & Biaya Langganan** | A2: Harga & Limitasi Kuota | 14.6% (2,085) | `limit, chat, kena, membayar, gambar, gratis, tiba, batas` |
| **T4** | **Protes Sensor Filter, Verifikasi Usia & Perubahan Karakter Bot** | A1 & A3 (Kebijakan Konten & Update Fitur) | 18.1% (2,588) | `verifikasi, update, chat, age, umur, even, age verifikasi, iklan` |
| **T5** | **Gangguan Iklan, Bug Pasca-Update & Ketidakstabilan Aplikasi** | A3: Performa, UI/UX & Server | 34.4% (4,922) | `iklan, update, verifikasi, menolong, usia, chat, bug, fitur` |

---

## 3. Lima Klaster Topik Pujian Utama Pengguna AI di Indonesia (Bintang 4–5)

| ID | Label Bisnis Topik | Aspek ABSA | Proporsi | Kata Kunci Dominan (LDA) |
| :---: | :--- | :--- | :---: | :--- |
| **T1** | **Kecepatan Respons, Kemudahan UI & Fitur Multimodal Gratis** | A2 & A3 (Aksesibilitas Gratis & UI/UX) | 21.9% (3,441) | `gambar, fitur, gratis, foto, mudah, chat, edit, limit` |
| **T2** | **Kecerdasan Penalaran, Akurasi Jawaban & Pemahaman Konteks** | A1: Akurasi & Kecerdasan AI | 27.0% (4,246) | `membantu, cepat, mudah, jawaban, lengkap, akurat, informasi, mencari` |
| **T3** | **Kualitas Bahasa Indonesia Luwes & Interaksi Percakapan Alami** | A4: Bahasa Indonesia & Konteks Lokal | 17.3% (2,723) | `teman, membantu, cerita, belajar, chat, pintar, menjadi, gratis` |
| **T4** | **Produktivitas Pekerjaan, Pencarian Informasi & Solusi Cepat** | A1 & A3 (Produktivitas & Kecepatan) | 17.7% (2,779) | `membantu, menolong, perbaiki, lama, pekerjaan, bug, lambat, sibuk` |
| **T5** | **Pendamping Tugas Kuliah, Skripsi, Belajar & Koding** | A1: Akurasi & Kecerdasan AI | 16.0% (2,514) | `membantu, tugas, mengerjakan, bicara, mengerjakan tugas, bahasa, lain, membantu mengerjakan` |
