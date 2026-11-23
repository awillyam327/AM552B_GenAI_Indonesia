# LAPORAN TOPIC MODELING (LDA) & TOPIC DISCOVERY PENGALAMAN PENGGUNA
> **Penanggung Jawab**: Tiara Agustien Kusumawati (NIM: 682024059) — Member 3 (EDA & Topic Discovery)  
> **Mata Kuliah**: AM552 - Data Science of IS (FTI UKSW)  
> **Metode**: Unsupervised Latent Dirichlet Allocation (LDA) ($k=5$ topik global, $k=3$ topik spesifik aplikasi)  
> **Korpus Input**: 71.221 Ulasan Google Play Store Indonesia (Negative: 16,648, Positive: 34,123)

---

## 1. Top 5 Klaster Topik Keluhan Utama Pengguna AI di Indonesia (Bintang 1–2)

| ID Topik | Label Bisnis Keluhan | Top Kata Kunci (Keywords) Dominan | Interpretasi Masalah Pengguna |
| :---: | :--- | :--- | :--- |
| **Topik 1** | **Masalah Server, Koneksi Jaringan & Gagal Login** | `makin, banyak, iklan, update, bug, lama, jelek, terus` | Pengguna mengalami kegagalan akses server, pemutusan koneksi di tengah generasi jawaban, dan kegagalan autentikasi akun Google. |
| **Topik 2** | **Sensor/Filter Ketat, Perubahan Memori Bot & Protes Update (Character.AI)** | `tidak, verifikasi, update, harus, tapi, mengapa, sekarang, umur` | Sentimen negatif masif dari pengguna Character.AI akibat pembatasan sensor kata (*NSFW filter*) dan amnesia karakter setelah pembaruan aplikasi. |
| **Topik 3** | **Batasan Kuota / Limit Pesan Cepat Habis & Langganan Berbayar Mahal** | `sekarang, tidak, dulu, tapi, suka, padahal, bagus, dikit` | Friksi pada limitasi kuota harian/pesan gratis, harga paket langganan bulanan (*Plus/Pro*) yang dinilai terlalu mahal untuk pelajar/mahasiswa. |
| **Topik 4** | **Bug Aplikasi, Layar Hitam, Crash & Keluar Sendiri** | `tidak, bagus, jelas, aneh, malah, sering, karakter, tidak jelas` | Masalah stabilitas teknis mobile app: aplikasi tiba-tiba keluar (*force close*), layar hitam (*blank screen*), dan bug antarmuka pasca-update. |
| **Topik 5** | **Penurunan Kualitas Jawaban, Halusinasi & Terjemahan Kaku** | `tiba, you, please, update, old, age, but, back` | Kekecewaan terhadap akurasi jawaban yang melenceng (*halusinasi*), jawaban berputar-putar tanpa solusi konkret, dan terjemahan Bahasa Indonesia yang kaku. |

---

## 2. Top 5 Klaster Topik Pujian Utama Pengguna AI di Indonesia (Bintang 4–5)

| ID Topik | Label Bisnis Pujian | Top Kata Kunci (Keywords) Dominan |
| :---: | :--- | :--- |
| **Topik 1** | **Bantuan Tugas Akademik, Skripsi & Pembelajaran** | `membantu, mantap, cepat, gambar, dulu, bahasa, respon, coba` |
| **Topik 2** | **Kecepatan & Kemudahan Penggunaan Sehari-hari** | `bagus, tidak, tapi, sekali, terbaik, bagus sekali, berguna, belajar` |
| **Topik 3** | **Kemampuan Penalaran, Koding & Pemrograman** | `membantu, baik, sekali, good, tugas, mudah, membantu sekali, hal` |
| **Topik 4** | **Kecerdasan Jawaban & Informasi Akurat** | `bagus, chatgpt, bagus membantu, gpt, best, akurat, you, curhat` |
| **Topik 5** | **Rekomendasi Layanan AI Terbaik & Fitur Gratis Bermanfaat** | `kasih, terima, terima kasih, suka, keren, bagus, membuat, bermanfaat` |

---

## 3. Bedah Kasus Anomali Aplikasi (App-Specific Topic Discovery)

### A. Mengapa Rata-rata Rating Character.AI Anjlok ke 1.58?
1. **Sensor & Filter Dialog**: Kata kunci `makin, tidak, sekarang, dulu, update` membuktikan bahwa filter keamanan dinilai membunuh kreativitas *roleplay*.
2. **Kehilangan Memori Bot & UI Baru**: Kata kunci `you, old, update, please, age` mengonfirmasi bahwa komunitas merasa bot menjadi 'bodoh' dan melupakan riwayat percakapan sebelumnya setelah pembaruan UI.

### B. Mengapa DeepSeek Ratingnya 3.77 Padahal Dianggap Sangat Pintar?
1. **Server Sibuk / Kelebihan Beban**: Kata kunci `tidak, server, sering, sibuk, error` menunjukkan keluhan utama adalah *server capacity bottleneck*, bukan kecerdasan AI.

---
*Laporan ini dihasilkan secara otomatis oleh script `04_modeling/run_topic_modeling_experiment.py` untuk mendukung Bab 4 & Bab 7 Laporan Akhir AM552.*
