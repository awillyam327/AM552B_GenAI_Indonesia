# ANALISIS BUKTI EMPIRIS: MENGAPA MICROSOFT COPILOT PALING UNGGUL SECARA MENYELURUH

> **Sumber Data**: `dataset/RAW_DATASET_GENAI_INDONESIA_AM522B.csv` (`71,221` ulasan terverifikasi, `100%` region `id_ID`)  
> **Sampel Microsoft Copilot**: `5,459` ulasan unik (`2023-12-27` s/d `2026-11-04`)  
> **Penanggung Jawab**: Tiara Agustien Kusumawati (682024059)

---

## 1. Ringkasan Eksekutif (4 Bukti Statistik Utama)

1. **Peringkat #1 Indeks Komposit (`85.53/100`) & Rating Aktual Terkoreksi Tertinggi (`4.62/5,00`)**: Setelah mengoreksi bias *Rating-Text Discrepancy*, Microsoft Copilot menempati posisi **#1 tertinggi**, mengungguli Poe (`4.58`), Perplexity (`4.52`), dan ChatGPT (`4.43`).
2. **Satu-satunya Aplikasi tanpa Titik Lemah Aspek (`4/4` Aspek ABSA Bernilai Positif)**: Dari 8 aplikasi yang diuji, **hanya Microsoft Copilot** yang mencetak *Net Sentiment Score* (NSS) positif di keempat dimensi sekaligus (`A1: +84.1%`, `A2: +42.4%`, `A3: +14.8%`, `A4: +45.0%`) dengan rata-rata NSS **`+46.58%`** (Peringkat #1).
3. **Juara Mutlak pada Aksesibilitas Harga/Kuota (`A2: +42,4%`) & Bahasa Indonesia (`A4: +45,0%`)**: Di saat kompetitor utama dikeluhkan karena *paywall* dan limit kuota harian (ChatGPT `-30,4%`, Gemini `-30,7%`, Claude `-25,8%`, Poe `-21,1%`), Copilot memberikan akses gratis ke model penalaran cerdas dan generator gambar AI tanpa pembatasan agresif.
4. **Tingkat Keluhan & Discrepancy Sangat Rendah pada Sampel Besar (`N = 5,459`)**: Proporsi ulasan positif mencapai **`91.88%`** (*Wilson 95% Lower Bound* = `91.13%`), rasio Bintang 1 hanya **`4.51%`**, simpangan baku rendah (**`0.94`**), dan *discrepancy* hanya **`2.95%`**.

---

## 2. Tabel Peringkat Indeks Keunggulan Multidimensi (8 Aplikasi)

| Peringkat | Aplikasi | N Ulasan | Rating Mentah | Rating Terkoreksi | Positif (%) | Bintang 1 (%) | Rata-rata NSS ABSA | Aspek Positif | Skor Komposit |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Microsoft Copilot** | 5,459 | 4.65 | **4.62** | 91.9% | 4.5% | +46.58% | 4/4 | **85.53** |
| **2** | **Poe - Fast AI Chat** | 1,736 | 4.66 | **4.58** | 92.1% | 3.8% | +26.68% | 3/4 | **81.57** |
| **3** | **Perplexity - Ask Anything** | 8,043 | 4.54 | **4.52** | 89.0% | 6.7% | +31.55% | 3/4 | **81.50** |
| **4** | **ChatGPT (OpenAI)** | 22,000 | 4.46 | **4.43** | 86.5% | 8.5% | +12.73% | 2/4 | **76.91** |
| **5** | **Google Gemini** | 12,600 | 4.29 | **4.27** | 81.7% | 12.0% | -0.55% | 1/4 | **72.10** |
| **6** | **DeepSeek - AI Assistant** | 5,374 | 3.77 | **3.65** | 66.6% | 20.8% | +5.35% | 2/4 | **64.63** |
| **7** | **Claude by Anthropic** | 4,009 | 3.85 | **3.74** | 69.3% | 21.4% | -6.30% | 1/4 | **63.78** |
| **8** | **Character.AI** | 12,000 | 1.58 | **1.50** | 11.4% | 77.5% | -83.90% | 0/4 | **18.09** |

---

## 3. Analisis Head-to-Head terhadap Kompetitor Terdekat

- **Microsoft Copilot vs Poe - Fast AI Chat**: Meski rating mentah Poe 4.66 vs Copilot 4.65, sampel Copilot 3,1x lebih besar (5,459 vs 1,736 ulasan). Setelah koreksi discrepancy, Rating Aktual Copilot unggul #1 (4.62 vs 4.58), dan Poe bernilai negatif pada A2 Harga & Kuota (-21.1% vs Copilot +42.4%).
- **Microsoft Copilot vs ChatGPT (OpenAI)**: ChatGPT unggul volume unduhan (22,000 ulasan), namun menderita keluhan berat pada A2 Harga & Kuota (-30.4%) serta A3 Performa/Server (-18.1%). Copilot unggul di seluruh 4 aspek dengan rata-rata NSS +46.58% vs ChatGPT +12.73%.
- **Microsoft Copilot vs Perplexity - Ask Anything**: Perplexity kuat pada pencarian & kecepatan (+40.0%), tetapi masih negatif pada A2 Harga & Kuota (-5.5%). Copilot unggul pada Rating Terkoreksi (4.62 vs 4.52) dan A4 Bahasa Indonesia (+45.0% vs +8.8%).

---

## 4. Faktor Pendorong Kepuasan pada Ulasan Positif Microsoft Copilot

| Faktor Keunggulan Utama | Dimensi ABSA | Jumlah Ulasan Eksplisit | % Ulasan Substantif (>=4 Kata) |
| :--- | :--- | :---: | :---: |
| Kecepatan Respons, UI Ringan & Stabilitas Server | A3: Performa, UI/UX & Server | 252 | 11.9% |
| Akurasi Jawaban, Penalaran GPT-4 & Referensi Web | A1: Akurasi & Kecerdasan AI | 258 | 12.2% |
| Dukungan Tugas Kuliah, Skripsi, Koding & Produktivitas Kerja | A1: Akurasi & Kecerdasan AI | 172 | 8.1% |
| Pembuatan Gambar / Desain AI & Multimodalitas | A1 & A2 (Fitur Kreatif Gratis) | 142 | 6.7% |
| Pemahaman Bahasa Indonesia Luwes & Natural | A4: Bahasa Indonesia & Konteks Lokal | 67 | 3.2% |
| Akses Fitur Premium Gratis Tanpa Paywall Agresif | A2: Harga & Limitasi Kuota | 59 | 2.8% |

---

## 5. Konsistensi Performa Lintas Tahun (2023–2026)

| Tahun | Jumlah Ulasan | Rata-rata Bintang | Proporsi Bintang 4–5 (%) | Proporsi Bintang 1–2 (%) |
| :---: | :---: | :---: | :---: | :---: |
| 2023 | 16 | 4.75 | 93.8% | 6.2% |
| 2024 | 2,554 | 4.65 | 91.7% | 5.9% |
| 2025 | 2,259 | 4.70 | 93.2% | 4.7% |
| 2026 | 630 | 4.52 | 88.1% | 9.4% |

---

## 6. Sampel Bukti Ulasan Nyata Pengguna Indonesia

- **[2024-10-07 | Bintang 5 | 82 Like]**: *"Sangat Terbantu Untuk Masalah Tertentu, Termasuk Masalah Pencairan Dengan Ai Yang Sebelumnya menggunakan Microsoft Edge dan SwiftKey, sekarang sudah di proses dengan baik dan cepat. Terimakasih banyak Microsoft©"*
- **[2024-11-26 | Bintang 5 | 59 Like]**: *"Sangat membantu untuk mencari informasi, bukan memecahkan masalah. Anggaplah ini seperti bicara dengan mesin pencari, dia akan menjawab dengan hasil pencarian. Jangan sampai salah guna hingga menganggapnya bisa menggantikan kita atau orang lain mengerjakan tugas."*
- **[2024-05-30 | Bintang 5 | 59 Like]**: *"Kita jangan pernah dukung perkembangan AI milik google, karena google bakal menggunakan buat mesin perang. Microsoft Copilot ini menjaga kuat etika penggunaan AI untuk hal positif. Microsoft Copilot inilah yang harus kita dukung supaya bisa merajai dunia AI"*
- **[2025-03-10 | Bintang 5 | 57 Like]**: *"Aplikasi ini keren banget woiiii sumpah😱😱😱😱...bikin cinta...pokonya kalian yg mau berbicara atau sekedar chat doang itu bagus banget krna bisa berbagai bahasa,, terutama untuk kalian yg pengen latihan bahasa apa pun bisa pakai aplikasi ini...."*
- **[2024-05-13 | Bintang 5 | 51 Like]**: *"sangat kreatif dan pintar dalam menjawab persoalan gambar dan soal lainnya tapi mohon jadikan Microsoft copilot sebagai asisten AI di semua ponsel Android karena ketika lagi sibuk tidak sempat untuk membuka APPS nya makanya tolong sediakan asisten AI NYA dong"*
- **[2025-04-14 | Bintang 5 | 43 Like]**: *"Penalaran Ai Ini bagus, Layout Ui nya bagus, cuma font dari aplikasi ini tidak nyaman dibaca, dan pewarnaan Ui tidak enak dilihat lama, kalau bisa seperti grox dari X, latar belakang dark mode, warna textnya putih"*
