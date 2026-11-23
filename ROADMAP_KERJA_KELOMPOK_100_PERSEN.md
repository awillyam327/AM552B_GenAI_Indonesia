# ROADMAP & AGENDA KERJA KELOMPOK MENUJU 100% SELESAI (23 NOV – 30 NOV 2026)
## Proyek Akhir AM552 - Data Science of IS (Program Studi S1 Sistem Informasi, FTI UKSW)

> **Judul Proyek**: *Analisis Komparatif Sentimen Berbasis Aspek (ABSA) dan Pemodelan Topik Pengalaman Pengguna terhadap Ekosistem Aplikasi Mobile Generative AI di Indonesia*  
> **Deadline Pengumpulan FLearn**: **30 November 2026 Pukul 23.00 WIB**  
> **Format Penamaan File Pengumpulan**: `682024046_[NomorKelompok]_DS`  
> **Repository GitHub**: [https://github.com/awillyam327/AM552B_GenAI_Indonesia](https://github.com/awillyam327/AM552B_GenAI_Indonesia)  
> **Live Prototype Streamlit**: [https://genai-indonesia-am552b.streamlit.app](https://genai-indonesia-am552b.streamlit.app)

---

## 👥 Komposisi Tim & Pembagian Wilayah File (*Zero-Overlap Rule*)

Agar tidak pernah terjadi bentrok kode (*merge conflict*) di GitHub, setiap anggota memiliki **Wilayah File Utama** masing-masing:

| Anggota | NIM & Nama Lengkap | Peran Utama | Wilayah File Utama yang Diedit | Porsi |
| :---: | :--- | :--- | :--- | :---: |
| **Member 1** | **682024046**<br>Arthur Willyam Liang | **Project Lead, ML Modeling, Tuning 5-Fold, Anti-Leakage Audit & Deployment** | - `notebooks/03_Modeling_Tuning_dan_Evaluasi.ipynb`<br>- `05_evaluation/metric_sanity_audit.py`<br>- `models/genai_absa_pipeline.joblib` | **34%** |
| **Member 2** | **682024016**<br>Brendan Wilson Santoso | **Data Engineering, Text Preprocessing, Kamus Slang, ABSA & Testing Prototype** | - `notebooks/02_Data_Preparation_dan_ABSA.ipynb`<br>- `03_data_preparation/slang_dictionary_id.json`<br>- `prototype/test_prototype_scenarios.py` | **33%** |
| **Member 3** | **682024059**<br>Tiara Agustien Kusumawati | **Exploratory Data Analysis (EDA), Topic Discovery, Laporan 7 Bab & Slide PDF** | - `notebooks/01_Data_Understanding_dan_EDA.ipynb`<br>- `06_deployment_reporting/682024046_KEL_DS_Laporan.docx`<br>- `06_deployment_reporting/682024046_KEL_DS_Presentasi.pdf` | **33%** |

---

## 📊 Posisi Progress Proyek Saat Ini vs Target 100%

Saat ini pondasi teknis (*Starter Kit & Baseline Architecture*) sudah berada di posisi **~65%**. Sisa **35%** adalah **eksekusi kerja kelompok, pengayaan analisis, validasi eksperimen, pemolesan laporan, dan pembuatan slide presentasi** yang dibagi ke dalam **6 Tahap Terstruktur** di bawah ini:

```text
[STATUS SAAT INI: 65%] ──► TAHAP 1 (70%) ──► TAHAP 2 (78%) ──► TAHAP 3 (86%) ──► TAHAP 4 (92%) ──► TAHAP 5 (98%) ──► TAHAP 6 (100% SUBMIT FLEARN)
  Pondasi, Dataset 71k,     Onboarding &      Kamus Slang &     Tuning 3 Model    Uji Skenario &    Finalisasi Word   QA Akhir, Latihan Sidang,
  Baseline & Web Live       Setup 3 Laptop    EDA Notebook      & Audit Metrik    Screenshot Baru   & Slide PPT PDF   & Upload 4 Deliverables
```

---

## 🚀 RINCIAN 6 TAHAP PENGERJAAN MENUJU 100% SELESAI

---

### 📍 TAHAP 1: Onboarding Tim, Sinkronisasi GitHub & Setup Antigravity
* **Waktu Pelaksanaan**: **Hari Ini (Senin, 23 November 2026 — Sesi Kerja Kelompok Perdana, 45 Menit Pertama)**
* **Target Capaian**: Progress **65% $\rightarrow$ 70%**
* **Tujuan**: Memastikan ketiga anggota tim memiliki seluruh file proyek di laptop masing-masing, memahami gambaran besar topik proyek, dan mengaktifkan asisten AI Antigravity sesuai peran.

#### Checklist Langkah Kerja Tahap 1:
- [ ] **1.1 (Arthur)**: Mendemokan aplikasi web live [https://genai-indonesia-am552b.streamlit.app](https://genai-indonesia-am552b.streamlit.app) kepada Brendan dan Tiara (tunjukkan Tab 1, Tab 2 Batch CSV, dan Tab 3 Dashboard 8 Aplikasi).
- [ ] **1.2 (Arthur)**: Menjelaskan 2 konsep inti selama 5 menit menggunakan `TEAM_ONBOARDING_GUIDE.docx`:
  1. **4 Aspek ABSA**: *A1 Akurasi/Kecerdasan*, *A2 Harga/Kuota*, *A3 Performa/UI/Server*, *A4 Bahasa Indonesia*.
  2. **Rating-Text Discrepancy**: Ulasan Bintang 5 tapi isinya mengeluh/komplain, atau Bintang 1 tapi memuji.
- [ ] **1.3 (Brendan & Tiara)**: Menerima undangan *Collaborator* di GitHub (`https://github.com/awillyam327/AM552B_GenAI_Indonesia`), membuka **GitHub Desktop**, dan melakukan **Clone Repository** ke laptop masing-masing.
- [ ] **1.4 (Brendan & Tiara)**: Membuka folder proyek di aplikasi **Antigravity**, lalu menempelkan (*copy-paste*) **Master Prompt Member 2** (untuk Brendan) dan **Master Prompt Member 3** (untuk Tiara).
- [ ] **Definition of Done (Syarat Lulus Tahap 1)**: Ketiga laptop sudah memiliki folder proyek yang identik dan siap menjalankan kode.

---

### 🛠️ TAHAP 2: Pengayaan Kamus Slang, Validasi ABSA & Eksplorasi Data (EDA)
* **Waktu Pelaksanaan**: **23 – 24 November 2026 (Saat Kerja Kelompok & Lanjutan Mandiri)**
* **Target Capaian**: Progress **70% $\rightarrow$ 78%**
* **Tujuan**: Memperkuat kualitas pembersihan teks Bahasa Indonesia (Bab 3) dan memperdalam temuan eksplorasi data pada 8 aplikasi Generative AI (Bab 2).

#### Checklist Langkah Kerja Tahap 2:
- [ ] **2.1 (Brendan — Data Prep & Slang)**:
  - Buka `03_data_preparation/slang_dictionary_id.json` dan sampel data `03_data_preparation/sample_preprocessed_absa_1000.csv`.
  - Cari minimal **20–30 kata gaul/singkatan/typo baru** dari ulasan nyata pengguna Play Store (contoh: *bapuk $\rightarrow$ buruk*, *ampas $\rightarrow$ buruk*, *ngelag $\rightarrow$ lambat*, *halu $\rightarrow$ halusinasi*, *cengo $\rightarrow$ bingung*, * berbayar*, dll.) dan tambahkan ke `slang_dictionary_id.json`.
  - Periksa daftar kata kunci 4 aspek ABSA di `notebooks/02_Data_Preparation_dan_ABSA.ipynb` dan pastikan kata negasi (`tidak`, `kurang`, `jangan`, `bukan`, `belum`) tidak terhapus oleh stopword removal.
  - Jalankan (*Run All*) `notebooks/02_Data_Preparation_dan_ABSA.ipynb` hingga seluruh output tabel sebelum vs sesudah preprocessing muncul sempurna.
- [ ] **2.2 (Tiara — EDA & Pemahaman Data)**:
  - Buka dan jalankan (*Run All*) `notebooks/01_Data_Understanding_dan_EDA.ipynb`.
  - Verifikasi angka statistik Bab 2: Total `71.221` ulasan unik, `0%` duplikasi `reviewId`, `0` missing value pada kolom teks & rating, serta distribusi kelas target yang timpang (Bintang 5 mendominasi `>58%`).
  - Analisis alasan bisnis di balik grafik Bab 2.4:
    1. Mengapa **ChatGPT** (`22.000` ulasan, rating `4.46`) dan **Copilot** (`4.65`) sangat disukai?
    2. Mengapa **Character.AI** memiliki rating paling anjlok (`1.58`)? *(Karena keluhan filter sensor percakapan dan bug login/antrean server)*.
    3. Mengapa **12,4%** ulasan mengalami *Rating-Text Discrepancy*? *(Karena pengguna menyukai kecerdasan AI-nya sehingga memberi bintang 5, namun mengeluhkan kuota chat gratis yang cepat habis di dalam teks ulasan)*.
- [ ] **2.3 (Arthur — Pendampingan & Verifikasi Split)**:
  - Memastikan pembagian data **80% Train (`56.976` baris) : 20% Test (`14.245` baris)** secara *Stratified* (`random_state=42`) berjalan konsisten tanpa mengubah indeks data uji.
- [ ] **Definition of Done (Syarat Lulus Tahap 2)**: Kamus slang bertambah kaya, Notebook 01 dan Notebook 02 sudah dijalankan penuh (*Run All*) dengan output tersimpan, serta telah di-*Commit & Push* ke GitHub.

---

### 🧠 TAHAP 3: Eksperimen 3 Model, Hyperparameter Tuning 5-Fold & Audit Anti-Leakage
* **Waktu Pelaksanaan**: **25 – 26 November 2026**
* **Target Capaian**: Progress **78% $\rightarrow$ 86%**
* **Tujuan**: Memenuhi syarat mutlak Bab 4 & Bab 5 ketentuan dosen (1 Model Baseline Sederhana + Minimal 2 Model Alternatif + Dokumentasi Tuning 5-Fold CV + Uji Asumsi Model + Evaluasi Bebas Data Leakage).

#### Checklist Langkah Kerja Tahap 3:
- [ ] **3.1 (Arthur — Modeling & Hyperparameter Tuning)**:
  - Lakukan **Pull origin** di GitHub Desktop untuk mengambil kamus slang terbaru dari Brendan.
  - Buka dan jalankan `notebooks/03_Modeling_Tuning_dan_Evaluasi.ipynb`:
    1. **Model 1 (Pembanding Dasar / Baseline)**: `Logistic Regression` (`class_weight='balanced'`, `solver='liblinear'`).
    2. **Model 2 (Alternatif 1 — Probabilistic NLP)**: `Complement Naive Bayes (ComplementNB)` (varian Naive Bayes khusus data teks dengan kelas timpang).
    3. **Model 3 (Alternatif 2 — Margin Classifier)**: `Calibrated Linear Support Vector Machine (LinearSVC)`.
  - Jalankan `GridSearchCV` dengan **5-Fold Stratified Cross-Validation** (variasikan parameter regularisasi $C \in \{0.1, 1.0, 5.0\}$, smoothing $\alpha \in \{0.1, 0.5, 1.0\}$, dan `ngram_range`).
  - Catat perbandingan metrik **Macro F1-Score**, **ROC-AUC (OvR)**, dan **Rata-rata 5-Fold CV** ke dalam tabel evaluasi.
- [ ] **3.2 (Arthur — Forensik Anti-Leakage & Ekspor Model)**:
  - Jalankan script `python 05_evaluation/metric_sanity_audit.py` untuk memastikan ke-6 gerbang audit lulus (Zero Train-Test Overlap, mengalahkan Dummy Baseline secara telak, uji permutasi label acak ROC-AUC $\approx 0.50$).
  - Ekspor pipeline model terbaik terbaru ke `models/genai_absa_pipeline.joblib`.
- [ ] **3.3 (Brendan & Tiara — Analisis Kesalahan & Topik Bab 5)**:
  - Buka `05_evaluation/confident_errors_fp_fn.csv` untuk memilih 3–5 contoh nyata ulasan yang salah diprediksi model (*False Positive / False Negative*) sebagai bukti fenomena *Rating-Text Discrepancy* di Bab 5.3.
  - Periksa grafik *Confusion Matrix* dan grafik *Feature Attribution (SHAP / Top Words)* di `05_evaluation/figures/`.
- [ ] **Definition of Done (Syarat Lulus Tahap 3)**: Notebook 03 tuntas dijalankan (*Run All*), file `.joblib` terbaru tersimpan, dan ke-6 pemeriksaan audit berstatus `PASS`.

---

### 🌐 TAHAP 4: Uji 3 Skenario Prototype Live & Pengambilan Screenshot Bab 6
* **Waktu Pelaksanaan**: **27 November 2026**
* **Target Capaian**: Progress **86% $\rightarrow$ 92%**
* **Tujuan**: Memenuhi syarat mutlak Bab 6 ketentuan dosen (Prototype memiliki 3 fitur utama, menggunakan model `.joblib` yang identik dengan notebook, dan lolos uji 3 skenario wajib Bab 6.5).

#### Checklist Langkah Kerja Tahap 4:
- [ ] **4.1 (Arthur & Brendan — Verifikasi Sinkronisasi Prototype)**:
  - Push `models/genai_absa_pipeline.joblib` dan `slang_dictionary_id.json` terbaru ke GitHub (Streamlit Cloud akan otomatis memperbarui aplikasi live dalam 30 detik).
  - Jalankan script uji otomatis `python prototype/test_prototype_scenarios.py` di terminal dan pastikan seluruh skenario berstatus **100% PASS**.
- [ ] **4.2 (Brendan — Pengujian Manual 3 Skenario Wajib di Web Live)**:
  - Buka [https://genai-indonesia-am552b.streamlit.app](https://genai-indonesia-am552b.streamlit.app) dan uji:
    1. **Skenario 1 (Input Normal)**: Masukkan ulasan wajar (misal: *"Aplikasinya pintar banget bantu ngerjain tugas kuliah dan coding Python"* dengan Bintang 5) $\rightarrow$ Pastikan hasil prediksi sesuai dengan notebook (Sentimen Positif, Aspek A1).
    2. **Skenario 2 (Input Ekstrem / Slang & Discrepancy)**: Masukkan ulasan penuh slang & kontradiksi (misal: *"Jawabannya sih pinter bgt, tapi limit kuota gratisnya pelit bgt dikit2 suruh bayar langganan mahal!!"* dengan Bintang 5) $\rightarrow$ Pastikan aplikasi tidak error dan berhasil mendeteksi **Discrepancy** serta Aspek **A1 & A2**.
    3. **Skenario 3 (Input Kosong / Salah Format)**: Kosongkan kotak teks atau isi hanya spasi/simbol `"@#$%^"`, serta coba unggah file CSV tanpa kolom teks di Tab 2 $\rightarrow$ Pastikan muncul **kotak pesan peringatan kuning/merah yang ramah** dan aplikasi tidak crash.
- [ ] **4.3 (Brendan & Tiara — Dokumentasi Screenshot Bab 6.4)**:
  - Ambil tangkapan layar (*screenshot*) yang bersih dari web live untuk:
    1. Halaman Utama & Hasil Prediksi Ulasan Tunggal (Tab 1).
    2. Fitur Upload CSV & Tabel Hasil Prediksi Batch (Tab 2).
    3. Dashboard Komparasi 8 Aplikasi Generative AI (Tab 3).
- [ ] **Definition of Done (Syarat Lulus Tahap 4)**: Tabel 7 (Hasil Uji Coba Bab 6.5) dan gambar tangkapan layar Bab 6.4 lengkap serta tervalidasi.

---

### 📝 TAHAP 5: Finalisasi Laporan Word 7 Bab (`.docx`) & Slide Presentasi (`.pdf`)
* **Waktu Pelaksanaan**: **28 – 29 November 2026**
* **Target Capaian**: Progress **92% $\rightarrow$ 98%**
* **Tujuan**: Menyelesaikan dokumen laporan akhir sesuai `Template Laporan Project Akhir – AM552 Data Science of IS.docx` dan membuat slide presentasi `.pdf` yang siap dipresentasikan.

#### Checklist Langkah Kerja Tahap 5:
- [ ] **5.1 (Tiara — Pemeriksaan & Pemolesan Laporan Word 7 Bab)**:
  - Buka file laporan resmi `06_deployment_reporting/682024046_KEL_DS_Laporan.docx`.
  - Pastikan **Nomor Kelompok** sudah diisi pada Tabel 0 Halaman Judul.
  - Periksa kelengkapan seluruh **8 Tabel Baku (Tabel 0 s/d Tabel 7)** agar tidak ada teks placeholder `[...]` yang tertinggal.
  - Pastikan di bawah **setiap grafik pada Bab 2.4** terdapat tepat **1–2 kalimat penjelasan** (apa yang terlihat pada grafik dan apa artinya bagi pemodelan).
  - Periksa **Daftar Pustaka** di halaman terakhir (pastikan format sitasi konsisten menggunakan gaya **APA** dan mencantumkan sumber dataset Google Play Store serta minimal 3 jurnal/referensi ilmiah).
  - Rapikan *page break* agar tidak ada judul bab yang menggantung di bagian paling bawah halaman.
  - Simpan file dengan nama resmi pengumpulan: **`682024046_[NomorKelompok]_DS.docx`** (atau sesuai instruksi folder FLearn).
- [ ] **5.2 (Tiara, Brendan, Arthur — Pembuatan Slide Presentasi PowerPoint $\rightarrow$ PDF)**:
  - Buat slide presentasi berjumlah **10–12 Slide** di PowerPoint / Canva dengan struktur padat berikut:
    * **Slide 1 (Cover)**: Judul Proyek, Nomor Kelompok, Nama & NIM 3 Anggota (Arthur 682024046, Brendan 682024016, Tiara 682024059), Link GitHub & Link Live Streamlit.
    * **Slide 2 (Bab 1 - Pemahaman Bisnis)**: Masalah ketidakcukupan *Star Rating* di Play Store, fenomena *Rating-Text Discrepancy*, Tujuan Bisnis vs Tujuan Data Science, dan Target Metrik (Macro F1 $\ge 0.63$, ROC-AUC $\ge 0.85$).
    * **Slide 3 (Bab 2 - Dataset & Kondisi Data)**: Profil 71.221 ulasan unik 8 aplikasi GenAI Indonesia (ChatGPT, Gemini, Character.AI, Perplexity, Copilot, DeepSeek, Claude, Poe) & ketimpangan kelas rating.
    * **Slide 4 (Bab 2 - Temuan Visual EDA)**: Grafik distribusi aplikasi, anomali rating Character.AI (1.58), dan temuan 12,4% *Rating-Text Discrepancy*.
    * **Slide 5 (Bab 3 - Persiapan Data & Anti-Leakage)**: Pipeline normalisasi slang Indonesia, preservasi kata negasi (`tidak`, `kurang`), ekstraksi 4 aspek ABSA (A1–A4), dan pembagian data **80:20 Stratified Split sebelum TF-IDF**.
    * **Slide 6 (Bab 4 - Pemodelan & Tuning 5-Fold)**: Perbandingan 3 Algoritma (Baseline `Logistic Regression`, Alternatif 1 `ComplementNB`, Alternatif 2 `Calibrated LinearSVC`) beserta parameter `GridSearchCV` 5-fold.
    * **Slide 7 (Bab 5 - Evaluasi & Pencapaian Target)**: Tabel perbandingan metrik (Macro F1 `0.6512`, ROC-AUC `0.8751`) vs Target Bab 1.4 (**100% Tercapai**).
    * **Slide 8 (Bab 5 - Analisis Kesalahan & Explainable AI)**: Grafik *Confusion Matrix* & Kata Kunci Pendorong Prediksi (*SHAP / Feature Attribution*) + contoh kasus nyata *Discrepancy*.
    * **Slide 9 (Bab 6 - Arsitektur & Fitur Prototype Streamlit)**: Diagram alur sistem (`Input -> Slang & Validator -> Model .joblib -> Output ABSA`) dan screenshot 3 Tab aplikasi web.
    * **Slide 10 (Bab 6.5 - Hasil Uji 3 Skenario Prototype)**: Bukti kelulusan uji Skenario Normal, Skenario Ekstrem, dan Skenario Input Kosong/Salah Format.
    * **Slide 11 (Bab 7 - Kesimpulan & Saran Penerapan)**: Jawaban ringkas 4 rumusan masalah, rekomendasi manajerial untuk pengembang AI, dan rencana pengembangan selanjutnya (IndoBERT GPU).
  - Ekspor slide presentasi tersebut ke format **`.pdf`** dengan nama **`682024046_[NomorKelompok]_DS.pdf`**.
- [ ] **Definition of Done (Syarat Lulus Tahap 5)**: File laporan `.docx` dan file slide presentasi `.pdf` sudah selesai 100% dan disimpan di folder `06_deployment_reporting/`.

---

### 🏆 TAHAP 6: Final Quality Assurance (QA), Latihan Presentasi & Submit FLearn
* **Waktu Pelaksanaan**: **30 November 2026 (Maksimal Pukul 19.00 WIB — 4 Jam Sebelum Deadline 23.00 WIB)**
* **Target Capaian**: Progress **98% $\rightarrow$ 100% (TUNTAS)**
* **Tujuan**: Memastikan tidak ada kesalahan teknis sekecil apa pun pada ke-4 deliverables dan seluruh anggota siap menjawab pertanyaan dosen saat presentasi.

#### Checklist Langkah Kerja Tahap 6 (Final Gate):
- [ ] **6.1 (Arthur — Audit Final Jupyter Notebooks `.ipynb`)**:
  - Buka ketiga file notebook di `notebooks/`:
    1. `01_Data_Understanding_dan_EDA.ipynb`
    2. `02_Data_Preparation_dan_ABSA.ipynb`
    3. `03_Modeling_Tuning_dan_Evaluasi.ipynb`
  - Klik **Kernel $\rightarrow$ Restart & Run All** pada masing-masing notebook untuk memastikan nomor urutan eksekusi sel berurutan rapi (`[1]`, `[2]`, `[3]`, dst.), semua grafik tampil di dalam notebook, dan tidak ada pesan error merah.
  - Simpan (*Save*) dan push ke GitHub.
- [ ] **6.2 (Brendan — Audit Link Publik Mode Incognito)**:
  - Buka browser dalam mode *Incognito / Private Window* (tanpa login akun).
  - Klik link GitHub `https://github.com/awillyam327/AM552B_GenAI_Indonesia` dan link prototype `https://genai-indonesia-am552b.streamlit.app`.
  - Pastikan kedua link dapat dibuka oleh dosen penguji tanpa meminta password/akses, dan aplikasi Streamlit tidak sedang dalam kondisi *sleep* (klik tombol jalankan analisis 1 kali agar server tetap hangat).
- [ ] **6.3 (Tiara, Brendan, Arthur — Simulasi Presentasi 15 Menit & Persiapan Tanya-Jawab)**:
  - Latihan presentasi bersama selama 10–15 menit sesuai pembagian slide:
    * **Tiara**: Slide 1–4 (Bisnis, Dataset 71k, & Temuan EDA).
    * **Brendan**: Slide 5, 9, 10 (Data Prep, Slang, ABSA, & Demo/Uji Skenario Prototype).
    * **Arthur**: Slide 6, 7, 8, 11 (Modeling, Tuning 5-Fold, Evaluasi Anti-Leakage, SHAP, & Kesimpulan).
- [ ] **6.4 (Arthur — Pengumpulan Resmi ke FLearn UKSW)**:
  - Pastikan ke-4 Deliverables utama telah dikemas sesuai aturan penamaan `NIM_KEL_DS` (`682024046_[NomorKelompok]_DS`):
    1. ✅ **File Kode Program (`.ipynb`)**
    2. ✅ **File Laporan Akhir (`.docx`)**
    3. ✅ **File Slide Presentasi (`.pdf`)**
    4. ✅ **Link Live Prototype Streamlit & GitHub** (tercantum di dalam laporan, slide, dan kolom komentar pengumpulan FLearn)
  - Unggah ke **FLearn** sebelum **30 November 2026 pukul 23.00 WIB** dan simpan bukti tangkapan layar (*screenshot*) status *"Submitted for grading"*.

---

## 💡 Contekan Jawaban Sidang / Tanya-Jawab Dosen (Wajib Dikuasai Tim)

Saat presentasi akhir, dosen biasanya akan menguji pemahaman anggota kelompok dengan 4 pertanyaan kritis ini:

1. **Pertanyaan untuk Tiara (EDA & Bisnis)**:  
   *"Mengapa kalian menggunakan dataset ulasan aplikasi Generative AI, dan mengapa rating bintang saja tidak cukup?"*  
   👉 **Jawaban**: *"Karena rata-rata rating bintang menyembunyikan keluhan spesifik pengguna. Dari 71.221 ulasan yang kami analisis, kami menemukan 12,4% kasus Rating-Text Discrepancy—contohnya pengguna memberi Bintang 5 pada ChatGPT karena pintar membantu tugas (Aspek A1), tetapi di dalam teksnya mengeluhkan kuota chat gratis yang cepat habis dan harga langganan mahal (Aspek A2). Dengan ABSA, pengembang aplikasi bisa tahu persis aspek mana yang harus diperbaiki."*

2. **Pertanyaan untuk Brendan (Data Preparation & NLP)**:  
   *"Bagaimana kalian menangani bahasa gaul/slang Indonesia di Play Store, dan kenapa kata 'tidak' atau 'kurang' tidak boleh dihapus saat stopword removal?"*  
   👉 **Jawaban**: *"Kami membangun kamus normalisasi slang terkurasi (`slang_dictionary_id.json`) untuk mengubah ratusan singkatan tidak baku seperti 'bgt', 'ngelag', 'bapuk' menjadi kata baku sebelum masuk ke TF-IDF. Selain itu, kami menerapkan Negation Preservation—kata negasi seperti 'tidak', 'kurang', 'jangan', dan 'bukan' sengaja dipertahankan karena jika dihapus, frasa 'tidak akurat' akan berubah menjadi 'akurat' sehingga membalikkan sentimen secara fatal."*

3. **Pertanyaan untuk Arthur (Modeling & Anti-Leakage)**:  
   *"Mengapa kalian menggunakan Macro F1-Score sebagai ukuran keberhasilan utama, bukan Accuracy biasa? Dan bagaimana kalian menjamin tidak ada data leakage?"*  
   👉 **Jawaban**: *"Distribusi ulasan Play Store sangat timpang (Bintang 5 mendominasi lebih dari 58% data). Jika menggunakan Accuracy biasa, model yang hanya menebak semua ulasan sebagai Positif akan mendapat akurasi tinggi palsu tetapi gagal mengenali keluhan negatif. Macro F1-Score memberi bobot setara (1/3) untuk setiap kelas. Untuk mencegah data leakage, kami melakukan pembagian data 80:20 Stratified Split terlebih dahulu sebelum melakukan `TfidfVectorizer.fit()`, sehingga kosakata data uji 100% terisolasi."*

4. **Pertanyaan untuk Bersama (Prototype Bab 6)**:  
   *"Bagaimana kalian memastikan model di aplikasi Streamlit sama persis dengan yang ada di Jupyter Notebook?"*  
   👉 **Jawaban**: *"Seluruh pipeline dari vektorisasi TF-IDF hingga classifier terbaik diekspor menjadi satu objek biner utuh (`models/genai_absa_pipeline.joblib`) dan dipanggil oleh modul `core_predictor.py` menggunakan kamus `slang_dictionary_id.json` yang sama. Kami juga telah mengujinya dengan 3 skenario wajib (input normal, input ekstrem/slang, dan input kosong/salah format) dengan hasil 100% lulus tanpa error."*
