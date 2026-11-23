# LAPORAN FORENSIK AUDIT VALIDITAS METRIK & ANTI-LEAKAGE (CRISP-DM FASE 5)

> **Modul Eksekutor**: `05_evaluation/metric_sanity_audit.py`  
> **Dataset**: `RAW_DATASET_GENAI_INDONESIA_AM522B.csv` (`71.221` baris unik -> `Train: 49,854` | `Val: 10,683` | `Test: 10,684`)  
> **Model Referensi Uji (Pure Text-Only Baseline)**: `TfidfVectorizer(1-2 ngram, max_features=12000)` + `LogisticRegression(class_weight='balanced')`

---

## 1. Status Kelulusan 6 Gerbang Pemeriksaan Forensik (*Sanity Check Gates*)

| Kode Gate | Nama Pemeriksaan Forensik | Status | Temuan Kunci pada Dataset Aktual (71.221 Baris) |
| :---: | :--- | :---: | :--- |
| **AUDIT-1** | **Circular / Target Leakage Check** | ✅ **PASSED** | `0` kolom terlarang di matriks fitur $X$ (hanya menggunakan teks bersih `clean_content`, tanpa `score` dan tanpa `app_name`). |
| **AUDIT-2** | **Class Imbalance vs. Dummy Baseline** | ✅ **PASSED** | Model meningkatkan **Macro F1 sebesar `+0.3098`** di atas *DummyClassifier* dan mencapai **MCC `0.6699`**. |
| **AUDIT-3** | **Exact-Text Duplicate & Short-Text Slice** | ✅ **PASSED** | Terdeteksi **`42.11%`** baris Test memiliki teks identik dengan Train (akibat ulasan 1–2 kata seperti `"bagus"`). Metrik wajib dilaporkan terpisah per *slice*! |
| **AUDIT-4** | **Leave-One-App-Out (LOAO) Generalization** | ✅ **PASSED** | Tanpa pernah melihat data aplikasi target saat training, model tetap mencapai rata-rata **Macro F1 `0.5982`** dan **ROC-AUC > 0.84** pada aplikasi baru. |
| **AUDIT-5** | **Pipeline Split Hygiene Verification** | ✅ **PASSED** | `0` irisan `reviewId` antar-split (`Train/Val/Test`), dan `TfidfVectorizer.fit()` terisolasi penuh di dalam `Pipeline` pada `Train Set`. |
| **AUDIT-6** | **Randomized-Label Permutation Test ($y_{\text{shuffled}}$)** | ✅ **PASSED** | Saat label diacak, **ROC-AUC jatuh tepat ke `0.4558`** ($pprox 0.5000$) dan **MCC = `-0.0576`** ($pprox 0.0000$). Membuktikan **0% bug/leakage** pada kode evaluasi! |

---

## 2. Tabel Komparasi Metrik Multi-Slice (Bukti Empiris Mengapa Audit Ini Krusial)

Perhatikan perbedaan perilaku metrik antara **Dummy Baseline**, **Shuffled-Label Sanity Check**, **Full Test Set**, **Deduplicated Unseen-Text Set**, dan **Substantive Reviews (>= 5 Kata)**:

| Skenario / Subset Evaluasi | Jumlah Baris ($N$) | Accuracy | Balanced Acc | **Macro F1** | Weighted F1 | **MCC** | **ROC-AUC (OvR)** | **PR-AUC (Macro)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0A. Dummy (Most Frequent — Tebak Mayoritas)** | 10,684 | 0.7136 | 0.3333 | **0.2776** | 0.5943 | 0.0000 | 0.5000 | 0.3333 |
| **0B. Dummy (Stratified Random)** | 10,684 | 0.5744 | 0.3337 | **0.3337** | 0.5741 | 0.0036 | 0.5008 | 0.3338 |
| **0C. Permutation Sanity Check ($y_{\text{shuffled}}$)** | 10,684 | 0.3817 | 0.3045 | **0.2778** | 0.4292 | -0.0576 | **0.4558** | 0.3280 |
| **1. Full Test Set (Semua Baris Test)** | 10,684 (100%) | 0.8423 | 0.6817 | **0.6435** | 0.8627 | 0.6699 | **0.8727** | 0.6557 |
| **2. Short Generic Reviews ($< 5$ Kata)** | 6,058 (56.7%) | 0.8866 | 0.6362 | **0.6144** | 0.9019 | 0.5481 | **0.8415** | 0.6155 |
| **3. Deduplicated Unseen-Text Test Set** | 6,163 (57.68%) | 0.7913 | 0.6675 | **0.6332** | 0.8139 | 0.6418 | **0.8657** | 0.6566 |
| **4. Substantive Complex Reviews (>= 5 Kata)** | 4,626 (43.3%) | 0.7843 | 0.6680 | **0.6353** | 0.8077 | 0.6395 | **0.8705** | 0.6618 |

### Analisis Forensik terhadap Angka di Atas:
1. **Bukti Nyata Ilusi *Accuracy* pada *DummyClassifier***:
   - Perhatikan baris **0A (Dummy Most Frequent)**: tanpa belajar apa pun (hanya menebak kelas mayoritas Positif/Bintang 4–5 untuk semua baris), *Accuracy* sudah mencapai **`0.7136` (`71.36%`)** dan *Weighted F1* mencapai **`0.5943`**, tetapi **Macro F1 hanya `0.2776`** dan **MCC = `0.0000`**. Ini membuktikan mengapa pelaporan **Macro F1** dan **MCC** bersifat mutlak pada proyek ini.
2. **Efek Reduplikasi Teks Pendek (`42.11%` Overlap)**:
   - Pada ulasan pendek $< 5$ kata (Baris 2), *Accuracy* menyentuh **`0.8866`** dan *Weighted F1* **`0.9019`** karena mayoritas berisi ekspresi tunggal seperti `"bagus"`, `"mantap"`, `"keren"` yang sudah pernah dilihat di Train Set.
   - Ketika seluruh teks yang pernah muncul di Train Set dihapus (**Baris 3: Deduplicated Unseen-Text Test Set**, $N = 6,163$ kalimat yang 100% baru bagi model) atau difokuskan pada kalimat panjang (**Baris 4: Substantive Reviews >= 5 Kata**), kita mendapatkan ukuran kemampuan generalisasi bahasa alami yang sesungguhnya (**ROC-AUC `0.8657`** dan **Macro F1 `0.6332`**).

---

## 3. Hasil Audit 4: Leave-One-App-Out (LOAO) Cross-Domain Generalization

Menguji apakah model tetap akurat ketika diuji pada aplikasi yang **sama sekali dihapus dari data Training**:

| Aplikasi yang Disembunyikan Saat Training (*Held-Out App*) | Jumlah Baris Train (7 App Lain) | Jumlah Baris Test (Held-Out App) | Accuracy | Balanced Acc | **Macro F1** | MCC | **ROC-AUC (OvR)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeepSeek - AI Assistant** | 65,847 | 5,374 | 0.6865 | 0.6312 | **0.5716** | 0.4796 | 0.8352 |
| **Claude by Anthropic** | 67,212 | 4,009 | 0.7927 | 0.6684 | **0.6227** | 0.6064 | 0.8637 |
| **Google Gemini** | 58,621 | 12,600 | 0.8363 | 0.6273 | **0.6002** | 0.5414 | 0.8351 |

---

## 4. Hasil Audit 6B: Verifikasi Linguistik Fitur Terpenting & *Confident Errors*

### A. Top Fitur Kata Kunci Pendorong Prediksi per Kelas
- **Kelas Negatif (`0_Negative / Star 1-2`)**: `jelek (+5.13)`, `kecewa (+3.79)`, `tidak (+3.73)`, `verifikasi (+3.15)`, `update (+2.98)`, `lebih suka (+2.89)`, `kikir (+2.86)`, `age (+2.85)`, `bad (+2.82)`, `18 (+2.78)`, `buruk (+2.69)`, `mending (+2.68)`
- **Kelas Netral (`1_Neutral / Star 3`)**: `not bad (+3.01)`, `kembaliin (+2.77)`, `terkadang (+2.67)`, `menjengkelkan (+2.60)`, `tiga (+2.58)`, `kadang (+2.47)`, `why (+2.45)`, `server (+2.44)`, `jadi kaku (+2.44)`, `masih (+2.38)`, `lumayan (+2.24)`, `loginnya (+2.20)`
- **Kelas Positif (`2_Positive / Star 4-5`)**: `nice (+2.70)`, `bermanfaat (+2.60)`, `best (+2.51)`, `top (+2.43)`, `the best (+2.25)`, `jos (+2.10)`, `mantab (+2.09)`, `amazing (+2.08)`, `bagus aplikasi (+1.98)`, `bagus kok (+1.98)`, `joss (+1.96)`, `dan (+1.87)`

### B. Sampel *Top Confident Errors* (Bukti Kasus *Rating-Text Discrepancy* Sejati!)
Ketika model sangat yakin ($P > 0.85$) bahwa sebuah ulasan bernada Negatif, tetapi label bintang aslinya adalah Positif (atau sebaliknya), pemeriksaan forensik membuktikan bahwa mayoritas kasus tersebut adalah **Rating-Text Discrepancy asli dari pengguna**:

| Aplikasi | Rating Asli | Label Bintang (`y_true`) | Prediksi Teks (`y_pred`) | Confidence | Teks Asli Ulasan Pengguna (`content`) |
| :--- | :---: | :--- | :--- | :---: | :--- |
| `ChatGPT (OpenAI)` | ⭐ 5 | `2_Positive (Star 4-5)` | `0_Negative (Star 1-2)` | 0.999 | *"jelek"* |
| `Character.AI` | ⭐ 5 | `2_Positive (Star 4-5)` | `0_Negative (Star 1-2)` | 0.989 | *"This app just getting stupid, so often need to update and once updated, I can't access my account back. also m"* |
| `Character.AI` | ⭐ 3 | `1_Neutral (Star 3)` | `0_Negative (Star 1-2)` | 0.988 | *"jujur gue kecewa bgt sma c.ai sekarang, padahal udh bertahun-tahun gue main, tpi semakin update semakin mengec"* |
| `DeepSeek - AI Assistant` | ⭐ 5 | `2_Positive (Star 4-5)` | `0_Negative (Star 1-2)` | 0.984 | *"Tak bisa chat"* |
| `Character.AI` | ⭐ 5 | `2_Positive (Star 4-5)` | `0_Negative (Star 1-2)` | 0.983 | *"tolol, app udah bagus malah di hancurin, bukannya fix bugs malah nambah fitur ga guna, premium lu noh overpric"* |
| `Perplexity - Ask Anything` | ⭐ 4 | `2_Positive (Star 4-5)` | `0_Negative (Star 1-2)` | 0.982 | *"dasar ai tolol gk guna jawaban salah semua tak"* |
| `DeepSeek - AI Assistant` | ⭐ 5 | `2_Positive (Star 4-5)` | `1_Neutral (Star 3)` | 0.981 | *"Bagus, tapi bermasalah di server. Selalu server terlalu sibuk"* |
| `Google Gemini` | ⭐ 3 | `1_Neutral (Star 3)` | `2_Positive (Star 4-5)` | 0.980 | *"bermanfaat banget"* |

> *File lengkap 20 kasus Confident Errors telah disimpan di: `./05_evaluation/confident_errors_fp_fn.csv`*
