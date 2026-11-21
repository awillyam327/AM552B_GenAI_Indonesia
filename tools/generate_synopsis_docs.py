import json
import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SUMMARY_JSON_PATH = os.path.join(ROOT_DIR, "dataset", "audit_summary.json")
MD_OUT_PATH = os.path.join(ROOT_DIR, "Sinopsis_GenerativeAI_Indonesia_682024046.md")
DOCX_OUT_PATH = os.path.join(ROOT_DIR, "Sinopsis_GenerativeAI_Indonesia_682024046.docx")


def set_cell_background(cell, fill_hex):
    """Set warna latar belakang sel tabel Word."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill_hex)
    tcPr.append(shd)


def format_number_id(n):
    """Format ribuan dengan titik (standar Bahasa Indonesia)."""
    return f"{n:,}".replace(",", ".")


def build_markdown(summary):
    total_rows = summary["total_rows"]
    total_rows_str = format_number_id(total_rows)
    total_cols = summary["total_cols"]
    min_date = summary["min_date"][:10]
    max_date = summary["max_date"][:10]
    app_breakdown = summary["app_breakdown"]
    score_dist = summary["score_distribution"]

    app_table_rows = []
    for idx, item in enumerate(app_breakdown, 1):
        cnt_str = format_number_id(item["count"])
        pct_str = f"{item['pct']:.2f}%"
        avg_str = f"{item['avg_score']:.2f} / 5.00"
        period_str = f"{item['min_date']} s/d {item['max_date']}"
        app_table_rows.append(
            f"| {idx} | **{item['app_name']}** | `{item['app_id']}` | {cnt_str} | {pct_str} | {avg_str} | {period_str} |"
        )
    app_table_md = "\n".join(app_table_rows)

    score_rows = []
    for star in ["5", "4", "3", "2", "1"]:
        if star in score_dist:
            c_str = format_number_id(score_dist[star]["count"])
            p_str = f"{score_dist[star]['pct']:.2f}%"
            label = {
                "5": "Sangat Puas (Bintang 5)",
                "4": "Puas (Bintang 4)",
                "3": "Netral / Ragu-ragu (Bintang 3)",
                "2": "Tidak Puas (Bintang 2)",
                "1": "Sangat Kecewa / Komplain Kritis (Bintang 1)",
            }[star]
            score_rows.append(f"| ⭐ {star} | {label} | {c_str} | {p_str} |")
    score_table_md = "\n".join(score_rows)

    md_content = f"""# SINOPSIS PROYEK AKHIR DATA SCIENCE (AM522B)

---

## 1. Judul Usulan Proyek

### Judul Utama (Rekomendasi Utama)
> **"Analisis Komparatif Sentimen Berbasis Aspek (ABSA) dan Pemodelan Topik Pengalaman Pengguna terhadap Ekosistem Aplikasi Mobile Generative AI di Indonesia"**

### Variasi Judul Alternatif
1. **Fokus *Predictive User Satisfaction & Churn Risk***:  
   *"Prediksi Ketidakpuasan Pengguna dan Deteksi Anomali Rating-Ulasan (Rating-Text Discrepancy) pada Aplikasi Mobile Generative AI di Indonesia Menggunakan Ensemble Gradient Boosting dan SHAP Explainability"*
2. **Fokus *NLP Transformer Benchmarking & Topic Discovery***:  
   *"Studi Komparasi Arsitektur Pre-Trained IndoBERT dan Gradient Boosting serta Ekstraksi Topik Berbasis BERTopic pada {total_rows_str} Ulasan Pengguna Aplikasi Generative AI di Google Play Store Indonesia"*

---

## 2. Latar Belakang & Urgensi Penelitian

Ekosistem komputasi personal dan produktivitas digital di Indonesia tengah mengalami transformasi struktural seiring ledakan adopsi aplikasi **Mobile Generative AI** (*Large Language Models* dan asisten percakapan multimodal). Platform global seperti **ChatGPT (OpenAI)**, **Google Gemini**, **DeepSeek**, **Microsoft Copilot**, **Perplexity**, **Claude (Anthropic)**, **Poe**, hingga **Character.AI** dan **Qwen** kini digunakan secara masif oleh pelajar, mahasiswa, praktisi teknologi, hingga pekerja kreatif di Indonesia untuk mendukung penulisan akademik, pemrograman (*coding*), pencarian informasi *real-time*, serta otomatisasi tugas harian.

Meskipun tingkat unduhan aplikasi Generative AI di Indonesia tumbuh secara eksponensial, persaingan ketat antar-platform memunculkan dinamika pengalaman pengguna (*User Experience*) yang sangat kompleks dan terpolarisasi di Google Play Store:
1. **Kualitas Pemahaman Konteks & Bahasa Indonesia**: Pengguna di Indonesia sering mengeluhkan ketimpangan kemampuan model dalam memahami struktur bahasa kasual/slang lokal, konteks budaya Indonesia, hingga fenomena **halusinasi model** (*hallucination*) saat menjawab pertanyaan faktual atau teknis.
2. **Friksi Monetisasi, Batasan Kuota Gratis, dan Harga Langganan**: Kebijakan pembatasan pesan harian (*rate limit* model flagship seperti GPT-4o, Claude 3.5/3.7 Sonnet, atau akses fitur gambar/suara) serta tingginya biaya berlangganan (*Plus/Pro tier*) dibandingkan daya beli pengguna lokal menjadi pemicu utama migrasi pengguna ke alternatif gratis seperti DeepSeek.
3. **Reliabilitas Teknis, Latensi Mobile, dan Fitur Multimodal**: Keluhan terkait kegagalan *login* autentikasi, *server down/busy* (terutama saat lonjakan trafik rilis model baru), konsumsi baterai, respons fitur *voice mode* yang lambat, hingga perubahan antarmuka (*UI/UX*) berdampak langsung terhadap retensi pengguna.
4. **Fenomena *Rating-Text Discrepancy* (paradoks skor bintang vs. teks ulasan)**: Banyak pengguna di Indonesia memberikan rating bintang 5 tetapi menuliskan keluhan teknis di dalam teks ulasan (atau sebaliknya, memberikan bintang 1 hanya karena salah paham cara login). Oleh sebab itu, metrik agregat bintang semata tidak lagi memadai untuk mengevaluasi kualitas layanan secara objektif.

Melalui pendekatan **Aspect-Based Sentiment Analysis (ABSA)** dan **Pemodelan Topik (Topic Modeling)** terhadap **{total_rows_str} data ulasan nyata (*100% real scraped data*)**, penelitian ini memiliki urgensi tinggi untuk membedah peta kekuatan, kelemahan, dan determinan kepuasan pengguna ekosistem Generative AI di Indonesia secara empiris dan berbasis bukti (*data-driven*).

---

## 3. Rumusan Masalah & Tujuan Analisis

### A. Rumusan Masalah
1. Bagaimana perbandingan distribusi polarisasi sentimen (positif, netral, negatif) dan dinamika tingkat kepuasan pengguna antar-aplikasi mobile Generative AI terpopuler di Indonesia dari waktu ke waktu?
2. Aspek spesifik layanan apa saja (**Akurasi & Halusinasi Model**, **Harga & Limitasi Kuota Gratis**, **UI/UX & Kecepatan Server/Autentikasi**, serta **Kemampuan Bahasa Indonesia & Fitur Multimodal**) yang paling dominan memicu sentimen negatif maupun positif pada masing-masing platform?
3. Seberapa besar tingkat ketidaksesuaian (*rating-text discrepancy*) antara skor bintang numerik (1–5) yang diberikan pengguna dengan polaritas sentimen aktual di dalam teks ulasannya?
4. Sejauh mana keunggulan performa model *Pre-trained Language Model* (**IndoBERT**) dibandingkan algoritma *Ensemble Machine Learning* (**CatBoost, XGBoost, LightGBM**) dalam mengklasifikasikan sentimen dan mendeteksi *rating-text discrepancy*, serta fitur kata kunci apa yang paling berpengaruh berdasarkan analisis **SHAP**?

### B. Tujuan Analisis
1. Memetakan lanskap komparatif kepuasan dan polarisasi sentimen pengguna lintas aplikasi Generative AI utama di Indonesia menggunakan **{total_rows_str} ulasan unik** dari Google Play Store.
2. Mengekstraksi struktur topik laten dan memetakan sentimen berbasis aspek (*ABSA*) menggunakan **BERTopic** dan **Latent Dirichlet Allocation (LDA)** untuk mengidentifikasi *pain points* dan keunggulan kompetitif tiap platform.
3. Mengukur dan mengklasifikasikan fenomena *rating-text discrepancy* guna menyaring bias penilaian bintang pada ulasan aplikasi mobile.
4. Membangun, mengevaluasi, dan menginterpretasikan model klasifikasi sentimen multi-arsitektur (**CatBoost/XGBoost/LightGBM vs. IndoBERT**) lengkap dengan atribusi fitur berbasis **SHAP (*SHapley Additive exPlanations*)**.

---

## 4. Spesifikasi Raw Dataset (Data Nyata Hasil Scraping)

Dataset dikumpulkan secara langsung (*100% real scraped data*, tanpa data sintetis/dummy) dari server **Google Play Store** menggunakan pustaka `google-play-scraper` dengan mekanisme *high-yield pagination* (`continuation_token` per batch 200 ulasan, kombinasi `Sort.NEWEST`, `Sort.MOST_RELEVANT`, dan iterasi `filter_score_with` bintang 1 s/d 5) serta telah melalui proses deduplikasi ketat berdasarkan `reviewId`.

### A. Ringkasan Metrik Utama Dataset
- **TOTAL RAW DATA (Baris Unik Eksak)**: **`{total_rows_str}` baris unik** (Melampaui batas minimum kelas 37.455 baris).
- **Jumlah Kolom Atribut**: **`{total_cols}` kolom**.
- **Rentang Waktu Ulasan Aktual**: **`{min_date}` hingga `{max_date}`**.
- **File Raw Dataset Utama**: `./dataset/RAW_DATASET_GENAI_INDONESIA_AM522B.csv`

### B. Tabel Rincian Jumlah Record per Aplikasi dan Rentang Waktu Data

| No | Nama Aplikasi (`app_name`) | Package ID (`app_id`) | Jumlah Baris Unik | Proporsi (%) | Rata-rata Rating | Rentang Tanggal Ulasan |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
{app_table_md}
| **TOTAL** | **Ekosistem Generative AI ({len(app_breakdown)} Aplikasi)** | **Terdeduplikasi (`reviewId` Unik)** | **{total_rows_str}** | **100.00%** | **-** | **`{min_date}` s/d `{max_date}`** |

### C. Tabel Distribusi Rating Bintang (Skor 1 s/d 5)

| Skor Bintang (`score`) | Kategori Kepuasan Eksplisit | Jumlah Baris | Proporsi (%) |
| :---: | :--- | :---: | :---: |
{score_table_md}

### D. Kamus Data (*Data Dictionary*) Raw Dataset

| No | Nama Kolom | Tipe Data | Deskripsi dan Peran Analitis dalam Penelitian |
| :---: | :--- | :---: | :--- |
| 1 | `reviewId` | `String (UUID)` | Identifikasi unik (*Primary Key*) dari server Google Play Store untuk menjamin integritas 0% duplikasi baris. |
| 2 | `app_name` | `String` | Nama resmi aplikasi Generative AI yang dianalisis (mis. *ChatGPT (OpenAI)*, *Google Gemini*, *DeepSeek - AI Assistant*). |
| 3 | `app_id` | `String` | Package identifier aplikasi di Google Play Store (mis. `com.openai.chatgpt`, `com.google.android.apps.bard`). |
| 4 | `userName` | `String` | Nama tampilan pengguna Google yang memberikan ulasan di Play Store. |
| 5 | `content` | `String (Text)` | Teks asli ulasan pengguna (*raw review text*) yang menjadi korpus utama analisis NLP, ABSA, dan Topic Modeling. |
| 6 | `score` | `Integer (1-5)` | Skor rating bintang (skala 1 hingga 5) yang diberikan pengguna; digunakan sebagai target eksplisit & analisis *discrepancy*. |
| 7 | `thumbsUpCount` | `Integer` | Jumlah dukungan (*likes/helpful votes*) dari pengguna lain terhadap ulasan tersebut; indikator bobot dampak keluhan/pujian. |
| 8 | `reviewCreatedVersion` | `String` | Versi rilis aplikasi saat ulasan ditulis; krusial untuk melacak bug spesifik versi atau dampak pembaruan fitur. |
| 9 | `at` | `Datetime` | Timestamp eksak (`YYYY-MM-DD HH:MM:SS`) saat ulasan dipublikasikan; digunakan untuk analisis deret waktu (*time-series*). |
| 10 | `replyContent` | `String (Nullable)` | Teks balasan resmi dari pengembang (*developer response*) jika ada; indikator responsivitas layanan pelanggan. |
| 11 | `language_region` | `String` | Label transparan kode bahasa dan region pengambilan ulasan (`id_ID` / `en_ID`) pada Google Play Store. |

---

## 5. Arsitektur Metodologi Data Science (Framework CRISP-DM)

Penelitian ini dirancang menggunakan metodologi standar industri **CRISP-DM (*Cross-Industry Standard Process for Data Mining*)** yang diadaptasi khusus untuk *Natural Language Processing (NLP)* dan *Predictive Text Analytics* berskala besar:

### Tahap 1: Business & Data Understanding
- Memetakan lanskap kompetisi ekosistem aplikasi Generative AI di Indonesia (Model Flagship Umum vs. Reasoning/Search AI vs. Companion/Multi-bot AI).
- Merumuskan taksonomi 4 aspek inti evaluasi layanan AI: **(1) Akurasi, Penalaran & Halusinasi**, **(2) Kuota Gratis, Paywall & Harga Langganan**, **(3) UI/UX, Stabilitas Server & Autentikasi**, serta **(4) Kapabilitas Bahasa Lokal & Fitur Suara/Multimodal**.

### Tahap 2: Data Engineering & Text Preprocessing
1. **Data Integrity & Deduplication**: Validasi skema kolom, penghapusan ulasan kosong, dan deduplikasi berbasis `reviewId` atas {total_rows_str} baris data mentah.
2. **Colloquial Indonesian Text Normalization**: Pembersihan karakter non-alfanumerik/emoji berlebih, *case folding*, serta normalisasi kata tidak baku/slang Bahasa Indonesia (mis. *"gk" -> "tidak"*, *"bgt" -> "banget"*, *"lemot" -> "lambat"*, *"eror" -> "error"*) menggunakan kamus *Colloquial Indonesian Lexicon*.
3. **Tokenization, Stopword Removal & Stemming Selektif**: Mempertahankan kata negasi (*"tidak"*, *"kurang"*, *"jangan"*, *"bukan"*) agar konteks sentimen tidak terbalik saat pembentukan fitur.
4. **Pelabelan *Rating-Sentiment Discrepancy***: Mengombinasikan skor leksikon sentimen (*InSet Lexicon* / VADER multilingual) dengan skor bintang (`score`) untuk menandai ulasan yang mengalami kontradiksi antara rating bintang dan polaritas teks (mis. Rating 5 tetapi teks bernada keluhan negatif, atau Rating 1 dengan teks positif).

### Tahap 3: Exploratory Data Analysis (EDA)
1. **Time-Series Review Surge Analysis**: Analisis tren volume ulasan dan fluktuasi rating mingguan/bulanan untuk mendeteksi anomali lonjakan ulasan (*surge*) saat momen krusial, seperti rilis **GPT-4o**, peluncuran viral **DeepSeek-R1**, rebranding **Bard menjadi Google Gemini**, hingga insiden gangguan server global.
2. **Comparative N-Gram (Unigram, Bigram, Trigram) Analysis**: Membandingkan frasa keluhan spesifik antar-aplikasi (mis. *"limit habis"*, *"login gagal"*, *"jawaban ngawur"*, *"server sibuk"*).
3. **Review Length & Helpfulness Distribution**: Menganalisis korelasi antara panjang karakter ulasan, skor bintang, dan jumlah `thumbsUpCount` (ulasan negatif panjang cenderung memperoleh *thumbs up* jauh lebih tinggi).

### Tahap 4: Dual-Stage Modeling (Eksperimen Komparatif)

| Tahap Eksperimen | Fokus Pemodelan | Algoritma & Arsitektur yang Dibandingkan | Rekayasa Fitur (Feature Engineering) |
| :---: | :--- | :--- | :--- |
| **Experiment 1** | **Sentiment & Rating-Discrepancy Classification** | **Ensemble Boosting**: `CatBoost`, `XGBoost`, `LightGBM`<br>**Deep Learning Transformer**: `IndoBERT` (`indobenchmark/indobert-base-p1`) | TF-IDF (1-3 n-grams) + Lexicon Polarity Score + Metadata (`review_length`, `thumbsUpCount`) vs. Contextual Dense Embeddings (768-dim) |
| **Experiment 2** | **Topic Modeling & Aspect-Based Sentiment Extraction (ABSA)** | **Neural Topic Modeling**: `BERTopic` (Sentence-Transformers + UMAP + HDBSCAN + c-TF-IDF)<br>**Probabilistic Baseline**: `LDA` (*Latent Dirichlet Allocation*) | Klasterisasi topik keluhan & pujian per aplikasi, dipetakan ke 4 dimensi aspek layanan AI untuk menghitung *Net Sentiment Score* per aspek |

### Tahap 5: Model Evaluation & Explainability (XAI)
1. **Evaluasi Performa Klasifikasi (Experiment 1)**: Menggunakan skema *Stratified K-Fold Cross Validation* dengan metrik **Accuracy**, **Precision**, **Recall**, **Macro F1-Score** (mengatasi *class imbalance* pada rating bintang 5 vs 1–4), **ROC-AUC (One-vs-Rest)**, serta visualisasi **Confusion Matrix**.
2. **Evaluasi Koherensi Topik (Experiment 2)**: Mengukur kualitas pemisahan topik antara BERTopic dan LDA menggunakan metrik **Topic Coherence ($C_v$)** dan **Topic Diversity**.
3. **Explainable AI dengan SHAP (*SHapley Additive exPlanations*)**: Menghitung nilai kontribusi marginal (*SHAP values*) dari setiap fitur kata kunci dan variabel metadata terhadap probabilitas prediksi ketidakpuasan pengguna, sehingga menghasilkan penjelasan transparan mengenai pendorong utama churn/komplain pada tiap aplikasi AI.

---

## 6. Expected Output & Kontribusi Praktis

1. **Matriks Benchmarking Komparatif Ekosistem Generative AI Indonesia**: Peta kuadran kinerja pengalaman pengguna lintas aplikasi (ChatGPT, Gemini, DeepSeek, Copilot, Perplexity, Claude, Poe, Character.AI, dll.) berdasarkan *Net Sentiment Score* pada tiap aspek layanan (Akurasi, Harga/Kuota, UI/UX & Kecepatan, Bahasa Lokal).
2. **Sistem Deteksi *Rating-Text Discrepancy* & Prioritisasi Komplain Kritis**: Model terlatih yang mampu menyaring ulasan "bintang palsu/bias" dan mengurutkan keluhan teknis paling berdampak (berdasarkan prediksi sentimen negatif + bobot `thumbsUpCount`) bagi tim *Product Manager* dan *AI Engineer*.
3. **Rekomendasi Strategis Berbasis Bukti (*Actionable Insights*)**: Panduan empiris bagi pengembang aplikasi AI dan pemangku kepentingan di Indonesia mengenai fitur apa yang paling disukai pengguna lokal serta friksi teknis apa yang paling mendesak untuk diperbaiki.
"""
    with open(MD_OUT_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[OK] File Markdown berhasil dibuat: {MD_OUT_PATH}")


def add_styled_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    hdr_cells = table.rows[0].cells
    for i, h_text in enumerate(headers):
        hdr_cells[i].text = ""
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h_text)
        run.bold = True
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(255, 255, 255)
        set_cell_background(hdr_cells[i], "1F4E79")

    for r_idx, row_data in enumerate(rows):
        row_cells = table.add_row().cells
        bg_color = "F2F5F9" if r_idx % 2 == 1 else "FFFFFF"
        is_total_row = (r_idx == len(rows) - 1) and ("TOTAL" in str(row_data[0]) or "TOTAL" in str(row_data[1]))
        if is_total_row:
            bg_color = "D9E1F2"

        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = ""
            p = row_cells[c_idx].paragraphs[0]
            if c_idx in [0, 3, 4, 5]:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(str(val))
            run.font.size = Pt(9)
            if is_total_row or c_idx == 1:
                run.bold = True
            set_cell_background(row_cells[c_idx], bg_color)

    if col_widths:
        for row in table.rows:
            for idx, width in enumerate(col_widths):
                row.cells[idx].width = Inches(width)

    doc.add_paragraph()
    return table


def build_docx(summary):
    total_rows = summary["total_rows"]
    total_rows_str = format_number_id(total_rows)
    total_cols = summary["total_cols"]
    min_date = summary["min_date"][:10]
    max_date = summary["max_date"][:10]
    app_breakdown = summary["app_breakdown"]
    score_dist = summary["score_distribution"]

    doc = Document()

    # Atur margin halaman
    for section in doc.sections:
        section.top_margin = Inches(0.85)
        section.bottom_margin = Inches(0.85)
        section.left_margin = Inches(0.9)
        section.right_margin = Inches(0.9)

    # Judul Dokumen Header
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = title_p.add_run("DOKUMEN SINOPSIS PENGAJUAN PROYEK INDIVIDU\nMATA KULIAH DATA SCIENCE (AM522B)\n")
    r_sub.bold = True
    r_sub.font.size = Pt(11)
    r_sub.font.color.rgb = RGBColor(31, 78, 121)

    r_main = title_p.add_run(
        "\n\"Analisis Komparatif Sentimen Berbasis Aspek (ABSA) dan Pemodelan Topik Pengalaman Pengguna "
        "terhadap Ekosistem Aplikasi Mobile Generative AI di Indonesia\""
    )
    r_main.bold = True
    r_main.font.size = Pt(14)
    r_main.font.color.rgb = RGBColor(15, 36, 62)

    meta_p = doc.add_paragraph()
    meta_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_meta = meta_p.add_run(
        f"Total Raw Dataset: {total_rows_str} Ulasan Unik Google Play Store | "
        f"Periode Data: {min_date} s/d {max_date} | Status Data: 100% Real Scraped Data"
    )
    r_meta.italic = True
    r_meta.font.size = Pt(9.5)
    r_meta.font.color.rgb = RGBColor(89, 89, 89)

    # -------------------------------------------------------------------------
    # BAGIAN 1: JUDUL USULAN PROYEK
    # -------------------------------------------------------------------------
    h1 = doc.add_heading("1. Judul Usulan Proyek", level=1)
    h1.runs[0].font.color.rgb = RGBColor(31, 78, 121)

    p = doc.add_paragraph()
    r = p.add_run("Judul Utama (Rekomendasi Utama):\n")
    r.bold = True
    r2 = p.add_run(
        "\"Analisis Komparatif Sentimen Berbasis Aspek (ABSA) dan Pemodelan Topik Pengalaman Pengguna "
        "terhadap Ekosistem Aplikasi Mobile Generative AI di Indonesia\""
    )
    r2.bold = True
    r2.italic = True

    p_alt = doc.add_paragraph()
    r_alt = p_alt.add_run("Variasi Judul Alternatif:")
    r_alt.bold = True

    b1 = doc.add_paragraph(style="List Bullet")
    b1.add_run("Alternatif 1 (Fokus Predictive Satisfaction & Discrepancy Detection): ").bold = True
    b1.add_run(
        "\"Prediksi Ketidakpuasan Pengguna dan Deteksi Anomali Rating-Ulasan (Rating-Text Discrepancy) "
        "pada Aplikasi Mobile Generative AI di Indonesia Menggunakan Ensemble Gradient Boosting dan SHAP Explainability\""
    )

    b2 = doc.add_paragraph(style="List Bullet")
    b2.add_run("Alternatif 2 (Fokus NLP Transformer Benchmarking & Topic Discovery): ").bold = True
    b2.add_run(
        f"\"Studi Komparasi Arsitektur Pre-Trained IndoBERT dan Gradient Boosting serta Ekstraksi Topik Berbasis BERTopic "
        f"pada {total_rows_str} Ulasan Pengguna Aplikasi Generative AI di Google Play Store Indonesia\""
    )

    # -------------------------------------------------------------------------
    # BAGIAN 2: LATAR BELAKANG & URGENSI PENELITIAN
    # -------------------------------------------------------------------------
    h2 = doc.add_heading("2. Latar Belakang & Urgensi Penelitian", level=1)
    h2.runs[0].font.color.rgb = RGBColor(31, 78, 121)

    doc.add_paragraph(
        "Ekosistem komputasi personal dan produktivitas digital di Indonesia tengah mengalami transformasi struktural "
        "seiring ledakan adopsi aplikasi Mobile Generative AI (Large Language Models dan asisten percakapan multimodal). "
        "Platform global seperti ChatGPT (OpenAI), Google Gemini, DeepSeek, Microsoft Copilot, Perplexity, Claude (Anthropic), "
        "Poe, hingga Character.AI dan Qwen kini digunakan secara masif oleh pelajar, mahasiswa, praktisi teknologi, "
        "hingga masyarakat umum di Indonesia untuk mendukung penulisan akademik, pemrograman (coding), pencarian informasi "
        "real-time, serta otomatisasi pekerjaan sehari-hari."
    )

    doc.add_paragraph(
        "Meskipun tingkat unduhan aplikasi Generative AI di Indonesia tumbuh secara eksponensial, kompetisi ketat antar-platform "
        "memunculkan dinamika pengalaman pengguna (User Experience) yang sangat kompleks dan terpolarisasi di Google Play Store:"
    )

    urg_items = [
        (
            "Kualitas Pemahaman Konteks & Bahasa Indonesia: ",
            "Pengguna di Indonesia kerap mengeluhkan ketimpangan kemampuan model dalam memahami struktur bahasa kasual/slang lokal, "
            "konteks spesifik Indonesia, hingga fenomena halusinasi model (hallucination) saat menjawab pertanyaan faktual atau teknis.",
        ),
        (
            "Friksi Monetisasi, Batasan Kuota Gratis, dan Harga Langganan: ",
            "Kebijakan pembatasan pesan harian (rate limit pada model flagship seperti GPT-4o, Claude Sonnet, atau pembuatan gambar) "
            "serta tingginya harga langganan bulanan (Plus/Pro tier) terhadap daya beli pengguna lokal memicu pergeseran pengguna "
            "ke alternatif gratis seperti DeepSeek.",
        ),
        (
            "Reliabilitas Teknis, Latensi Aplikasi Mobile, dan Fitur Multimodal: ",
            "Keluhan terkait kegagalan autentikasi login, server sibuk (server down/busy saat lonjakan trafik rilis model baru), "
            "fitur voice/percakapan suara yang patah-patah, hingga perubahan UI/UX berdampak langsung pada kepuasan pengguna.",
        ),
        (
            "Fenomena Rating-Text Discrepancy (Ketidaksesuaian Bintang vs. Teks): ",
            "Banyak pengguna memberikan rating bintang 5 tetapi menuliskan keluhan kritis di dalam teks ulasan (atau sebaliknya). "
            "Hal ini menyebabkan metrik rata-rata bintang semata menjadi bias dan tidak mencerminkan kualitas aspek layanan yang sebenarnya.",
        ),
    ]
    for title_txt, body_txt in urg_items:
        bp = doc.add_paragraph(style="List Bullet")
        bp.add_run(title_txt).bold = True
        bp.add_run(body_txt)

    # -------------------------------------------------------------------------
    # BAGIAN 3: RUMUSAN MASALAH & TUJUAN ANALISIS
    # -------------------------------------------------------------------------
    h3 = doc.add_heading("3. Rumusan Masalah & Tujuan Analisis", level=1)
    h3.runs[0].font.color.rgb = RGBColor(31, 78, 121)

    p_rm = doc.add_paragraph()
    p_rm.add_run("A. Rumusan Masalah Penelitian").bold = True

    rm_list = [
        "Bagaimana perbandingan distribusi polarisasi sentimen (positif, netral, negatif) dan tingkat kepuasan pengguna antar-aplikasi mobile Generative AI di Indonesia?",
        "Aspek layanan apa (Akurasi/Halusinasi Model, Harga/Limitasi Kuota Gratis, UI/UX & Kecepatan Server, serta Kemampuan Bahasa Lokal & Multimodal) yang paling memicu sentimen negatif maupun positif pada masing-masing platform?",
        "Seberapa besar ketidaksesuaian (rating-text discrepancy) antara skor bintang numerik (1-5) yang diberikan pengguna dengan polaritas sentimen aktual pada isi teks ulasannya?",
        "Bagaimana perbandingan performa model Ensemble (CatBoost, XGBoost, LightGBM) melawan Pre-trained Transformer (IndoBERT) dalam mengklasifikasikan sentimen dan mendeteksi discrepancy, serta kata kunci apa yang paling dominan menurut analisis SHAP?",
    ]
    for item in rm_list:
        doc.add_paragraph(item, style="List Number")

    p_tj = doc.add_paragraph()
    p_tj.add_run("B. Tujuan Analisis").bold = True

    tj_list = [
        f"Melakukan pemetaan komparatif polarisasi sentimen dan kepuasan pengguna lintas aplikasi Generative AI di Indonesia menggunakan {total_rows_str} baris data ulasan nyata.",
        "Mengekstraksi topik keluhan dan pujian utama menggunakan BERTopic dan LDA untuk memetakan kekuatan serta kelemahan spesifik tiap aplikasi pada 4 dimensi aspek layanan.",
        "Mengidentifikasi dan mengkuantifikasi proporsi rating-text discrepancy guna menghasilkan metrik kepuasan pengguna yang bebas bias.",
        "Membangun, mengevaluasi, dan menginterpretasikan model klasifikasi berbasis IndoBERT dan Gradient Boosting lengkap dengan analisis SHAP (SHapley Additive exPlanations).",
    ]
    for item in tj_list:
        doc.add_paragraph(item, style="List Number")

    # -------------------------------------------------------------------------
    # BAGIAN 4: SPESIFIKASI RAW DATASET (DATA NYATA HASIL SCRAPING)
    # -------------------------------------------------------------------------
    h4 = doc.add_heading("4. Spesifikasi Raw Dataset (Data Nyata Hasil Scraping)", level=1)
    h4.runs[0].font.color.rgb = RGBColor(31, 78, 121)

    doc.add_paragraph(
        f"Seluruh data dalam penelitian ini merupakan 100% Real Scraped Data yang ditarik langsung dari server Google Play Store "
        f"menggunakan pustaka google-play-scraper (tanpa data dummy atau sintetis). Pengambilan data dilakukan menggunakan "
        f"strategi high-yield batch continuation_token (count=200) dengan mengiterasi kombinasi Sort.NEWEST, Sort.MOST_RELEVANT, "
        f"serta filter bintang (1 s/d 5) dan telah dideduplikasi penuh berdasarkan reviewId unik."
    )

    spec_bullets = [
        ("TOTAL RAW DATA (Eksak): ", f"{total_rows_str} baris ulasan unik (Melampaui baseline kelas 37.455 baris)."),
        ("Dimensi Kolom: ", f"{total_cols} atribut kolom lengkap per baris."),
        ("Rentang Tanggal Ulasan: ", f"{min_date} hingga {max_date}."),
        ("Lokasi File Dataset: ", "./dataset/RAW_DATASET_GENAI_INDONESIA_AM522B.csv"),
    ]
    for k, v in spec_bullets:
        sp = doc.add_paragraph(style="List Bullet")
        sp.add_run(k).bold = True
        sp.add_run(v)

    p_t1 = doc.add_paragraph()
    p_t1.add_run("Tabel 4.1 Rincian Jumlah Record per Aplikasi dan Rentang Waktu Data").bold = True

    t1_headers = ["No", "Nama Aplikasi", "Package ID (app_id)", "Jumlah Baris", "Proporsi", "Rata-rata Rating", "Rentang Waktu Data"]
    t1_rows = []
    for idx, item in enumerate(app_breakdown, 1):
        t1_rows.append(
            [
                str(idx),
                item["app_name"],
                item["app_id"],
                format_number_id(item["count"]),
                f"{item['pct']:.2f}%",
                f"{item['avg_score']:.2f}",
                f"{item['min_date']} s/d {item['max_date']}",
            ]
        )
    t1_rows.append(
        [
            "TOTAL",
            f"Seluruh Aplikasi ({len(app_breakdown)} App)",
            "100% Unique reviewId",
            total_rows_str,
            "100.00%",
            "-",
            f"{min_date} s/d {max_date}",
        ]
    )
    add_styled_table(doc, t1_headers, t1_rows)

    p_t2 = doc.add_paragraph()
    p_t2.add_run("Tabel 4.2 Distribusi Rating Bintang (Skor 1 s/d 5)").bold = True
    t2_headers = ["Skor Bintang", "Kategori Kepuasan Eksplisit", "Jumlah Baris Unik", "Proporsi (%)"]
    t2_rows = []
    for star in ["5", "4", "3", "2", "1"]:
        if star in score_dist:
            label = {
                "5": "Bintang 5 - Sangat Puas",
                "4": "Bintang 4 - Puas",
                "3": "Bintang 3 - Netral / Ragu-ragu",
                "2": "Bintang 2 - Tidak Puas",
                "1": "Bintang 1 - Sangat Kecewa / Komplain Kritis",
            }[star]
            t2_rows.append(
                [
                    f"Bintang {star}",
                    label,
                    format_number_id(score_dist[star]["count"]),
                    f"{score_dist[star]['pct']:.2f}%",
                ]
            )
    add_styled_table(doc, t2_headers, t2_rows)

    p_t3 = doc.add_paragraph()
    p_t3.add_run("Tabel 4.3 Kamus Data (Data Dictionary) Raw Dataset").bold = True
    t3_headers = ["No", "Nama Kolom", "Tipe Data", "Deskripsi & Fungsi dalam Pipeline Data Science"]
    t3_rows = [
        ["1", "reviewId", "String (UUID)", "Identifikasi unik (Primary Key) ulasan dari Google Play Store untuk menjamin 0% duplikasi."],
        ["2", "app_name", "String", "Nama aplikasi Generative AI (ChatGPT, Google Gemini, DeepSeek, Copilot, Perplexity, Claude, Poe, Character.AI, dll.)."],
        ["3", "app_id", "String", "Package ID resmi aplikasi di server Google Play Store."],
        ["4", "userName", "String", "Nama akun pengguna Google yang mempublikasikan ulasan."],
        ["5", "content", "String (Text)", "Teks asli ulasan pengguna (raw text) sebagai input utama pemodelan ABSA, IndoBERT, dan BERTopic."],
        ["6", "score", "Integer (1-5)", "Rating bintang numerik (1 s/d 5) untuk evaluasi kepuasan eksplisit dan deteksi Rating-Text Discrepancy."],
        ["7", "thumbsUpCount", "Integer", "Jumlah likes/dukungan pembaca lain pada ulasan; digunakan sebagai indikator bobot urgensi keluhan."],
        ["8", "reviewCreatedVersion", "String", "Versi rilis aplikasi saat ulasan dikirimkan; memetakan regresi bug per pembaruan versi."],
        ["9", "at", "Datetime", "Timestamp asli ulasan dipublikasikan (YYYY-MM-DD HH:MM:SS) untuk analisis deret waktu (time-series)."],
        ["10", "replyContent", "String (Nullable)", "Teks tanggapan resmi dari pengembang aplikasi terhadap keluhan/ulasan pengguna."],
        ["11", "language_region", "String", "Penanda transparan kode bahasa dan region pengambilan ulasan di Google Play Store (id_ID / en_ID)."],
    ]
    add_styled_table(doc, t3_headers, t3_rows)

    # -------------------------------------------------------------------------
    # BAGIAN 5: ARSITEKTUR METODOLOGI DATA SCIENCE (FRAMEWORK CRISP-DM)
    # -------------------------------------------------------------------------
    h5 = doc.add_heading("5. Arsitektur Metodologi Data Science (Framework CRISP-DM)", level=1)
    h5.runs[0].font.color.rgb = RGBColor(31, 78, 121)

    crisp_stages = [
        (
            "1. Business & Data Understanding: ",
            "Memetakan lanskap kompetisi aplikasi Generative AI di Indonesia dan mendefinisikan 4 dimensi aspek utama pengalaman pengguna: "
            "(a) Akurasi & Halusinasi Model, (b) Harga & Limitasi Kuota Gratis, (c) UI/UX, Stabilitas Server & Login, serta "
            "(d) Kemampuan Bahasa Indonesia & Fitur Suara/Multimodal.",
        ),
        (
            "2. Data Engineering & Text Preprocessing: ",
            "Meliputi deduplikasi berbasis reviewId, pembersihan karakter noise, normalisasi kata slang/singkatan Bahasa Indonesia "
            "(colloquial Indonesian normalization menggunakan kamus baku-slang), tokenisasi dengan preservasi kata negasi, "
            "serta pelabelan otomatis Rating-Sentiment Discrepancy (ketidaksesuaian antara skor bintang dan polaritas leksikon teks).",
        ),
        (
            "3. Exploratory Data Analysis (EDA): ",
            "Melakukan analisis deret waktu (time-series review surge) saat momen rilis model besar (seperti GPT-4o, DeepSeek-R1, dan Gemini Pro), "
            "analisis frekuensi N-Gram (Unigram, Bigram, Trigram) per aplikasi, serta analisis distribusi panjang ulasan terhadap jumlah thumbsUpCount.",
        ),
        (
            "4. Dual-Stage Modeling (Eksperimen Komparatif): ",
            "Menerapkan dua eksperimen pemodelan terintegrasi: "
            "[Experiment 1 - Sentiment & Discrepancy Classification] membandingkan algoritma Ensemble Gradient Boosting (CatBoost, XGBoost, LightGBM "
            "dengan fitur TF-IDF + Lexicon + Metadata) melawan Pre-trained Transformer IndoBERT. "
            "[Experiment 2 - Topic Modeling & Aspect Extraction] mengekstraksi klaster topik keluhan dan pujian menggunakan BERTopic "
            "yang dikomparasikan dengan LDA (Latent Dirichlet Allocation).",
        ),
        (
            "5. Model Evaluation & Explainability (SHAP): ",
            "Mengukur kinerja klasifikasi dengan Accuracy, Macro F1-Score, ROC-AUC, dan Confusion Matrix, mengevaluasi koherensi topik (Cv Coherence), "
            "serta menerapkan SHAP (SHapley Additive exPlanations) untuk mengidentifikasi kata kunci dan fitur pendorong utama ketidakpuasan pengguna.",
        ),
    ]
    for st_title, st_desc in crisp_stages:
        cp = doc.add_paragraph(style="List Bullet")
        cp.add_run(st_title).bold = True
        cp.add_run(st_desc)

    # -------------------------------------------------------------------------
    # BAGIAN 6: EXPECTED OUTPUT & KONTRIBUSI PRAKTIS
    # -------------------------------------------------------------------------
    h6 = doc.add_heading("6. Expected Output & Kontribusi Praktis", level=1)
    h6.runs[0].font.color.rgb = RGBColor(31, 78, 121)

    outputs = [
        (
            "Dashboard & Matriks Komparasi Performa Pengalaman Pengguna: ",
            "Menyajikan peta perbandingan skor sentimen berbasis aspek (ABSA) lintas aplikasi Generative AI di Indonesia "
            "sehingga terlihat jelas keunggulan dan kelemahan spesifik tiap kompetitor (ChatGPT, Gemini, DeepSeek, Copilot, Perplexity, Claude, Poe, Character.AI).",
        ),
        (
            "Framework Deteksi Rating-Text Discrepancy: ",
            "Memberikan metode penyaringan ulasan bias pada toko aplikasi guna membantu tim pengembang memprioritaskan keluhan teknis "
            "yang benar-benar kritis meskipun diberi bintang tinggi atau sebaliknya.",
        ),
        (
            "Wawasan Strategis Pengembangan Produk AI Lokal: ",
            "Menghasilkan rekomendasi berbasis bukti (explainable SHAP insights) mengenai ekspektasi pengguna Indonesia terhadap "
            "akurasi Bahasa Indonesia, batas kuota freemium, dan stabilitas aplikasi mobile.",
        ),
    ]
    for o_title, o_desc in outputs:
        op = doc.add_paragraph(style="List Bullet")
        op.add_run(o_title).bold = True
        op.add_run(o_desc)

    doc.save(DOCX_OUT_PATH)
    print(f"[OK] File Word (.docx) berhasil dibuat: {DOCX_OUT_PATH}")


def main():
    if not os.path.exists(SUMMARY_JSON_PATH):
        raise FileNotFoundError(f"File {SUMMARY_JSON_PATH} belum tersedia. Jalankan verify_dataset.py terlebih dahulu.")
    with open(SUMMARY_JSON_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)
    build_markdown(summary)
    build_docx(summary)


if __name__ == "__main__":
    main()
