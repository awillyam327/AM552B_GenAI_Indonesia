import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCX_OUT = os.path.join(ROOT_DIR, "TEAM_ONBOARDING_GUIDE.docx")


def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill_hex)
    tcPr.append(shd)


def add_styled_table(doc, headers, rows):
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
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = ""
            p = row_cells[c_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(str(val))
            run.font.size = Pt(9)
            if c_idx == 0:
                run.bold = True
            set_cell_background(row_cells[c_idx], bg_color)

    doc.add_paragraph()
    return table


def main():
    doc = Document()
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)

    # Header Block
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = title_p.add_run("PANDUAN ONBOARDING TIM & KURIKULUM BELAJAR KILAT\nPROYEK TUGAS AKHIR DATA SCIENCE (AM522B)\n")
    r_sub.bold = True
    r_sub.font.size = Pt(11)
    r_sub.font.color.rgb = RGBColor(31, 78, 121)

    r_main = title_p.add_run(
        "\n\"Analisis Komparatif Sentimen Berbasis Aspek (ABSA) dan Pemodelan Topik Pengalaman Pengguna "
        "terhadap Ekosistem Aplikasi Mobile Generative AI di Indonesia\""
    )
    r_main.bold = True
    r_main.font.size = Pt(13.5)
    r_main.font.color.rgb = RGBColor(15, 36, 62)

    meta_p = doc.add_paragraph()
    meta_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_meta = meta_p.add_run(
        "Komposisi Tim: 3 Anggota | Dataset: 71.221 Ulasan Unik Google Play Store Indonesia (8 Aplikasi Generative AI)"
    )
    r_meta.italic = True
    r_meta.font.size = Pt(9.5)
    r_meta.font.color.rgb = RGBColor(89, 89, 89)

    # 1. THE BIG PICTURE
    h1 = doc.add_heading("1. Penjelasan Topik Tanpa Jargon Rumit (The Big Picture)", level=1)
    h1.runs[0].font.color.rgb = RGBColor(31, 78, 121)

    doc.add_paragraph(
        "Proyek ini meneliti bagaimana pengalaman nyata pengguna di Indonesia terhadap 8 aplikasi Mobile Generative AI "
        "terpopuler di Google Play Store: ChatGPT (22.000 ulasan), Google Gemini (12.600 ulasan), Character.AI (12.000 ulasan), "
        "Perplexity (8.043 ulasan), Microsoft Copilot (5.459 ulasan), DeepSeek (5.374 ulasan), Claude by Anthropic (4.009 ulasan), "
        "dan Poe (1.736 ulasan) — dengan total 71.221 baris data asli."
    )
    doc.add_paragraph(
        "Mengapa proyek ini sangat menarik? Karena skor rata-rata bintang di Google Play Store sering kali menipu. "
        "Banyak aplikasi memiliki rating tinggi di atas 4.3, tetapi menyimpan ribuan keluhan kritis terkait batasan kuota gratis, "
        "harga langganan Plus/Pro yang mahal, bug login, server sibuk, hingga jawaban Bahasa Indonesia yang kaku atau mengalami "
        "halusinasi (mengarang fakta). Sebaliknya, Character.AI mengalami kejatuhan rating drastis (rata-rata 1.58) akibat gelombang "
        "protes fitur filter dan perubahan UI."
    )

    p_concepts = doc.add_paragraph()
    p_concepts.add_run("Intuisi 4 Konsep Inti yang Kita Gunakan:").bold = True

    concepts = [
        (
            "a. Sentiment Analysis vs. Aspect-Based Sentiment Analysis (ABSA): ",
            "Analisis sentimen biasa hanya memberi label 'Positif' atau 'Negatif' pada satu ulasan utuh. Padahal satu ulasan bisa "
            "memuji kepintaran AI sekaligus mengeluhkan kuota gratis yang pelit. Dengan ABSA, kita membedah sentimen ke dalam "
            "4 aspek spesifik: (1) Akurasi & Kecerdasan AI, (2) Harga & Limitasi Kuota Gratis, (3) Performa, Kecepatan & Bug/UI Aplikasi, "
            "serta (4) Dukungan & Keluwesan Bahasa Indonesia.",
        ),
        (
            "b. Rating-Text Discrepancy (Ketidaksesuaian Bintang vs. Isi Ulasan): ",
            "Fenomena ketika pengguna memberi Bintang 5 tetapi isi ulasannya berisi komplain/keluhan teknis (misal: 'Kasih bintang 5 biar "
            "dibaca developer, aplikasi sering error'), atau memberi Bintang 1 padahal isi teksnya memuji. Kita mendeteksi anomali ini "
            "agar evaluasi kualitas aplikasi menjadi jauh lebih akurat.",
        ),
        (
            "c. Topic Modeling (LDA & BERTopic): ",
            "Teknik kecerdasan buatan tanpa pengawasan (unsupervised learning) untuk mengelompokkan 71.221 ulasan secara otomatis ke dalam "
            "topik-topik perbincangan utama (misal: topik keluhan server sibuk DeepSeek, topik sensor Character.AI, topik bantuan tugas kuliah).",
        ),
        (
            "d. Explainable AI dengan SHAP (SHapley Additive exPlanations): ",
            "Teknik untuk membuka 'kotak hitam' model Machine Learning kita dan membuktikan secara visual kata kunci atau fitur apa saja "
            "(seperti kata 'limit', 'bayar', 'error', 'macet') yang paling kuat mendorong prediksi ketidakpuasan pengguna.",
        ),
    ]
    for title_txt, desc_txt in concepts:
        bp = doc.add_paragraph(style="List Bullet")
        bp.add_run(title_txt).bold = True
        bp.add_run(desc_txt)

    # 2. KURIKULUM BELAJAR KILAT
    h2 = doc.add_heading("2. Kurikulum Belajar Terarah (Learning Checklist per Peran)", level=1)
    h2.runs[0].font.color.rgb = RGBColor(31, 78, 121)

    p_m0 = doc.add_paragraph()
    p_m0.add_run("A. Materi Wajib Bersama (Seluruh Anggota Tim - Estimasi 3 Jam):").bold = True
    common_items = [
        "Dasar Python Pandas: pd.read_csv(), filtering baris, groupby(), value_counts(), dan penanganan missing values.",
        "Memahami 11 kolom dataset kita: reviewId, app_name, app_id, userName, content, score, thumbsUpCount, reviewCreatedVersion, at, replyContent, language_region.",
        "Konsep dasar NLP Bahasa Indonesia: Case Folding, Tokenization, Stopwords (dengan mempertahankan kata negasi seperti 'tidak', 'kurang', 'jangan'), Slang Normalization, dan TF-IDF.",
    ]
    for item in common_items:
        doc.add_paragraph(item, style="List Bullet")

    p_m1 = doc.add_paragraph()
    p_m1.add_run("B. Materi Spesifik Sesuai Peran:").bold = True
    role_curricula = [
        (
            "Member 1 (Project Lead, Predictive Modeling & SHAP Explainability): ",
            "Mempelajari Supervised Text Classification, penanganan Class Imbalance (Macro F1-Score, ROC-AUC), algoritma Gradient Boosting "
            "(CatBoost, XGBoost, LightGBM), fine-tuning/embedding IndoBERT (transformers), serta interpretabilitas model dengan library shap.",
        ),
        (
            "Member 2 (Data Engineering, Preprocessing & ABSA Tagging): ",
            "Mempelajari pembersihan teks ulasan Bahasa Indonesia (regex, penghapusan noise/emoji berlebih), normalisasi kata gaul/slang "
            "(Colloquial Indonesian Lexicon & PySastrawi), aturan ekstraksi 4 aspek ABSA, serta logika pelabelan Rating-Text Discrepancy.",
        ),
        (
            "Member 3 (Exploratory Data Analysis, Topic Modeling & Visualisasi/Laporan): ",
            "Mempelajari visualisasi data komparatif (matplotlib, seaborn), analisis frekuensi kata N-Gram (Unigram/Bigram/Trigram), "
            "analisis tren waktu (time-series review surge), serta pemodelan topik menggunakan Latent Dirichlet Allocation (scikit-learn) dan BERTopic.",
        ),
    ]
    for r_title, r_desc in role_curricula:
        rp = doc.add_paragraph(style="List Bullet")
        rp.add_run(r_title).bold = True
        rp.add_run(r_desc)

    # 3. PEMBAGIAN PERAN & TANGGUNG JAWAB
    h3 = doc.add_heading("3. Matriks Pembagian Peran & Tanggung Jawab (3 Anggota Tim)", level=1)
    h3.runs[0].font.color.rgb = RGBColor(31, 78, 121)

    headers = [
        "Dimensi Tugas",
        "Member 1: Project Lead, Modeling & Explainability",
        "Member 2: Data Engineering, Preprocessing & ABSA",
        "Member 3: EDA, Topic Modeling & Visualisasi/Laporan",
    ]
    rows = [
        [
            "Fase CRISP-DM Utama",
            "Fase 1 (Business Understanding), Fase 4 (Supervised Modeling), Fase 5 (Evaluation & SHAP)",
            "Fase 2 (Data Audit), Fase 3 (Data Preparation, Normalization & ABSA Tagging)",
            "Fase 2 (Exploratory Data Analysis), Fase 4 (Topic Modeling), Fase 6 (Reporting & Presentasi)",
        ],
        [
            "Folder Kerja di Workspace",
            "01_business_understanding/\n04_modeling/\n05_evaluation/",
            "03_data_preparation/\n04_modeling/ (ABSA Matrix)",
            "02_data_understanding/\n04_modeling/ (BERTopic/LDA)\n06_deployment_reporting/",
        ],
        [
            "Tanggung Jawab Teknis",
            "1. Merancang arsitektur CRISP-DM & KPI.\n2. Melatih model klasifikasi sentimen & discrepancy (CatBoost/XGBoost/LightGBM vs IndoBERT).\n3. Mengevaluasi Macro-F1, ROC-AUC, Confusion Matrix, dan SHAP.",
            "1. Membersihkan 71.221 teks ulasan.\n2. Membangun kamus normalisasi slang Bahasa Indonesia.\n3. Melabeli 4 aspek layanan (ABSA) dan mendeteksi Rating-Text Discrepancy.\n4. Menyiapkan fitur TF-IDF & Lexicon.",
            "1. Menjalankan EDA komparatif 8 aplikasi & tren waktu.\n2. Mengekstraksi topik keluhan & pujian dengan BERTopic dan LDA.\n3. Menyusun grafik dashboard insight & laporan akhir.",
        ],
        [
            "Library Utama",
            "scikit-learn, catboost, xgboost, lightgbm, transformers (IndoBERT), shap",
            "pandas, re, Sastrawi, nltk, scikit-learn (TfidfVectorizer)",
            "pandas, matplotlib, seaborn, wordcloud, bertopic, scikit-learn (LDA)",
        ],
    ]
    add_styled_table(doc, headers, rows)

    doc.save(DOCX_OUT)
    print(f"[OK] File Word Onboarding berhasil dibuat: {DOCX_OUT}")


if __name__ == "__main__":
    main()
