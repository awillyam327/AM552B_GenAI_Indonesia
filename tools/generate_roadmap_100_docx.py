"""
=========================================================================================
GENERATOR DOKUMEN WORD: ROADMAP_KERJA_KELOMPOK_100_PERSEN.docx
=========================================================================================
Mengonversi rencana kerja kelompok 6 Tahap menuju 100% selesai menjadi dokumen Word (.docx)
yang rapi, terformat tabel profesional, dan siap dibagikan ke WhatsApp kelompok / dicetak.
=========================================================================================
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DOCX = os.path.join(ROOT_DIR, "ROADMAP_KERJA_KELOMPOK_100_PERSEN.docx")


def set_cell_shading(cell, color_hex):
    shading = OxmlElement("w:shd")
    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:color"), "auto")
    shading.set(qn("w:fill"), color_hex)
    cell._tc.get_or_add_tcPr().append(shading)


def add_formatted_table(doc, headers, rows_data, header_bg="1B365D"):
    table = doc.add_table(rows=1 + len(rows_data), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header
    hdr_cells = table.rows[0].cells
    for i, h_text in enumerate(headers):
        hdr_cells[i].text = h_text
        set_cell_shading(hdr_cells[i], header_bg)
        for p in hdr_cells[i].paragraphs:
            for r in p.runs:
                r.bold = True
                r.font.color.rgb = RGBColor(255, 255, 255)
                r.font.size = Pt(9.5)

    # Rows
    for r_idx, r_data in enumerate(rows_data):
        row_cells = table.rows[r_idx + 1].cells
        bg = "F4F7FA" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(r_data):
            row_cells[c_idx].text = val
            set_cell_shading(row_cells[c_idx], bg)
            for p in row_cells[c_idx].paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
    doc.add_paragraph()
    return table


def generate_roadmap_docx():
    doc = docx.Document()

    # Page setup
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    # Title
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = title_p.add_run("ROADMAP & AGENDA KERJA KELOMPOK MENUJU 100% SELESAI\n(23 NOVEMBER – 30 NOVEMBER 2026)")
    r_title.bold = True
    r_title.font.size = Pt(15)
    r_title.font.color.rgb = RGBColor(27, 54, 93)

    sub_p = doc.add_paragraph()
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = sub_p.add_run(
        "Proyek Akhir Mata Kuliah AM552 - Data Science of IS (S1 Sistem Informasi, FTI UKSW)\n"
        "Topik: Analisis Komparatif Sentimen Berbasis Aspek (ABSA) & Pemodelan Topik Aplikasi Generative AI di Indonesia\n"
        "Deadline FLearn: 30 November 2026 Pukul 23.00 WIB | Penamaan File: 682024046_[NomorKelompok]_DS"
    )
    r_sub.font.size = Pt(10)
    r_sub.italic = True

    # Section 1: Team Composition
    h1 = doc.add_heading("1. Komposisi Tim & Pembagian Wilayah File (Zero-Overlap Rule)", level=1)
    doc.add_paragraph(
        "Agar tidak terjadi bentrok kode (merge conflict) di GitHub, setiap anggota memiliki wilayah file utama masing-masing:"
    )

    add_formatted_table(
        doc,
        ["Anggota", "NIM & Nama Lengkap", "Peran Utama dalam Proyek", "Wilayah File Utama yang Diedit", "Porsi"],
        [
            [
                "Member 1",
                "682024046\nArthur Willyam Liang",
                "Project Lead, ML Modeling, Tuning 5-Fold CV, Anti-Leakage Audit & Deployment",
                "1. notebooks/03_Modeling_Tuning_dan_Evaluasi.ipynb\n2. 05_evaluation/metric_sanity_audit.py\n3. models/genai_absa_pipeline.joblib",
                "34%",
            ],
            [
                "Member 2",
                "682024016\nBrendan Wilson Santoso",
                "Data Engineering, Text Preprocessing, Kamus Slang, ABSA & Testing Prototype",
                "1. notebooks/02_Data_Preparation_dan_ABSA.ipynb\n2. 03_data_preparation/slang_dictionary_id.json\n3. prototype/test_prototype_scenarios.py",
                "33%",
            ],
            [
                "Member 3",
                "682024059\nTiara Agustien Kusumawati",
                "Exploratory Data Analysis (EDA), Topic Discovery, Laporan 7 Bab & Slide PDF",
                "1. notebooks/01_Data_Understanding_dan_EDA.ipynb\n2. Laporan_Project_Akhir_AM552_Terisi.docx\n3. 06_deployment_reporting/682024046_KEL_DS_Presentasi.pdf",
                "33%",
            ],
        ],
    )

    # Section 2: Overview of 6 Stages
    doc.add_heading("2. Ringkasan 6 Tahap Pengerjaan Menuju 100% Selesai", level=1)
    add_formatted_table(
        doc,
        ["Tahap", "Jadwal Target", "Progress", "Fokus Utama Tahapan", "Output / Deliverable Konkret"],
        [
            [
                "TAHAP 1",
                "23 Nov 2026\n(Kerja Kelompok)",
                "65% -> 70%",
                "Onboarding Tim, Demo Web Live, Setup GitHub Desktop & Antigravity di 3 Laptop",
                "3 Laptop anggota sudah ter-clone repo GitHub & aktif dengan Master Prompt Antigravity masing-masing.",
            ],
            [
                "TAHAP 2",
                "23 – 24 Nov 2026",
                "70% -> 78%",
                "Pengayaan Kamus Slang Indonesia, Validasi 4 Aspek ABSA & Eksplorasi Data (EDA)",
                "slang_dictionary_id.json bertambah (+20-30 kata baru), Notebook 01 & Notebook 02 selesai dijalankan penuh.",
            ],
            [
                "TAHAP 3",
                "25 – 26 Nov 2026",
                "78% -> 86%",
                "Eksperimen 3 Model (Baseline + 2 Alternatif), Tuning 5-Fold CV & Audit Anti-Leakage",
                "Notebook 03 selesai dijalankan penuh, 6 Audit Gate PASS, & models/genai_absa_pipeline.joblib diperbarui.",
            ],
            [
                "TAHAP 4",
                "27 Nov 2026",
                "86% -> 92%",
                "Uji 3 Skenario Prototype Live (Bab 6.5) & Pengambilan Screenshot Baru (Bab 6.4)",
                "Hasil uji Normal, Ekstrem, & Kosong/Salah Format 100% Lulus + Screenshot Tab 1, Tab 2, Tab 3.",
            ],
            [
                "TAHAP 5",
                "28 – 29 Nov 2026",
                "92% -> 98%",
                "Finalisasi Laporan Word 7 Bab (.docx) & Pembuatan Slide Presentasi (.pdf)",
                "File 682024046_[Kel]_DS.docx (7 Bab rapi) & 682024046_[Kel]_DS.pdf (10-12 Slide Presentasi).",
            ],
            [
                "TAHAP 6",
                "30 Nov 2026\n(Sebelum 23.00)",
                "98% -> 100%",
                "Final QA (Restart & Run All .ipynb), Simulasi Presentasi 15 Menit & Submit FLearn",
                "Ke-4 Deliverables resmi ter-upload ke FLearn UKSW sebelum pukul 23.00 WIB.",
            ],
        ],
    )

    # Section 3: Detailed Stage Checklists
    doc.add_heading("3. Checklist Detail Langkah Kerja per Tahap", level=1)

    stages = [
        (
            "TAHAP 1: Onboarding Tim, Sinkronisasi GitHub & Setup Antigravity (23 Nov 2026)",
            [
                "[Arthur] Mendemokan aplikasi live https://genai-indonesia-am552b.streamlit.app (Tab 1, Tab 2 Batch CSV, Tab 3 Dashboard) kepada Brendan dan Tiara.",
                "[Arthur] Menjelaskan 2 konsep inti proyek selama 5 menit: (1) 4 Dimensi Aspek ABSA (A1 Akurasi, A2 Harga/Kuota, A3 Performa/UI/Server, A4 Bahasa Indonesia) dan (2) Fenomena Rating-Text Discrepancy.",
                "[Brendan & Tiara] Menerima undangan Collaborator di GitHub, membuka GitHub Desktop, dan melakukan Clone Repository awillyam327/AM552B_GenAI_Indonesia ke laptop masing-masing.",
                "[Brendan & Tiara] Membuka folder hasil clone di aplikasi Antigravity, lalu menempelkan Master Prompt Member 2 (Brendan) dan Master Prompt Member 3 (Tiara).",
            ],
        ),
        (
            "TAHAP 2: Pengayaan Kamus Slang, Validasi ABSA & Eksplorasi Data / EDA (23 – 24 Nov 2026)",
            [
                "[Brendan] Membuka 03_data_preparation/slang_dictionary_id.json dan menambahkan 20–30 kosakata slang/typo baru dari sampel ulasan nyata Play Store.",
                "[Brendan] Memverifikasi preservasi kata negasi ('tidak', 'kurang', 'jangan', 'bukan', 'belum') dan menjalankan penuh (Run All) notebooks/02_Data_Preparation_dan_ABSA.ipynb.",
                "[Tiara] Menjalankan penuh (Run All) notebooks/01_Data_Understanding_dan_EDA.ipynb dan memverifikasi statistik Bab 2 (71.221 baris unik, 0% duplikat, distribusi rating timpang).",
                "[Tiara] Merumuskan penjelasan bisnis di balik grafik Bab 2.4: dominasi ChatGPT (22.000 ulasan), anjloknya rating Character.AI (1.58 akibat filter sensor), dan 12,4% kasus Discrepancy.",
                "[Brendan & Tiara] Melakukan Commit to main dan Push origin via GitHub Desktop.",
            ],
        ),
        (
            "TAHAP 3: Eksperimen 3 Model, Hyperparameter Tuning 5-Fold & Audit Anti-Leakage (25 – 26 Nov 2026)",
            [
                "[Arthur] Melakukan Pull origin di GitHub Desktop untuk mengambil update kamus slang terbaru dari Brendan.",
                "[Arthur] Menjalankan notebooks/03_Modeling_Tuning_dan_Evaluasi.ipynb yang membandingkan 3 model wajib Bab 4: (1) Baseline Logistic Regression, (2) Alternatif 1 ComplementNB, dan (3) Alternatif 2 Calibrated LinearSVC.",
                "[Arthur] Menjalankan GridSearchCV 5-Fold Stratified Cross-Validation dan mencatat perbandingan Macro F1-Score serta ROC-AUC (OvR).",
                "[Arthur] Mengeksekusi 05_evaluation/metric_sanity_audit.py (memastikan 6 gerbang anti-leakage PASS) dan mengekspor model terbaik ke models/genai_absa_pipeline.joblib.",
                "[Brendan & Tiara] Memilih contoh nyata False Positive / False Negative dari confident_errors_fp_fn.csv untuk pembahasan Bab 5.3.",
            ],
        ),
        (
            "TAHAP 4: Uji 3 Skenario Prototype Live & Pengambilan Screenshot Bab 6 (27 Nov 2026)",
            [
                "[Arthur & Brendan] Mem-push model .joblib terbaru ke GitHub dan menjalankan python prototype/test_prototype_scenarios.py (pastikan 7/7 tes otomatis PASS).",
                "[Brendan] Menguji langsung di web https://genai-indonesia-am552b.streamlit.app untuk 3 Skenario Bab 6.5: (1) Skenario 1 Input Normal, (2) Skenario 2 Input Ekstrem / Slang & Discrepancy, dan (3) Skenario 3 Input Kosong / Salah Format.",
                "[Brendan & Tiara] Mengambil screenshot bersih dari tampilan Tab 1, Tab 2 (Batch CSV 1.000 baris), dan Tab 3 untuk dokumentasi Bab 6.4.",
            ],
        ),
        (
            "TAHAP 5: Finalisasi Laporan Word 7 Bab (.docx) & Pembuatan Slide Presentasi (.pdf) (28 – 29 Nov 2026)",
            [
                "[Tiara] Membuka Laporan_Project_Akhir_AM552_Terisi.docx, mengisi Nomor Kelompok pada Tabel 0 Halaman Judul, dan memeriksa kerapian seluruh 8 Tabel Baku (Tabel 0 s/d Tabel 7).",
                "[Tiara] Memastikan setiap grafik di Bab 2.4 memiliki tepat 1–2 kalimat interpretasi di bawahnya dan Daftar Pustaka menggunakan format APA yang konsisten.",
                "[Tiara, Brendan, Arthur] Menyusun slide presentasi PowerPoint / Canva (10–12 slide) yang merangkum Bab 1 s/d Bab 7 beserta tangkapan layar prototype.",
                "[Tiara] Mengekspor slide presentasi ke format PDF (682024046_[NomorKelompok]_DS.pdf) dan menyimpan laporan final Word (682024046_[NomorKelompok]_DS.docx).",
            ],
        ),
        (
            "TAHAP 6: Final Quality Assurance (QA), Latihan Presentasi & Submit FLearn (30 Nov 2026)",
            [
                "[Arthur] Menjalankan Kernel -> Restart & Run All pada ke-3 file Jupyter Notebook (.ipynb) agar nomor eksekusi sel berurutan rapi [1], [2], [3] tanpa error.",
                "[Brendan] Menguji akses link GitHub dan link Streamlit melalui mode Incognito browser untuk memastikan dapat diakses publik oleh dosen.",
                "[Semua Anggota] Melakukan simulasi presentasi 15 menit dan menguasai 4 jawaban inti sidang (Alasan ABSA & Discrepancy, Negation Preservation & Slang, Macro F1 & Anti-Leakage 80:20 Split, serta Konsistensi Pipeline .joblib).",
                "[Arthur] Mengunggah ke-4 Deliverables utama ke FLearn UKSW sebelum pukul 23.00 WIB dan menyimpan bukti screenshot pengumpulan.",
            ],
        ),
    ]

    for stage_title, items in stages:
        h_st = doc.add_heading(stage_title, level=2)
        for item in items:
            p_item = doc.add_paragraph(item, style="List Bullet")
            for r in p_item.runs:
                r.font.size = Pt(9.5)

    # Section 4: Presentation Slide Structure
    doc.add_heading("4. Struktur Baku Slide Presentasi (10–12 Slide untuk Diekspor ke .pdf)", level=1)
    add_formatted_table(
        doc,
        ["No Slide", "Judul & Isi Pokok Slide", "Presenter Saat Sidang"],
        [
            ["Slide 1", "Cover: Judul Proyek, Kelompok, Nama & NIM 3 Anggota (Arthur, Brendan, Tiara), Link GitHub & Streamlit", "Tiara"],
            ["Slide 2", "Bab 1 (Pemahaman Bisnis): Masalah Star Rating Play Store, Fenomena Discrepancy, Tujuan & Target Metrik", "Tiara"],
            ["Slide 3", "Bab 2 (Dataset): Profil 71.221 Ulasan Unik 8 Aplikasi GenAI Indonesia & Ketimpangan Kelas Rating", "Tiara"],
            ["Slide 4", "Bab 2 (Temuan EDA): Grafik Distribusi Aplikasi, Anomali Character.AI (1.58), & 12,4% Kasus Discrepancy", "Tiara"],
            ["Slide 5", "Bab 3 (Persiapan Data): Kamus Slang Indonesia, Preservasi Negasi, 4 Aspek ABSA, & 80:20 Stratified Split", "Brendan"],
            ["Slide 6", "Bab 4 (Pemodelan & Tuning): Perbandingan Baseline Logistic Regression vs ComplementNB vs LinearSVC (5-Fold CV)", "Arthur"],
            ["Slide 7", "Bab 5 (Evaluasi Metrik): Tabel Hasil Macro F1 (0.6512) & ROC-AUC (0.8751) vs Target Bab 1.4 (Tercapai)", "Arthur"],
            ["Slide 8", "Bab 5 (Analisis Kesalahan & XAI): Confusion Matrix, Kata Kunci SHAP Terpenting, & Contoh Kasus Discrepancy", "Arthur"],
            ["Slide 9", "Bab 6 (Arsitektur & Fitur Prototype): Alur Sistem Streamlit + Screenshot 3 Tab Fitur Utama", "Brendan"],
            ["Slide 10", "Bab 6.5 (Hasil Uji 3 Skenario): Bukti Kelulusan Input Normal, Input Ekstrem, & Input Kosong/Salah Format", "Brendan"],
            ["Slide 11", "Bab 7 (Kesimpulan & Saran): Jawaban 4 Rumusan Masalah, Rekomendasi Industri, & Rencana Pengembangan", "Arthur"],
        ],
    )

    doc.save(OUT_DOCX)
    print(f"[SUCCESS] Dokumen Word Roadmap 100% berhasil dibuat: {OUT_DOCX}")


if __name__ == "__main__":
    generate_roadmap_docx()
