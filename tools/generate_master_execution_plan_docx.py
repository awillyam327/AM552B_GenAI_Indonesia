import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCX_OUT = os.path.join(ROOT_DIR, "MASTER_EXECUTION_PLAN.docx")


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
        run.font.size = Pt(8.5)
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
            run.font.size = Pt(8.5)
            if c_idx == 0:
                run.bold = True
            set_cell_background(row_cells[c_idx], bg_color)

    doc.add_paragraph()
    return table


def main():
    doc = Document()
    for section in doc.sections:
        section.top_margin = Inches(0.7)
        section.bottom_margin = Inches(0.7)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = title_p.add_run(
        "MASTER EXECUTION PLAN & JADWAL SPRINT 9 HARI (21 NOV – 30 NOV 2026)\n"
        "MATA KULIAH AM552 - DATA SCIENCE OF IS (S1 SISTEM INFORMASI, FTI UKSW)\n"
    )
    r_sub.bold = True
    r_sub.font.size = Pt(11)
    r_sub.font.color.rgb = RGBColor(31, 78, 121)

    r_main = title_p.add_run(
        "\n\"Analisis Komparatif Sentimen Berbasis Aspek (ABSA) dan Pemodelan Topik Pengalaman Pengguna "
        "terhadap Ekosistem Aplikasi Mobile Generative AI di Indonesia\""
    )
    r_main.bold = True
    r_main.font.size = Pt(13)
    r_main.font.color.rgb = RGBColor(15, 36, 62)

    meta_p = doc.add_paragraph()
    meta_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_meta = meta_p.add_run(
        "Deadline FLearn: 30 November 2026 Pukul 23.00 WIB | Format Penamaan: 682024046_KEL_DS | Dataset: 71.221 Baris Unik"
    )
    r_meta.italic = True
    r_meta.font.size = Pt(9)
    r_meta.font.color.rgb = RGBColor(89, 89, 89)

    # 1. KEPATUHAN DELIVERABLES AM552
    h1 = doc.add_heading("1. Matriks Kepatuhan Ketentuan Resmi AM552 (FTI UKSW)", level=1)
    h1.runs[0].font.color.rgb = RGBColor(31, 78, 121)

    comp_headers = ["Deliverable Wajib", "Ketentuan Resmi Dosen AM552", "Artefak di Workspace Proyek Kita", "Status"]
    comp_rows = [
        [
            "1. File Kode (.ipynb)",
            "Jupyter Notebook (.ipynb) yang rapi, terurut, dan siap dijalankan ulang (Restart & Run All).",
            "• notebooks/01_Data_Understanding_dan_EDA.ipynb\n• notebooks/02_Data_Preparation_dan_ABSA.ipynb\n• notebooks/03_Modeling_Tuning_dan_Evaluasi.ipynb",
            "SIAP",
        ],
        [
            "2. File Laporan (.docx)",
            "Dokumen MS Word (.docx) mengikuti struktur baku 7 Bab FTI UKSW + Daftar Pustaka APA.",
            "• 06_deployment_reporting/682024046_KEL_DS_Laporan.docx",
            "SIAP",
        ],
        [
            "3. File Presentasi (.pdf)",
            "Slide PowerPoint yang diekspor ke .pdf dengan format nama NIM_KEL_DS.",
            "• 06_deployment_reporting/682024046_KEL_DS_Presentasi.pdf",
            "TERJADWAL",
        ],
        [
            "4. Live Prototype Web App",
            "Web app interaktif di Streamlit Community Cloud dengan model .joblib & preprocessing identik.",
            "• prototype/app.py\n• models/genai_absa_pipeline.joblib\n• prototype/test_prototype_scenarios.py",
            "SIAP DEPLOY",
        ],
    ]
    add_styled_table(doc, comp_headers, comp_rows)

    # 2. JADWAL SPRINT 9 HARI
    h2 = doc.add_heading("2. Jadwal Sprint Harian 9 Hari Menuju Deadline (21 Nov – 30 Nov 2026)", level=1)
    h2.runs[0].font.color.rgb = RGBColor(31, 78, 121)

    sprint_headers = [
        "Tanggal Sprint",
        "Target Bab & Fase",
        "Member 1 (Lead, Modeling, Audit & Backend)",
        "Member 2 (Data Eng, ABSA & UI Prototype)",
        "Member 3 (EDA, Topic Modeling & Laporan 7 Bab)",
    ]
    sprint_rows = [
        [
            "Hari 1-2\n(21-22 Nov)",
            "Bab 1 & Bab 2\n(Data Understanding)",
            "Finalisasi Bab 1 (Tujuan Bisnis vs Data Science, KPI Macro-F1 >= 0.63, ROC-AUC >= 0.85).",
            "Audit kondisi data 71.221 baris & kamus slang Bahasa Indonesia.",
            "Eksekusi 01_Data_Understanding_dan_EDA.ipynb & grafik Bab 2.4 + narasi 2 kalimat.",
        ],
        [
            "Hari 3-4\n(23-24 Nov)",
            "Bab 3\n(Data Preparation & ABSA)",
            "Verifikasi Stratified Split 80:20 (random_state=42) sebelum TfidfVectorizer.fit().",
            "Eksekusi 02_Data_Preparation_dan_ABSA.ipynb, pelabelan 4 aspek ABSA & Discrepancy.",
            "Menyusun Tabel 3.1 Langkah Persiapan, Diagram 3.2, dan Tabel 3.3 Data Akhir.",
        ],
        [
            "Hari 5-6\n(25-26 Nov)",
            "Bab 4 & Bab 5\n(Modeling & Evaluation)",
            "Eksekusi 03_Modeling_Tuning_dan_Evaluasi.ipynb (Baseline + 2 Model Alternatif + Tuning 5-Fold), Audit Forensik, & simpan .joblib.",
            "Analisis kesalahan prediksi (False Positive / False Negative) & matriks ABSA 8 aplikasi.",
            "Menyusun Bab 4 (Tabel Model, Tuning, Asumsi) & Bab 5 (Tabel Evaluasi, SHAP, Keterbatasan).",
        ],
        [
            "Hari 7-8\n(27-28 Nov)",
            "Bab 6 & Bab 7\n(Prototype & Penutup)",
            "Deploy prototype/app.py & models/genai_absa_pipeline.joblib ke Streamlit Community Cloud.",
            "Finalisasi UI 3 Tab Streamlit & eksekusi test_prototype_scenarios.py (3 Skenario Bab 6.5).",
            "Menyusun Bab 6 (Alur, Fitur, Tabel Uji 3 Skenario) & Bab 7 (Kesimpulan menjawab Bab 1.2).",
        ],
        [
            "Hari 9-10\n(29-30 Nov)",
            "Final QA & Submit FLearn (< 23.00 WIB)",
            "Uji Restart & Run All pada 3 file .ipynb & cek link live Streamlit.",
            "Verifikasi file CSV demo & screenshot aplikasi pada Bab 6.4-6.5.",
            "Finalisasi laporan 682024046_KEL_DS_Laporan.docx, slide PDF, & submit ke FLearn.",
        ],
    ]
    add_styled_table(doc, sprint_headers, sprint_rows)

    # 3. PEMBAGIAN PERAN 3 ANGGOTA
    h3 = doc.add_heading("3. Matriks Pembagian Peran 3 Anggota Tim", level=1)
    h3.runs[0].font.color.rgb = RGBColor(31, 78, 121)

    role_headers = ["Anggota", "Peran Utama", "Tanggung Jawab Kode (.ipynb & Prototype)", "Tanggung Jawab Laporan 7 Bab"]
    role_rows = [
        [
            "Member 1\n(682024046)",
            "Project Lead, Modeling, Metric Audit & Backend",
            "• 03_Modeling_Tuning_dan_Evaluasi.ipynb\n• 05_evaluation/metric_sanity_audit.py\n• models/genai_absa_pipeline.joblib\n• Backend & Deployment Streamlit Cloud",
            "• Bab 1 (Pendahuluan & Ukuran Keberhasilan)\n• Bab 4 (Pilihan Model, Tuning, Asumsi)\n• Bab 5 (Evaluasi Metrik, SHAP, Target)",
        ],
        [
            "Member 2\n(Anggota 2)",
            "Data Engineering, Preprocessing, ABSA & UI Testing",
            "• 02_Data_Preparation_dan_ABSA.ipynb\n• 03_data_preparation/slang_dictionary_id.json\n• UI prototype/app.py & test_prototype_scenarios.py",
            "• Bab 2.1–2.3 (Sumber, Kamus, Kondisi Data)\n• Bab 3 (Persiapan, Split 80:20, Data Akhir)\n• Bab 6.3–6.5 (Fitur & Tabel Uji 3 Skenario)",
        ],
        [
            "Member 3\n(Anggota 3)",
            "EDA, Topic Discovery, Visualisasi & Laporan/Slide",
            "• 01_Data_Understanding_dan_EDA.ipynb\n• Visualisasi Grafik Bab 2 & Tab 3 Dashboard\n• Kompilasi Slide Presentasi PDF",
            "• Abstrak & Bab 2.4–2.5 (Grafik + Narasi 2 Kalimat)\n• Bab 6.1–6.2 & Bab 7 (Kesimpulan & Saran)\n• Daftar Pustaka (Format APA)",
        ],
    ]
    add_styled_table(doc, role_headers, role_rows)

    doc.save(DOCX_OUT)
    print(f"[OK] MASTER_EXECUTION_PLAN.docx berhasil diperbarui: {DOCX_OUT}")


if __name__ == "__main__":
    main()
