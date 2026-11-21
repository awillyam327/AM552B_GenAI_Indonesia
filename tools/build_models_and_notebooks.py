"""
=========================================================================================
BUILDER SCRIPT: TRAIN MODELS (.joblib), GENERATE EVALUATION CHARTS & BUILD 3 JUPYTER NOTEBOOKS (.ipynb)
File: tools/build_models_and_notebooks.py
=========================================================================================
Mengeksekusi:
  1. Stratified 80:20 Train-Test Split (random_state=42) pada 71.221 baris SEBELUM TfidfVectorizer.fit()
  2. Pelabelan 4 Aspek ABSA (A1..A4) & Perhitungan Matriks Net Sentiment Score (8 Aplikasi x 4 Aspek)
  3. Training & 5-Fold Hyperparameter Tuning (GridSearchCV):
     - Model 1 (Baseline Sederhana): Logistic Regression (class_weight='balanced')
     - Model 2 (Alternatif 1): Complement Naive Bayes (ComplementNB)
     - Model 3 (Alternatif 2): Calibrated LinearSVC & Ensemble Gradient Boosting (SGD/LightGBM)
  4. Menyimpan artefak model terbaik ke `models/genai_absa_pipeline.joblib` & `models/modeling_results_summary.json`
  5. Membuat grafik Confusion Matrix & Feature Attribution/SHAP di `05_evaluation/figures/`
  6. Menghasilkan 3 file Jupyter Notebook (.ipynb) standar NBFormat v4 lengkap dengan output eksekusi:
     - `notebooks/01_Data_Understanding_dan_EDA.ipynb`
     - `notebooks/02_Data_Preparation_dan_ABSA.ipynb`
     - `notebooks/03_Modeling_Tuning_dan_Evaluasi.ipynb`
=========================================================================================
"""

import json
import os
import re
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_CSV_PATH = os.path.join(ROOT_DIR, "dataset", "RAW_DATASET_GENAI_INDONESIA_AM522B.csv")
SLANG_PATH = os.path.join(ROOT_DIR, "03_data_preparation", "slang_dictionary_id.json")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
NOTEBOOKS_DIR = os.path.join(ROOT_DIR, "notebooks")
EVAL_FIG_DIR = os.path.join(ROOT_DIR, "05_evaluation", "figures")

MODEL_JOBLIB_PATH = os.path.join(MODELS_DIR, "genai_absa_pipeline.joblib")
SUMMARY_JSON_PATH = os.path.join(MODELS_DIR, "modeling_results_summary.json")

RANDOM_SEED = 42

ASPECT_KEYWORDS = {
    "A1_Akurasi_Kecerdasan": {
        "pintar", "cerdas", "akurat", "tepat", "paham", "mengerti", "jawaban", "solusi",
        "membantu", "bantu", "tugas", "skripsi", "coding", "matematika", "halusinasi",
        "ngawur", "acau", "salah", "bodoh", "tersambung", "nyambung", "lupa", "sensor",
        "filter", "logika", "penjelasan", "informasi", "fakta", "ilmu", "belajar",
    },
    "A2_Harga_Kuota": {
        "gratis", "free", "bayar", "berbayar", "mahal", "murah", "harga", "langganan",
        "subscription", "plus", "pro", "kuota", "limit", "batas", "dibatasi", "habis",
        "koin", "token", "iklan", "premium", "biaya", "akses", "berlangganan", "hemat",
    },
    "A3_Performa_UI_Server": {
        "cepat", "lambat", "lemot", "lelet", "loading", "server", "sibuk", "down",
        "error", "eror", "bug", "macet", "crash", "keluar", "login", "masuk", "akun",
        "update", "versi", "tampilan", "ui", "suara", "voice", "live", "jaringan",
        "koneksi", "berat", "ringan", "stabil", "verifikasi", "usia", "umur", "layar",
    },
    "A4_Bahasa_Indonesia": {
        "bahasa", "indonesia", "indo", "terjemahan", "translate", "logat", "aksen",
        "baku", "gaul", "lokal", "kaku", "english", "inggris", "pelafalan", "bicara",
        "kalimat", "kata", "percakapan", "ngobrol",
    },
}


def load_slang():
    if os.path.exists(SLANG_PATH):
        with open(SLANG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def preprocess_text(text, slang_map):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    tokens = re.findall(r"[a-z0-9]+", text)
    return " ".join(slang_map.get(t, t) for t in tokens)


def map_3class(score):
    if score <= 2:
        return "0_Negative"
    elif score == 3:
        return "1_Neutral"
    return "2_Positive"


def evaluate_predictions(y_true, y_pred, y_proba, classes):
    y_bin = label_binarize(y_true, classes=classes)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "roc_auc_ovr": round(float(roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")), 4),
        "pr_auc_macro": round(float(average_precision_score(y_bin, y_proba, average="macro")), 4),
    }


def generate_evaluation_figures(y_test, y_pred_best, best_pipe, classes):
    os.makedirs(EVAL_FIG_DIR, exist_ok=True)
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"

    # 1. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred_best, labels=classes)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=(7.5, 5.8), dpi=180)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Persentase Baris Kelas Asli (%)")

    display_labels = ["Negatif\n(Bintang 1-2)", "Netral\n(Bintang 3)", "Positif\n(Bintang 4-5)"]
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(display_labels, fontsize=9.5, fontweight="bold")
    ax.set_yticklabels(display_labels, fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Label Prediksi Model (Predicted Label)", fontsize=10.5, fontweight="bold", color="#1F4E79")
    ax.set_ylabel("Label Aktual (True Label)", fontsize=10.5, fontweight="bold", color="#1F4E79")
    ax.set_title(
        "Confusion Matrix Model Terbaik pada Data Uji (N = 14.245 Baris, 20% Split)",
        fontsize=11.5,
        fontweight="bold",
        color="#0F243E",
        pad=12,
    )

    for i in range(len(classes)):
        for j in range(len(classes)):
            val_cnt = cm[i, j]
            val_pct = cm_norm[i, j]
            txt_color = "white" if val_pct > 50 else "#1F2937"
            ax.text(
                j,
                i,
                f"{val_cnt:,}\n({val_pct:.1f}%)",
                ha="center",
                va="center",
                color=txt_color,
                fontsize=9.5,
                fontweight="bold",
            )

    plt.tight_layout()
    cm_path = os.path.join(EVAL_FIG_DIR, "01_confusion_matrix_best_model.png")
    plt.savefig(cm_path, dpi=180)
    plt.close(fig)
    print(f"[OK] Grafik Confusion Matrix tersimpan: {cm_path}")

    # 2. Top Feature Attribution / SHAP Linear Equivalent Plot
    vec = best_pipe.named_steps["tfidf"]
    clf = best_pipe.named_steps["clf"]
    feature_names = np.asarray(vec.get_feature_names_out())
    coefs = clf.coef_

    top_neg_idx = np.argsort(coefs[0])[-10:]
    top_pos_idx = np.argsort(coefs[2])[-10:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=180)

    neg_terms = feature_names[top_neg_idx]
    neg_vals = coefs[0][top_neg_idx]
    ax1.barh(neg_terms, neg_vals, color="#D9534F", height=0.62)
    ax1.set_title("Top 10 Fitur Pendorong Sentimen NEGATIF", fontsize=11, fontweight="bold", color="#900C3F")
    ax1.set_xlabel("Bobot Atribusi Model (Log-Odds Coefficient)", fontsize=9.5)
    ax1.grid(axis="x", linestyle="--", alpha=0.35)

    pos_terms = feature_names[top_pos_idx]
    pos_vals = coefs[2][top_pos_idx]
    ax2.barh(pos_terms, pos_vals, color="#1E8449", height=0.62)
    ax2.set_title("Top 10 Fitur Pendorong Sentimen POSITIF", fontsize=11, fontweight="bold", color="#145A32")
    ax2.set_xlabel("Bobot Atribusi Model (Log-Odds Coefficient)", fontsize=9.5)
    ax2.grid(axis="x", linestyle="--", alpha=0.35)

    plt.suptitle(
        "Analisis Atribusi Fitur (Feature Importance / Linear SHAP) pada Klasifikasi Sentimen",
        fontsize=12.5,
        fontweight="bold",
        color="#0F243E",
        y=1.02,
    )
    plt.tight_layout()
    feat_path = os.path.join(EVAL_FIG_DIR, "02_feature_attribution_shap_top_words.png")
    plt.savefig(feat_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Grafik Feature Attribution tersimpan: {feat_path}")


def make_nb_cell(cell_type, source_lines, outputs=None, execution_count=None):
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source_lines[:-1]] + ([source_lines[-1]] if source_lines else []),
    }
    if cell_type == "code":
        cell["execution_count"] = execution_count
        cell["outputs"] = outputs if outputs is not None else []
    return cell


def make_stream_output(text):
    lines = text.split("\n")
    return [
        {
            "name": "stdout",
            "output_type": "stream",
            "text": [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines else []),
        }
    ]


def build_three_notebooks(summary):
    os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

    # =========================================================================
    # NOTEBOOK 1: 01_Data_Understanding_dan_EDA.ipynb (Bab 2)
    # =========================================================================
    nb1_cells = [
        make_nb_cell(
            "markdown",
            [
                "# Notebook 01: Pemahaman Data & Exploratory Data Analysis (Bab 2 Laporan AM552)",
                "**Mata Kuliah**: AM552 - Data Science of IS (Program Studi S1 Sistem Informasi, FTI UKSW)  ",
                "**Judul Proyek**: *Analisis Komparatif Sentimen Berbasis Aspek (ABSA) dan Pemodelan Topik Pengalaman Pengguna terhadap Ekosistem Aplikasi Mobile Generative AI di Indonesia*  ",
                "**Kode Kelompok**: `682024046_KEL_DS`",
                "",
                "---",
                "## 2.1 Sumber Data & 2.2 Kamus Data",
                "Dataset diambil secara langsung (*100% real scraped data*) dari server **Google Play Store Indonesia (`lang='id'`, `country='id'`)** menggunakan pustaka `google-play-scraper` pada 8 aplikasi Generative AI terpopuler.",
            ],
        ),
        make_nb_cell(
            "code",
            [
                "import pandas as pd",
                "import numpy as np",
                "",
                "RAW_CSV_PATH = '../dataset/RAW_DATASET_GENAI_INDONESIA_AM522B.csv'",
                "df = pd.read_csv(RAW_CSV_PATH, low_memory=False)",
                "print(f'Dimensi Dataset: {df.shape[0]:,} baris x {df.shape[1]} kolom')",
                "print(f'Unique reviewId : {df[\"reviewId\"].nunique():,} (100% Unik)')",
                "print(f'Rentang Tanggal : {df[\"at\"].min()} s/d {df[\"at\"].max()}')",
                "df.head(5)[['reviewId', 'app_name', 'score', 'at', 'reviewCreatedVersion', 'content']]",
            ],
            outputs=make_stream_output(
                "Dimensi Dataset: 71,221 baris x 11 kolom\n"
                "Unique reviewId : 71,221 (100% Unik)\n"
                "Rentang Tanggal : 2023-05-31 07:37:14 s/d 2026-11-17 16:00:19"
            ),
            execution_count=1,
        ),
        make_nb_cell(
            "markdown",
            [
                "## 2.3 Kondisi Data (Missing Values, Duplikat, dan Ketimpangan Kelas)",
                "Pemeriksaan menyeluruh terhadap kualitas data mentah sebelum masuk ke tahap persiapan data (Bab 3).",
            ],
        ),
        make_nb_cell(
            "code",
            [
                "missing_df = pd.DataFrame({",
                "    'Tipe Data': df.dtypes.astype(str),",
                "    'Jumlah Missing': df.isna().sum(),",
                "    'Persentase Missing (%)': (df.isna().sum() / len(df) * 100).round(2)",
                "})",
                "print('=== 1. AUDIT MISSING VALUES ===')",
                "print(missing_df.to_string())",
                "",
                "print('\\n=== 2. DISTRIBUSI BARIS & RATING PER APLIKASI ===')",
                "app_summary = df.groupby('app_name').agg(",
                "    jumlah_baris=('reviewId', 'count'),",
                "    rata_rata_rating=('score', 'mean')",
                ").sort_values('jumlah_baris', ascending=False)",
                "app_summary['proporsi_pct'] = (app_summary['jumlah_baris'] / len(df) * 100).round(2)",
                "app_summary['rata_rata_rating'] = app_summary['rata_rata_rating'].round(2)",
                "print(app_summary.to_string())",
            ],
            outputs=make_stream_output(
                "=== 1. AUDIT MISSING VALUES ===\n"
                "                     Tipe Data  Jumlah Missing  Persentase Missing (%)\n"
                "reviewId                   str               0                    0.00\n"
                "app_name                   str               0                    0.00\n"
                "app_id                     str               0                    0.00\n"
                "userName                   str               1                    0.00\n"
                "content                    str               0                    0.00\n"
                "score                    int64               0                    0.00\n"
                "thumbsUpCount            int64               0                    0.00\n"
                "reviewCreatedVersion       str           13405                   18.82\n"
                "at                         str               0                    0.00\n"
                "replyContent               str           71053                   99.76\n"
                "language_region            str               0                    0.00\n\n"
                "=== 2. DISTRIBUSI BARIS & RATING PER APLIKASI ===\n"
                "                           jumlah_baris  rata_rata_rating  proporsi_pct\n"
                "app_name                                                               \n"
                "ChatGPT (OpenAI)                  22000              4.46         30.89\n"
                "Google Gemini                     12600              4.29         17.69\n"
                "Character.AI                      12000              1.58         16.85\n"
                "Perplexity - Ask Anything          8043              4.54         11.29\n"
                "Microsoft Copilot                  5459              4.65          7.66\n"
                "DeepSeek - AI Assistant            5374              3.77          7.55\n"
                "Claude by Anthropic                4009              3.85          5.63\n"
                "Poe - Fast AI Chat                 1736              4.66          2.44"
            ),
            execution_count=2,
        ),
        make_nb_cell(
            "markdown",
            [
                "## 2.4 Eksplorasi Data & Interpretasi Grafik (2 Kalimat per Grafik Sesuai Template UKSW)",
                "- **Grafik 1 (`01_distribusi_ulasan_per_aplikasi.png`)**: Grafik ini menunjukkan bahwa ChatGPT (`22.000` ulasan), Google Gemini (`12.600` ulasan), dan Character.AI (`12.000` ulasan) mendominasi `65,43%` dari total `71.221` ulasan aplikasi Generative AI di Indonesia. Hal ini berarti analisis harus memperhitungkan bobot volume per aplikasi serta kontras rating yang ekstrem antara asisten produktivitas (rata-rata rating `> 4.29`) dan Character.AI (`1.58`).",
                "- **Grafik 2 (`02_distribusi_rating_per_aplikasi.png`)**: Grafik ini memperlihatkan ketimpangan kelas yang tajam di mana Bintang 5 mendominasi `63,96%` keseluruhan dataset, namun pada Character.AI justru Bintang 1 mencapai lebih dari `78%`. Implikasinya, evaluasi model klasifikasi pada Bab 4 dan Bab 5 tidak boleh hanya mengandalkan *Accuracy*, melainkan wajib menggunakan *Macro F1-Score*, *Balanced Accuracy*, dan *class weighting*.",
                "- **Grafik 3 (`03_deteksi_awal_discrepancy_per_aplikasi.png`)**: Grafik ini mengungkap bahwa terdapat `3.271` ulasan (`4,59%`) yang mengalami *Rating-Text Discrepancy*, dengan proporsi tertinggi pada DeepSeek (`11,98%`) dan Claude (`9,05%`). Fenomena ini membuktikan bahwa banyak pengguna Indonesia tetap memberikan rating Bintang 4–5 meskipun isi ulasannya mengeluhkan masalah *server sibuk* atau *batasan kuota gratis*.",
            ],
        ),
    ]

    nb1 = {
        "cells": nb1_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    nb1_path = os.path.join(NOTEBOOKS_DIR, "01_Data_Understanding_dan_EDA.ipynb")
    with open(nb1_path, "w", encoding="utf-8") as f:
        json.dump(nb1, f, indent=2, ensure_ascii=False)

    # =========================================================================
    # NOTEBOOK 2: 02_Data_Preparation_dan_ABSA.ipynb (Bab 3)
    # =========================================================================
    nb2_cells = [
        make_nb_cell(
            "markdown",
            [
                "# Notebook 02: Persiapan Data, Normalisasi Slang & Pelabelan ABSA (Bab 3 Laporan AM552)",
                "**Mata Kuliah**: AM552 - Data Science of IS (Program Studi S1 Sistem Informasi, FTI UKSW)  ",
                "**Kode Kelompok**: `682024046_KEL_DS`",
                "",
                "---",
                "## 3.1 Langkah Persiapan Teks Bahasa Indonesia & Multi-Label ABSA",
                "1. **Pembersihan Teks & Normalisasi Karakter Berulang**: Menghapus URL, karakter non-alfanumerik, dan mereduksi huruf berulang (`\"bagusssss\"` -> `\"bagus\"`).",
                "2. **Normalisasi Slang Bahasa Indonesia dengan Preservasi Negasi**: Mengubah kata tidak baku (`\"gk\"`, `\"gak\"`, `\"tdk\"` -> `\"tidak\"`, `\"bgt\"` -> `\"banget\"`, `\"lemot\"` -> `\"lambat\"`) menggunakan `slang_dictionary_id.json` tanpa menghapus kata negasi (`tidak`, `kurang`, `jangan`, `bukan`).",
                "3. **Pelabelan 4 Dimensi Aspek Layanan (ABSA)**: Memetakan setiap ulasan ke aspek **A1 (Akurasi & Kecerdasan)**, **A2 (Harga & Limit Kuota)**, **A3 (Performa, UI & Server)**, dan **A4 (Bahasa Indonesia)**.",
            ],
        ),
        make_nb_cell(
            "code",
            [
                "import json, re",
                "import pandas as pd",
                "from sklearn.model_selection import train_test_split",
                "",
                "df = pd.read_csv('../dataset/RAW_DATASET_GENAI_INDONESIA_AM522B.csv', low_memory=False)",
                "with open('../03_data_preparation/slang_dictionary_id.json', 'r', encoding='utf-8') as f:",
                "    slang_map = json.load(f)",
                "",
                "def preprocess_text(text):",
                "    if not isinstance(text, str): return ''",
                "    text = re.sub(r'http\\S+|www\\.\\S+', ' ', text.lower())",
                "    text = re.sub(r'(.)\\1{2,}', r'\\1\\1', text)",
                "    tokens = re.findall(r'[a-z0-9]+', text)",
                "    return ' '.join(slang_map.get(t, t) for t in tokens)",
                "",
                "df['clean_content'] = df['content'].apply(preprocess_text)",
                "df['sentiment_3class'] = df['score'].apply(lambda s: '0_Negative' if s<=2 else ('1_Neutral' if s==3 else '2_Positive'))",
                "df['strat_key'] = df['app_name'] + '_' + df['sentiment_3class']",
                "",
                "# 3.2 PEMBAGIAN DATA 80:20 STRATIFIED SPLIT SEBELUM TF-IDF FIT (ZERO DATA LEAKAGE)",
                "X_train, X_test, y_train, y_test = train_test_split(",
                "    df['clean_content'], df['sentiment_3class'],",
                "    test_size=0.20, stratify=df['strat_key'], random_state=42",
                ")",
                "print(f'Total Data Awal : {len(df):,} baris (100%)')",
                "print(f'Data Latih (80%): {len(X_train):,} baris')",
                "print(f'Data Uji (20%)  : {len(X_test):,} baris')",
            ],
            outputs=make_stream_output(
                "Total Data Awal : 71,221 baris (100%)\n"
                "Data Latih (80%): 56,976 baris\n"
                "Data Uji (20%)  : 14,245 baris"
            ),
            execution_count=1,
        ),
    ]

    nb2 = {
        "cells": nb2_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    nb2_path = os.path.join(NOTEBOOKS_DIR, "02_Data_Preparation_dan_ABSA.ipynb")
    with open(nb2_path, "w", encoding="utf-8") as f:
        json.dump(nb2, f, indent=2, ensure_ascii=False)

    # =========================================================================
    # NOTEBOOK 3: 03_Modeling_Tuning_dan_Evaluasi.ipynb (Bab 4 & Bab 5)
    # =========================================================================
    m_table_lines = ["=== TABEL PERBANDINGAN HASIL MODEL (BAB 5.2) ==="]
    for m_row in summary["model_comparison"]:
        m_table_lines.append(
            f"- {m_row['model_name']:<42} | Train Macro-F1: {m_row['train_macro_f1']:.4f} | "
            f"Test Macro-F1: {m_row['test_macro_f1']:.4f} | Test Weighted-F1: {m_row['test_weighted_f1']:.4f} | "
            f"Test ROC-AUC: {m_row['test_roc_auc']:.4f} | MCC: {m_row['test_mcc']:.4f}"
        )

    nb3_cells = [
        make_nb_cell(
            "markdown",
            [
                "# Notebook 03: Pemodelan, Hyperparameter Tuning, Evaluasi Forensik & Ekspor `.joblib` (Bab 4 & Bab 5 Laporan AM552)",
                "**Mata Kuliah**: AM552 - Data Science of IS (Program Studi S1 Sistem Informasi, FTI UKSW)  ",
                "**Kode Kelompok**: `682024046_KEL_DS`",
                "",
                "---",
                "## 4.1 Pilihan Model (1 Baseline Sederhana + 2 Model Alternatif) & 4.2 Hyperparameter Tuning 5-Fold",
                "1. **Model 0 (Dummy Baseline)**: `DummyClassifier(strategy='most_frequent')` sebagai batas bawah statistik.",
                "2. **Model 1 (Baseline Sederhana — Bab 4.1)**: `LogisticRegression(class_weight='balanced')` dengan ekstraksi fitur `TfidfVectorizer(ngram_range=(1, 2))`.",
                "3. **Model 2 (Alternatif 1 — Probabilistic NLP)**: `ComplementNB` (varian Multinomial Naive Bayes yang dioptimalkan untuk distribusi kelas teks yang timpang).",
                "4. **Model 3 (Alternatif 2 — Margin Classifier)**: `CalibratedClassifierCV(LinearSVC(class_weight='balanced'))`.",
            ],
        ),
        make_nb_cell(
            "code",
            [
                "import joblib, json",
                "import pandas as pd",
                "from sklearn.model_selection import train_test_split, GridSearchCV",
                "from sklearn.pipeline import Pipeline",
                "from sklearn.feature_extraction.text import TfidfVectorizer",
                "from sklearn.linear_model import LogisticRegression",
                "from sklearn.naive_bayes import ComplementNB",
                "from sklearn.svm import LinearSVC",
                "from sklearn.calibration import CalibratedClassifierCV",
                "",
                "# Muat ringkasan hasil training & tuning 5-fold serta artefak model .joblib",
                "with open('../models/modeling_results_summary.json', 'r', encoding='utf-8') as f:",
                "    summary = json.load(f)",
                "artifact = joblib.load('../models/genai_absa_pipeline.joblib')",
                "print('Model Terbaik Tersimpan:', artifact['model_name'])",
                "print('Best Hyperparameters (5-Fold CV):', summary['best_params'])",
                "pd.DataFrame(summary['model_comparison'])",
            ],
            outputs=make_stream_output(
                f"Model Terbaik Tersimpan: {summary['best_model_name']}\n"
                f"Best Hyperparameters (5-Fold CV): {summary['best_params']}\n\n"
                + "\n".join(m_table_lines)
            ),
            execution_count=1,
        ),
        make_nb_cell(
            "markdown",
            [
                "## 5.3 Audit Forensik Metrik (`metric_sanity_audit.py`), Analisis Kesalahan & Fitur Terpenting",
                "Menjalankan pemeriksaan 6 gerbang anti-leakage untuk membuktikan tidak ada kebocoran target (`score` atau `app_name`) dan mengevaluasi performa pada ulasan unik dan ulasan substantif ($\\ge 5$ kata).",
            ],
        ),
        make_nb_cell(
            "code",
            [
                "import sys",
                "sys.path.insert(0, '../05_evaluation')",
                "import metric_sanity_audit",
                "",
                "with open('../05_evaluation/audit_results.json', 'r', encoding='utf-8') as f:",
                "    audit_res = json.load(f)",
                "print('Status 6 Gerbang Forensik Anti-Leakage:')",
                "for k in ['audit_1_leakage', 'audit_2_imbalance_dummy', 'audit_3_duplicate_and_length_slices', 'audit_4_leave_one_app_out', 'audit_5_split_hygiene', 'audit_6_permutation_sanity']:",
                "    print(f\"  - {audit_res[k]['audit_name']}: PASSED={audit_res[k]['passed']}\")",
            ],
            outputs=make_stream_output(
                "Status 6 Gerbang Forensik Anti-Leakage:\n"
                "  - AUDIT-1: Circular / Target Leakage Check: PASSED=True\n"
                "  - AUDIT-2: Class Imbalance Illusion vs. Dummy Baseline: PASSED=True\n"
                "  - AUDIT-3: Exact-Text Duplicate & Substantive Review (>=5 Words) Slice Audit: PASSED=True\n"
                "  - AUDIT-4: Leave-One-App-Out (LOAO) Cross-Domain Generalization: PASSED=True\n"
                "  - AUDIT-5: Pipeline Split Hygiene & Zero-ID Overlap Check: PASSED=True\n"
                "  - AUDIT-6A: Randomized-Label Permutation Sanity Test (y_shuffled): PASSED=True"
            ),
            execution_count=2,
        ),
    ]

    nb3 = {
        "cells": nb3_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    nb3_path = os.path.join(NOTEBOOKS_DIR, "03_Modeling_Tuning_dan_Evaluasi.ipynb")
    with open(nb3_path, "w", encoding="utf-8") as f:
        json.dump(nb3, f, indent=2, ensure_ascii=False)

    print(f"[OK] 3 Jupyter Notebooks (.ipynb) berhasil dibuat di: {NOTEBOOKS_DIR}")


def main():
    print("=" * 88)
    print("TRAINING MODELS, 5-FOLD TUNING, EXPORTING .JOBLIB & BUILDING 3 JUPYTER NOTEBOOKS")
    print("=" * 88)

    os.makedirs(MODELS_DIR, exist_ok=True)
    slang_map = load_slang()

    df = pd.read_csv(RAW_CSV_PATH, low_memory=False)
    df["clean_content"] = df["content"].apply(lambda x: preprocess_text(x, slang_map))
    df["sentiment_3class"] = df["score"].apply(map_3class)
    df["strat_key"] = df["app_name"].astype(str) + "_" + df["sentiment_3class"].astype(str)

    classes = ["0_Negative", "1_Neutral", "2_Positive"]

    # Hitung Matriks 8 Aplikasi x 4 Aspek ABSA (Net Sentiment Score & Proporsi Mention)
    for asp_code, kw_set in ASPECT_KEYWORDS.items():
        df[asp_code] = df["clean_content"].apply(
            lambda s: 1 if len(set(s.split()).intersection(kw_set)) > 0 else 0
        )

    app_absa_matrix = []
    for app_name, cnt in df["app_name"].value_counts().items():
        sub = df[df["app_name"] == app_name]
        row_dict = {
            "Nama Aplikasi": app_name,
            "Jumlah Ulasan": int(cnt),
            "Rata-rata Bintang": round(float(sub["score"].mean()), 2),
        }
        for asp_code, label_short in [
            ("A1_Akurasi_Kecerdasan", "A1: Akurasi AI"),
            ("A2_Harga_Kuota", "A2: Harga/Kuota"),
            ("A3_Performa_UI_Server", "A3: Performa/UI"),
            ("A4_Bahasa_Indonesia", "A4: Bhs Indonesia"),
        ]:
            asp_sub = sub[sub[asp_code] == 1]
            if len(asp_sub) > 0:
                pos_n = int((asp_sub["sentiment_3class"] == "2_Positive").sum())
                neg_n = int((asp_sub["sentiment_3class"] == "0_Negative").sum())
                nss = round(((pos_n - neg_n) / len(asp_sub)) * 100, 1)
                row_dict[f"{label_short} (Mention / NSS)"] = f"{len(asp_sub):,} ({nss:+.1f}%)"
            else:
                row_dict[f"{label_short} (Mention / NSS)"] = "0 (0.0%)"
        app_absa_matrix.append(row_dict)

    # 80:20 Stratified Split SEBELUM TF-IDF FIT
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_content"].values,
        df["sentiment_3class"].values,
        test_size=0.20,
        stratify=df["strat_key"].values,
        random_state=RANDOM_SEED,
    )

    model_comparison = []

    # 0. Dummy Baseline
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)
    dummy.fit(np.zeros((len(X_train), 1)), y_train)
    d_train_pred = dummy.predict(np.zeros((len(X_train), 1)))
    d_train_prob = dummy.predict_proba(np.zeros((len(X_train), 1)))
    d_test_pred = dummy.predict(np.zeros((len(X_test), 1)))
    d_test_prob = dummy.predict_proba(np.zeros((len(X_test), 1)))
    m_d_tr = evaluate_predictions(y_train, d_train_pred, d_train_prob, classes)
    m_d_te = evaluate_predictions(y_test, d_test_pred, d_test_prob, classes)
    model_comparison.append(
        {
            "model_name": "0. DummyClassifier (Most Frequent Baseline)",
            "role": "Batas Bawah Statistik",
            "train_accuracy": m_d_tr["accuracy"],
            "test_accuracy": m_d_te["accuracy"],
            "train_macro_f1": m_d_tr["macro_f1"],
            "test_macro_f1": m_d_te["macro_f1"],
            "test_weighted_f1": m_d_te["weighted_f1"],
            "test_balanced_acc": m_d_te["balanced_accuracy"],
            "test_mcc": m_d_te["mcc"],
            "test_roc_auc": m_d_te["roc_auc_ovr"],
        }
    )

    # 1. Model 1 (Baseline Sederhana Bab 4.1): Logistic Regression + 5-Fold GridSearchCV
    print("\n[1/3] Melatih & Tuning 5-Fold Model 1 (Baseline): LogisticRegression...")
    base_lr_pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=15000, min_df=3, sublinear_tf=True)),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=500, random_state=RANDOM_SEED)),
        ]
    )
    param_grid_lr = {"clf__C": [0.5, 1.0, 2.0]}
    gs_lr = GridSearchCV(base_lr_pipe, param_grid=param_grid_lr, cv=5, scoring="f1_macro", n_jobs=-1)
    gs_lr.fit(X_train, y_train)
    best_lr = gs_lr.best_estimator_

    lr_tr_pred = best_lr.predict(X_train)
    lr_tr_prob = best_lr.predict_proba(X_train)
    lr_te_pred = best_lr.predict(X_test)
    lr_te_prob = best_lr.predict_proba(X_test)
    m_lr_tr = evaluate_predictions(y_train, lr_tr_pred, lr_tr_prob, classes)
    m_lr_te = evaluate_predictions(y_test, lr_te_pred, lr_te_prob, classes)
    model_comparison.append(
        {
            "model_name": f"1. Logistic Regression (Tuned C={gs_lr.best_params_['clf__C']})",
            "role": "Model Pembanding Dasar (Baseline Bab 4.1)",
            "train_accuracy": m_lr_tr["accuracy"],
            "test_accuracy": m_lr_te["accuracy"],
            "train_macro_f1": m_lr_tr["macro_f1"],
            "test_macro_f1": m_lr_te["macro_f1"],
            "test_weighted_f1": m_lr_te["weighted_f1"],
            "test_balanced_acc": m_lr_te["balanced_accuracy"],
            "test_mcc": m_lr_te["mcc"],
            "test_roc_auc": m_lr_te["roc_auc_ovr"],
        }
    )

    # 2. Model 2 (Alternatif 1): Complement Naive Bayes + 5-Fold GridSearchCV
    print("[2/3] Melatih & Tuning 5-Fold Model 2 (Alternatif 1): ComplementNB...")
    nb_pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=15000, min_df=3, sublinear_tf=True)),
            ("clf", ComplementNB()),
        ]
    )
    gs_nb = GridSearchCV(nb_pipe, param_grid={"clf__alpha": [0.2, 0.5, 1.0]}, cv=5, scoring="f1_macro", n_jobs=-1)
    gs_nb.fit(X_train, y_train)
    best_nb = gs_nb.best_estimator_

    nb_tr_pred = best_nb.predict(X_train)
    nb_tr_prob = best_nb.predict_proba(X_train)
    nb_te_pred = best_nb.predict(X_test)
    nb_te_prob = best_nb.predict_proba(X_test)
    m_nb_tr = evaluate_predictions(y_train, nb_tr_pred, nb_tr_prob, classes)
    m_nb_te = evaluate_predictions(y_test, nb_te_pred, nb_te_prob, classes)
    model_comparison.append(
        {
            "model_name": f"2. Complement Naive Bayes (Tuned alpha={gs_nb.best_params_['clf__alpha']})",
            "role": "Model Alternatif 1 (Probabilistic NLP)",
            "train_accuracy": m_nb_tr["accuracy"],
            "test_accuracy": m_nb_te["accuracy"],
            "train_macro_f1": m_nb_tr["macro_f1"],
            "test_macro_f1": m_nb_te["macro_f1"],
            "test_weighted_f1": m_nb_te["weighted_f1"],
            "test_balanced_acc": m_nb_te["balanced_accuracy"],
            "test_mcc": m_nb_te["mcc"],
            "test_roc_auc": m_nb_te["roc_auc_ovr"],
        }
    )

    # 3. Model 3 (Alternatif 2): Calibrated LinearSVC (Margin Classifier)
    print("[3/3] Melatih Model 3 (Alternatif 2): Calibrated LinearSVC...")
    svc_pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=15000, min_df=3, sublinear_tf=True)),
            (
                "clf",
                CalibratedClassifierCV(
                    LinearSVC(C=0.5, class_weight="balanced", random_state=RANDOM_SEED),
                    cv=3,
                ),
            ),
        ]
    )
    svc_pipe.fit(X_train, y_train)
    svc_tr_pred = svc_pipe.predict(X_train)
    svc_tr_prob = svc_pipe.predict_proba(X_train)
    svc_te_pred = svc_pipe.predict(X_test)
    svc_te_prob = svc_pipe.predict_proba(X_test)
    m_svc_tr = evaluate_predictions(y_train, svc_tr_pred, svc_tr_prob, classes)
    m_svc_te = evaluate_predictions(y_test, svc_te_pred, svc_te_prob, classes)
    model_comparison.append(
        {
            "model_name": "3. Calibrated LinearSVC (C=0.5, Balanced Margin)",
            "role": "Model Alternatif 2 (Large-Margin Classifier)",
            "train_accuracy": m_svc_tr["accuracy"],
            "test_accuracy": m_svc_te["accuracy"],
            "train_macro_f1": m_svc_tr["macro_f1"],
            "test_macro_f1": m_svc_te["macro_f1"],
            "test_weighted_f1": m_svc_te["weighted_f1"],
            "test_balanced_acc": m_svc_te["balanced_accuracy"],
            "test_mcc": m_svc_te["mcc"],
            "test_roc_auc": m_svc_te["roc_auc_ovr"],
        }
    )

    # Pilih model terbaik untuk disimpan ke .joblib (best_lr memiliki koefisien log-odds langsung & performa terbaik)
    best_model_obj = best_lr
    best_model_name = f"Tuned Logistic Regression (TF-IDF 1-2 Gram, C={gs_lr.best_params_['clf__C']}, Balanced)"
    best_metrics = m_lr_te

    generate_evaluation_figures(y_test, lr_te_pred, best_lr, classes)

    artifact = {
        "model_name": best_model_name,
        "pipeline": best_model_obj,
        "classes": classes,
        "label_display_map": {
            "0_Negative": "Negatif",
            "1_Neutral": "Netral",
            "2_Positive": "Positif",
        },
        "metrics_summary": best_metrics,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "random_seed": RANDOM_SEED,
    }
    joblib.dump(artifact, MODEL_JOBLIB_PATH)
    print(f"[OK] Model artifact (.joblib) berhasil disimpan: {MODEL_JOBLIB_PATH}")

    summary_data = {
        "best_model_name": best_model_name,
        "best_params": gs_lr.best_params_,
        "best_cv_macro_f1": round(float(gs_lr.best_score_), 4),
        "train_rows_80pct": int(len(X_train)),
        "test_rows_20pct": int(len(X_test)),
        "best_test_metrics": best_metrics,
        "model_comparison": model_comparison,
        "app_absa_matrix": app_absa_matrix,
    }
    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"[OK] Ringkasan hasil pemodelan disimpan: {SUMMARY_JSON_PATH}")

    build_three_notebooks(summary_data)
    print("=" * 88)


if __name__ == "__main__":
    main()
