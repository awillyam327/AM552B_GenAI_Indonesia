"""
Pelatihan Model Klasifikasi Sentimen & Hyperparameter Tuning 5-Fold CV
Mata Kuliah: AM552 - Data Science of IS (FTI UKSW)
Oleh: Arthur Willyam Liang (682024046)
"""

import json
import os
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
    ax.set_xlabel("Label Prediksi Model", fontsize=10.5, fontweight="bold", color="#1F4E79")
    ax.set_ylabel("Label Aktual", fontsize=10.5, fontweight="bold", color="#1F4E79")
    ax.set_title(
        "Confusion Matrix Model Terbaik pada Data Uji (N = 14.245 Baris)",
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

    vec = best_pipe.named_steps["tfidf"]
    clf = best_pipe.named_steps["clf"]
    feature_names = np.asarray(vec.get_feature_names_out())
    coefs = clf.coef_

    top_neg_idx = np.argsort(coefs[0])[-10:]
    top_pos_idx = np.argsort(coefs[2])[-10:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=180)

    ax1.barh(feature_names[top_neg_idx], coefs[0][top_neg_idx], color="#D9534F", height=0.62)
    ax1.set_title("Top 10 Fitur Pendorong Sentimen NEGATIF", fontsize=11, fontweight="bold", color="#900C3F")
    ax1.set_xlabel("Bobot Koefisien Log-Odds", fontsize=9.5)
    ax1.grid(axis="x", linestyle="--", alpha=0.35)

    ax2.barh(feature_names[top_pos_idx], coefs[2][top_pos_idx], color="#1E8449", height=0.62)
    ax2.set_title("Top 10 Fitur Pendorong Sentimen POSITIF", fontsize=11, fontweight="bold", color="#145A32")
    ax2.set_xlabel("Bobot Koefisien Log-Odds", fontsize=9.5)
    ax2.grid(axis="x", linestyle="--", alpha=0.35)

    plt.suptitle(
        "Atribusi Fitur Kata Kunci Terpenting pada Klasifikasi Sentimen",
        fontsize=12.5,
        fontweight="bold",
        color="#0F243E",
        y=1.02,
    )
    plt.tight_layout()
    feat_path = os.path.join(EVAL_FIG_DIR, "02_feature_attribution_shap_top_words.png")
    plt.savefig(feat_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    slang_map = load_slang()

    df = pd.read_csv(RAW_CSV_PATH, low_memory=False)
    df["clean_content"] = df["content"].apply(lambda x: preprocess_text(x, slang_map))
    df["sentiment_3class"] = df["score"].apply(map_3class)
    df["strat_key"] = df["app_name"].astype(str) + "_" + df["sentiment_3class"].astype(str)

    classes = ["0_Negative", "1_Neutral", "2_Positive"]

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

    X_train_list, X_test_list, y_train_list, y_test_list = train_test_split(
        df["clean_content"].tolist(),
        df["sentiment_3class"].tolist(),
        test_size=0.20,
        stratify=df["strat_key"].tolist(),
        random_state=RANDOM_SEED,
    )
    X_train = np.array(X_train_list, dtype=object)
    X_test = np.array(X_test_list, dtype=object)
    y_train = np.array(y_train_list)
    y_test = np.array(y_test_list)

    model_comparison = []

    # 0. Dummy Baseline
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)
    dummy.fit(np.zeros((len(X_train), 1)), y_train)
    m_d_tr = evaluate_predictions(y_train, dummy.predict(np.zeros((len(X_train), 1))), dummy.predict_proba(np.zeros((len(X_train), 1))), classes)
    m_d_te = evaluate_predictions(y_test, dummy.predict(np.zeros((len(X_test), 1))), dummy.predict_proba(np.zeros((len(X_test), 1))), classes)
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

    # 1. Logistic Regression + 5-Fold GridSearchCV
    print("[1/3] Melatih & Tuning 5-Fold LogisticRegression...")
    base_lr_pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=15000, min_df=3, sublinear_tf=True)),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=500, random_state=RANDOM_SEED)),
        ]
    )
    gs_lr = GridSearchCV(base_lr_pipe, param_grid={"clf__C": [0.5, 1.0, 2.0]}, cv=5, scoring="f1_macro", n_jobs=-1)
    gs_lr.fit(X_train, y_train)
    best_lr = gs_lr.best_estimator_

    lr_te_pred = best_lr.predict(X_test)
    m_lr_tr = evaluate_predictions(y_train, best_lr.predict(X_train), best_lr.predict_proba(X_train), classes)
    m_lr_te = evaluate_predictions(y_test, lr_te_pred, best_lr.predict_proba(X_test), classes)
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

    # 2. Complement Naive Bayes + 5-Fold GridSearchCV
    print("[2/3] Melatih & Tuning 5-Fold ComplementNB...")
    nb_pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=15000, min_df=3, sublinear_tf=True)),
            ("clf", ComplementNB()),
        ]
    )
    gs_nb = GridSearchCV(nb_pipe, param_grid={"clf__alpha": [0.2, 0.5, 1.0]}, cv=5, scoring="f1_macro", n_jobs=-1)
    gs_nb.fit(X_train, y_train)
    best_nb = gs_nb.best_estimator_

    m_nb_tr = evaluate_predictions(y_train, best_nb.predict(X_train), best_nb.predict_proba(X_train), classes)
    m_nb_te = evaluate_predictions(y_test, best_nb.predict(X_test), best_nb.predict_proba(X_test), classes)
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

    # 3. Calibrated LinearSVC
    print("[3/3] Melatih Calibrated LinearSVC...")
    svc_pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=15000, min_df=3, sublinear_tf=True)),
            ("clf", CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced", random_state=RANDOM_SEED), cv=3)),
        ]
    )
    svc_pipe.fit(X_train, y_train)
    m_svc_tr = evaluate_predictions(y_train, svc_pipe.predict(X_train), svc_pipe.predict_proba(X_train), classes)
    m_svc_te = evaluate_predictions(y_test, svc_pipe.predict(X_test), svc_pipe.predict_proba(X_test), classes)
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

    best_model_name = f"Tuned Logistic Regression (TF-IDF 1-2 Gram, C={gs_lr.best_params_['clf__C']}, Balanced)"
    generate_evaluation_figures(y_test, lr_te_pred, best_lr, classes)

    artifact = {
        "model_name": best_model_name,
        "pipeline": best_lr,
        "classes": classes,
        "label_display_map": {
            "0_Negative": "Negatif",
            "1_Neutral": "Netral",
            "2_Positive": "Positif",
        },
        "metrics_summary": m_lr_te,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "random_seed": RANDOM_SEED,
    }
    joblib.dump(artifact, MODEL_JOBLIB_PATH)
    print(f"Model tersimpan di: {MODEL_JOBLIB_PATH}")

    summary_data = {
        "best_model_name": best_model_name,
        "best_params": gs_lr.best_params_,
        "best_cv_macro_f1": round(float(gs_lr.best_score_), 4),
        "train_rows_80pct": int(len(X_train)),
        "test_rows_20pct": int(len(X_test)),
        "best_test_metrics": m_lr_te,
        "model_comparison": model_comparison,
        "app_absa_matrix": app_absa_matrix,
    }
    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"Ringkasan hasil disimpan di: {SUMMARY_JSON_PATH}")


if __name__ == "__main__":
    main()
