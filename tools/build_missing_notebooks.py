"""
Builder & In-Process Executor Notebook 04 (.ipynb) dengan output lengkap (stdout, HTML tables, PNG plots).
Menjalankan seluruh kode menggunakan environment proyek (.venv, scikit-learn 1.9.1) tanpa hang kernel ZMQ.
"""

import os
import io
import sys
import json
import base64
import contextlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NB_DIR = os.path.join(ROOT_DIR, "notebooks")
NB04_PATH = os.path.join(NB_DIR, "04_Topic_Modeling_LDA_Clause_ABSA_dan_Verifikasi_Prototipe.ipynb")


def make_md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def make_code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def execute_notebook_inprocess(nb: dict, cwd: str):
    """Jalankan setiap sel kode secara berurutan dan simpan output standar Jupyter (stream, HTML, PNG)."""
    orig_cwd = os.getcwd()
    os.chdir(cwd)
    exec_ns = {"__name__": "__main__"}

    try:
        exec_count = 0
        for idx, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            exec_count += 1
            cell["execution_count"] = exec_count
            outputs = []

            def _display(obj):
                if isinstance(obj, pd.DataFrame):
                    outputs.append({
                        "output_type": "display_data",
                        "data": {
                            "text/plain": obj.to_string(),
                            "text/html": obj.to_html(border=0, classes="dataframe"),
                        },
                        "metadata": {},
                    })
                else:
                    outputs.append({
                        "output_type": "display_data",
                        "data": {"text/plain": repr(obj)},
                        "metadata": {},
                    })

            def _show(*args, **kwargs):
                for fignum in plt.get_fignums():
                    fig = plt.figure(fignum)
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
                    buf.seek(0)
                    b64 = base64.b64encode(buf.read()).decode("ascii")
                    outputs.append({
                        "output_type": "display_data",
                        "data": {
                            "image/png": b64,
                            "text/plain": "<Figure>",
                        },
                        "metadata": {},
                    })
                    plt.close(fig)

            exec_ns["display"] = _display
            orig_show = plt.show
            plt.show = _show

            stdout_buf = io.StringIO()
            src = "".join(cell["source"])
            print(f"[*] Menjalankan Sel [{idx}] (exec_count={exec_count})...", flush=True)
            try:
                with contextlib.redirect_stdout(stdout_buf):
                    exec(compile(src, f"<cell-{idx}>", "exec"), exec_ns)
            finally:
                plt.show = orig_show

            out_text = stdout_buf.getvalue()
            if out_text:
                outputs.insert(0, {
                    "output_type": "stream",
                    "name": "stdout",
                    "text": out_text.splitlines(keepends=True),
                })
            cell["outputs"] = outputs
            print(f"    [OK] Sel [{idx}] selesai ({len(outputs)} output).", flush=True)
    finally:
        os.chdir(orig_cwd)


def build_and_run_notebook_04():
    cells = [
        make_md(
            "# Notebook 04: Dokumentasi Lengkap Topic Modeling (LDA), Normalisasi Slang & Typo, Clause-Level ABSA, Dashboard Komparasi, dan Pengujian 7 Skenario Prototipe\n\n"
            "**Mata Kuliah**: AM552B - Data Science of Information Systems (S1 Sistem Informasi, FTI UKSW)  \n"
            "**Proyek**: Analisis Komparatif Sentimen Berbasis Aspek (ABSA) dan Pemodelan Topik Pengalaman Pengguna terhadap Aplikasi Mobile Generative AI di Indonesia  \n"
            "**Tujuan Notebook**: Melengkapi dokumentasi *reproducible* `.ipynb` atas seluruh kegiatan & klaim pada *Progress Report Kelompok 2* yang sebelumnya hanya tersimpan pada skrip Python terpisah:\n"
            "1. **Verifikasi Kamus Slang (1.100 Entri) & Koreksi Typo Otomatis** (*Smart Elongation Reduction* & *Damerau-Levenshtein*).\n"
            "2. **Unsupervised Topic Modeling (LDA)** — 5 klaster topik keluhan, 5 klaster topik pujian, serta bedah kasus **Character.AI (1,58)** & **DeepSeek (3,77)**.\n"
            "3. **Clause-Level ABSA** — Pemecahan kalimat campuran berdasarkan konjungsi kontras (*\"tapi\"*, *\"namun\"*, *\"padahal\"*) untuk sentimen spesifik per aspek.\n"
            "4. **Discrepancy-Adjusted Rating, Matriks Kepuasan 4 Aspek (NSS) & Verifikasi Time-Series Dashboard**.\n"
            "5. **Audit Artefak Model `.joblib` (0,91 MB) & Eksekusi Live 7 Skenario Pengujian Prototipe**."
        ),
        make_code(
            """import os
import re
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

ROOT_DIR = os.path.abspath("..")
PROTOTYPE_DIR = os.path.join(ROOT_DIR, "prototype")
if PROTOTYPE_DIR not in sys.path:
    sys.path.insert(0, PROTOTYPE_DIR)

from core_predictor import (
    MODEL_PATH,
    SLANG_PATH,
    BENCHMARK_JSON_PATH,
    get_fresh_slang_dict,
    load_trained_artifact,
    build_typo_vocab_index,
    preprocess_text_indonesian,
    split_into_clean_clauses,
    predict_single_review,
    predict_batch_dataframe,
)

RAW_CSV_PATH = os.path.join(ROOT_DIR, "dataset", "RAW_DATASET_GENAI_INDONESIA_AM522B.csv")
df = pd.read_csv(RAW_CSV_PATH, low_memory=False)
artifact = load_trained_artifact()
vocab_index = build_typo_vocab_index(artifact["pipeline"])

print(f"Dataset terbaca : {len(df):,} baris x {df.shape[1]} kolom")
print(f"Rentang waktu   : {df['at'].min()} s/d {df['at'].max()}")
print(f"Model produksi  : {artifact['model_name']}")
"""
        ),
        make_md(
            "---\n"
            "## 1. Verifikasi Kamus Normalisasi Slang (1.100 Entri) & Koreksi Typo Otomatis\n"
            "Mendokumentasikan pembaruan kamus bahasa tidak baku (`03_data_preparation/slang_dictionary_id.json`) "
            "dan mekanisme koreksi salah ketik otomatis pada `prototype/core_predictor.py` menggunakan dua tahap:\n"
            "1. **Smart Elongation Reduction**: memendekkan huruf berulang berlebihan (mis. `baguuusss` → `bagus`, `lemottt` → `lambat`).\n"
            "2. **Fuzzy Damerau-Levenshtein**: mengoreksi salah ketik OOV panjang ≥ 4 huruf (mis. `akursi` → `akurasi`, `bagsu` → `bagus`)."
        ),
        make_code(
            """slang_dict = get_fresh_slang_dict()
print(f"Total entri kamus slang (slang_dictionary_id.json): {len(slang_dict):,} entri")

sample_keys = ["gk", "ga", "bgt", "apk", "lemot", "eror", "udh", "blm", "mantul", "ngawur"]
sample_slang_df = pd.DataFrame([
    {"Kata Slang / Singkatan": k, "Bentuk Baku": slang_dict.get(k, "(tidak ada)")}
    for k in sample_keys
])
print("\\nContoh Pemetaan Kamus Slang:")
display(sample_slang_df)

# Uji Koreksi Typo Otomatis & Elongation Reduction
typo_samples = [
    "akursi jawabannya baguuusss bgt tp apk sering lemottt",
    "bagsu sekali sangat membatnu tugas kuliah",
    "servre sibuk terus dan ga bisa logni",
    "jawban ai sering ngawur dan eror saat updte",
]
typo_rows = []
for raw_s in typo_samples:
    clean_s, corrs = preprocess_text_indonesian(
        raw_s, slang_dict=slang_dict, vocab_index=vocab_index, return_corrections=True
    )
    corr_txt = ", ".join(f"{c['asal']} -> {c['koreksi']}" for c in corrs) if corrs else "-"
    typo_rows.append({
        "Teks Asli (Slang + Typo)": raw_s,
        "Hasil Normalisasi": clean_s,
        "Koreksi Typo Otomatis": corr_txt,
    })

print("\\nHasil Uji Koreksi Typo & Normalisasi Slang Otomatis:")
display(pd.DataFrame(typo_rows))
"""
        ),
        make_md(
            "---\n"
            "## 2. Unsupervised Topic Modeling (Latent Dirichlet Allocation / LDA)\n"
            "Mengekstraksi klaster topik tanpa label secara otomatis dari 71.221 ulasan menggunakan `CountVectorizer(ngram_range=(1, 2), max_features=3000, min_df=3)` dan `LatentDirichletAllocation`:\n"
            "- **5 Klaster Topik Keluhan Utama** (korpus Bintang 1–2)\n"
            "- **5 Klaster Topik Pujian Utama** (korpus Bintang 4–5)\n"
            "- **Bedah Kasus Aplikasi Spesifik**: Mengapa **Character.AI** anjlok ke **1,58** dan **DeepSeek** tertahan di **3,77**."
        ),
        make_code(
            """CUSTOM_STOPWORDS = {
    "yang", "di", "dan", "ini", "itu", "dengan", "untuk", "pada", "dari", "ke",
    "dalam", "saya", "aku", "nya", "ya", "saja", "juga", "atau", "ada", "kalau",
    "karena", "sudah", "bisa", "sama", "jadi", "lebih", "mau", "apa", "lagi",
    "buat", "kan", "lah", "kok", "sih", "deh", "dong", "nih", "tuh", "pun",
    "kami", "kita", "mereka", "dia", "kamu", "anda", "seperti", "saat", "ketika",
    "oleh", "sebagai", "bahwa", "adalah", "yaitu", "the", "to", "and", "of",
    "in", "is", "it", "for", "on", "with", "as", "this", "that", "aplikasi", "apk",
    "app", "chat", "ai", "banget", "bgt", "sangat", "tolong", "bintang"
}

def clean_for_lda(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\\S+|www\\S+", "", text)
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    cleaned = []
    for t in text.split():
        w = slang_dict.get(t, t)
        if len(w) >= 3 and w not in CUSTOM_STOPWORDS:
            cleaned.append(w)
    return " ".join(cleaned)

def run_lda_topics(corpus, n_topics=3, n_top_words=7, max_features=1500):
    vec = CountVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=3)
    dtm = vec.fit_transform(corpus)
    feat_names = vec.get_feature_names_out()
    lda = LatentDirichletAllocation(
        n_components=n_topics, random_state=42, max_iter=5,
        learning_method="batch", n_jobs=1
    )
    lda.fit(dtm)
    res = []
    for idx, comp in enumerate(lda.components_):
        top_idx = comp.argsort()[:-n_top_words - 1:-1]
        res.append({
            "Topik": f"Topik {idx + 1}",
            "Top Keywords": ", ".join(feat_names[i] for i in top_idx),
            "words": [feat_names[i] for i in top_idx],
            "weights": [float(comp[i]) for i in top_idx],
        })
    return res

# 1. Muat hasil eksperimen LDA resmi (04_modeling/topic_modeling_results.json) dari 71.221 ulasan
lda_json_path = os.path.join(ROOT_DIR, "04_modeling", "topic_modeling_results.json")
with open(lda_json_path, "r", encoding="utf-8") as f:
    lda_official = json.load(f)

neg_topics = lda_official["global_complaint_topics"]
pos_topics = lda_official["global_praise_topics"]

print(f"Korpus LDA Global (resmi): Total={lda_official['dataset_size']:,} | Keluhan (Bintang 1-2)={lda_official['negative_reviews_count']:,} | Pujian (Bintang 4-5)={lda_official['positive_reviews_count']:,}")

print("\\n=== 5 KLASTER TOPIK KELUHAN GLOBAL (BINTANG 1-2) ===")
display(pd.DataFrame([
    {"ID": f"Topik {t['topic_id']}", "Label Bisnis": t["business_label"], "Top Keywords (LDA)": ", ".join(t["top_words"])}
    for t in neg_topics
]))

print("\\n=== 5 KLASTER TOPIK PUJIAN GLOBAL (BINTANG 4-5) ===")
display(pd.DataFrame([
    {"ID": f"Topik {t['topic_id']}", "Label Bisnis": t["business_label"], "Top Keywords (LDA)": ", ".join(t["top_words"])}
    for t in pos_topics
]))

# 2. Verifikasi Live LDA Bedah Kasus Aplikasi Spesifik: Character.AI (1,58) & DeepSeek (3,77)
sub_apps = df[df["app_name"].isin(["Character.AI", "DeepSeek - AI Assistant"])].copy()
sub_apps["lda_clean"] = sub_apps["content"].apply(clean_for_lda)

cai_corpus = sub_apps[(sub_apps["app_name"] == "Character.AI") & sub_apps["score"].isin([1, 2]) & (sub_apps["lda_clean"].str.len() > 5)]["lda_clean"].tolist()
deep_corpus = sub_apps[(sub_apps["app_name"] == "DeepSeek - AI Assistant") & sub_apps["score"].isin([1, 2, 3]) & (sub_apps["lda_clean"].str.len() > 5)]["lda_clean"].tolist()

cai_topics = run_lda_topics(cai_corpus, n_topics=3, n_top_words=7)
deep_topics = run_lda_topics(deep_corpus, n_topics=3, n_top_words=7)

print(f"\\n=== VERIFIKASI LIVE LDA: CHARACTER.AI (Rata-rata Rating 1,58 | N={len(cai_corpus):,}) ===")
display(pd.DataFrame([{"Topik": t["Topik"], "Kata Kunci Keluhan Character.AI": t["Top Keywords"]} for t in cai_topics]))

print(f"\\n=== VERIFIKASI LIVE LDA: DEEPSEEK (Rata-rata Rating 3,77 | N={len(deep_corpus):,}) ===")
display(pd.DataFrame([{"Topik": t["Topik"], "Kata Kunci Keluhan DeepSeek": t["Top Keywords"]} for t in deep_topics]))

# 3. Visualisasi Bobot Kata Kunci Top-3 Topik Keluhan Global (palet senada DSLC)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=120)
palette = ["#1F3B5C", "#315E8D", "#3B6FA1"]
for i in range(3):
    t = neg_topics[i]
    w_rev = t["top_words"][::-1]
    v_rev = t["weights"][::-1]
    axes[i].barh(w_rev, v_rev, color=palette[i])
    axes[i].set_title(f"Topik {t['topic_id']}: {t['business_label'][:34]}...", fontweight="bold", color="#1B314C", fontsize=9.5)
    axes[i].set_xlabel("Bobot Skor LDA")
    axes[i].grid(axis="x", linestyle="--", alpha=0.35)
plt.suptitle("Visualisasi Kata Kunci Dominan Hasil Topic Modeling LDA (Ulasan Keluhan Bintang 1-2)", fontweight="bold", color="#1B314C", y=1.02)
plt.tight_layout()
plt.show()
"""
        ),
        make_md(
            "---\n"
            "## 3. Clause-Level ABSA (Analisis Sentimen per Aspek Berbasis Klausa)\n"
            "Ulasan campuran sering memuat pujian pada satu aspek namun keluhan pada aspek lain. "
            "Modul `split_into_clean_clauses()` dan `enrich_aspects_with_sentiment()` memecah kalimat berdasarkan tanda baca "
            "dan kata hubung kontras (*tapi*, *tetapi*, *namun*, *padahal*, *cuma*, *sayang*) agar tiap aspek menerima label sentimen spesifiknya."
        ),
        make_code(
            """mixed_examples = [
    ("jawabannya akurat, tapi kuota cepat habis dan mahal", 3, "ChatGPT (OpenAI)"),
    ("jawabannya akurat, tapi kuotanya dikit", 3, "ChatGPT (OpenAI)"),
    ("AI nya pintar banget buat bantu tugas coding, namun server sering sibuk dan lemot", 4, "DeepSeek - AI Assistant"),
    ("Bahasa Indonesia sudah luwes dan cepat, cuma harga langganan plus terlalu mahal", 4, "Google Gemini"),
    ("Tampilan UI nyaman dan respons cepat, tapi jawaban kadang ngawur dan halusinasi", 3, "Perplexity - Ask Anything"),
    ("Saya kasih bintang 5 biar dibaca developer, aplikasi makin update makin lemot dan banyak iklan", 5, "Character.AI"),
]

clause_rows = []
for raw_txt, stars, app_nm in mixed_examples:
    clauses = split_into_clean_clauses(raw_txt, vocab_index=vocab_index)
    res = predict_single_review(raw_txt, star_rating=stars, app_name=app_nm, artifact=artifact)
    asp_summary = " | ".join(
        f"{a['aspect_label']} -> {a.get('aspect_sentiment', res['predicted_sentiment'])} ({a.get('aspect_confidence', res['confidence'])*100:.0f}%)"
        for a in res["detected_aspects"]
    )
    clause_rows.append({
        "Ulasan Campuran": raw_txt,
        "Rating": stars,
        "Klausa Terpecah": " // ".join(clauses),
        "Sentimen Global": f"{res['predicted_sentiment']} ({res['confidence']*100:.1f}%)",
        "Sentimen per Aspek (Clause-Level ABSA)": asp_summary,
        "Discrepancy": "YA" if res["is_discrepancy"] else "Konsisten",
    })

pd.set_option("display.max_colwidth", None)
display(pd.DataFrame(clause_rows))
"""
        ),
        make_md(
            "---\n"
            "## 4. Discrepancy-Adjusted Rating, Matriks Kepuasan 4 Aspek (NSS) & Verifikasi Time-Series Dashboard\n"
            "Mendokumentasikan data produksi yang ditampilkan pada **Menu 1 (Dashboard Komparasi)** di aplikasi Streamlit: "
            "perbandingan *Rating Play Store* vs *Rating Aktual Terkoreksi*, Net Sentiment Score (NSS) per aspek A1–A4, "
            "serta verifikasi file agregasi rentang tanggal (`models/dashboard_daily_metrics.csv`)."
        ),
        make_code(
            """with open(BENCHMARK_JSON_PATH, "r", encoding="utf-8") as f:
    bench = json.load(f)

df_rating_comp = pd.DataFrame(bench["rating_comparison"])
print("=== PERBANDINGAN RATING PLAY STORE VS RATING AKTUAL (DISCREPANCY-ADJUSTED) ===")
display(df_rating_comp)

df_asp_num = pd.DataFrame(bench["aspect_numeric_matrix"]).set_index("Nama Aplikasi")
print("\\n=== MATRIKS NET SENTIMENT SCORE (NSS %) PER 4 ASPEK LAYANAN ===")
display(df_asp_num)

# Heatmap Matriks NSS 4 Aspek Lintas 8 Aplikasi
fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=120)
im = ax.imshow(df_asp_num.values, cmap="RdYlGn", vmin=-100, vmax=100, aspect="auto")
ax.set_xticks(range(len(df_asp_num.columns)))
ax.set_xticklabels(df_asp_num.columns, fontweight="bold")
ax.set_yticks(range(len(df_asp_num.index)))
ax.set_yticklabels(df_asp_num.index)
for i in range(len(df_asp_num.index)):
    for j in range(len(df_asp_num.columns)):
        val = df_asp_num.values[i, j]
        txt_col = "white" if abs(val) > 55 else "#1B314C"
        ax.text(j, i, f"{val:+.1f}%", ha="center", va="center", fontweight="bold", color=txt_col, fontsize=9)
plt.colorbar(im, ax=ax, label="Net Sentiment Score (%)")
ax.set_title("Peta Kepuasan 4 Dimensi Aspek Layanan (ABSA) Lintas 8 Aplikasi Generative AI", fontweight="bold", color="#1B314C", pad=12)
plt.tight_layout()
plt.show()

daily_path = os.path.join(ROOT_DIR, "models", "dashboard_daily_metrics.csv")
if os.path.exists(daily_path):
    df_daily = pd.read_csv(daily_path)
    print(f"\\nVerifikasi Dashboard Time-Series (dashboard_daily_metrics.csv):")
    print(f"  - Total baris agregasi harian : {len(df_daily):,}")
    print(f"  - Kolom tersedia              : {list(df_daily.columns)}")
    if "date" in df_daily.columns:
        print(f"  - Rentang tanggal filter      : {df_daily['date'].min()} s/d {df_daily['date'].max()}")
"""
        ),
        make_md(
            "---\n"
            "## 5. Verifikasi Artefak Model `.joblib` & Pengujian 7 Skenario Prototipe\n"
            "Memverifikasi spesifikasi aktual file model tersimpan (`models/genai_absa_pipeline.joblib`), metrik evaluasi "
            "eksak 4 desimal pada data uji (14.245 baris), serta mengeksekusi secara langsung seluruh **7 skenario pengujian "
            "prototipe** (Input Normal Positif & Negatif, Input Ekstrem Slang/Discrepancy & >3.000 Karakter, Input Kosong, "
            "Input Angka/Simbol, dan CSV Salah Format)."
        ),
        make_code(
            """size_bytes = os.path.getsize(MODEL_PATH)
size_mb = size_bytes / (1024 * 1024)
vec = artifact["pipeline"].named_steps["tfidf"]
clf = artifact["pipeline"].named_steps["clf"]

print("=== SPESIFIKASI AKTUAL ARTEFAK MODEL PRODUKSI ===")
print(f"Path file        : {MODEL_PATH}")
print(f"Ukuran file      : {size_mb:.2f} MB ({size_bytes:,} bytes)")
print(f"Model terpilih   : {artifact['model_name']}")
print(f"TF-IDF features  : {len(vec.vocabulary_):,} fitur (ngram_range={vec.ngram_range})")
print(f"Hyperparameter   : C={clf.C}, class_weight={clf.class_weight}")

print("\\n=== METRIK EVALUASI EKSAK (DATA UJI 20% = 14.245 ULASAN) ===")
metrics_df = pd.DataFrame([
    {"Metrik Evaluasi": k, "Nilai Aktual (4 Desimal)": f"{v:.4f}", "Pembulatan 3 Desimal": f"{v:.3f}"}
    for k, v in bench["best_test_metrics"].items()
])
display(metrics_df)

scenarios = []

r1a = predict_single_review(
    "Aplikasinya sangat pintar dan cepat, bahasa Indonesianya luwes membantu tugas coding kuliah saya!",
    star_rating=5, app_name="ChatGPT (OpenAI)", artifact=artifact
)
scenarios.append(("1A. Input Normal Positif", r1a["status"], f"{r1a['predicted_sentiment']} ({r1a['confidence']*100:.1f}%)",
                  ", ".join(a["aspect_label"] for a in r1a["detected_aspects"]), r1a["discrepancy_status"]))

r1b = predict_single_review(
    "Server sibuk terus dan gagal login, padahal kuota gratisnya cepat habis harus bayar mahal.",
    star_rating=1, app_name="DeepSeek - AI Assistant", artifact=artifact
)
scenarios.append(("1B. Input Normal Negatif", r1b["status"], f"{r1b['predicted_sentiment']} ({r1b['confidence']*100:.1f}%)",
                  ", ".join(a["aspect_label"] for a in r1b["detected_aspects"]), r1b["discrepancy_status"]))

r2a = predict_single_review(
    "GUE KASIH BINTANG 5 BIAR DIBACA DEVELOPER!! Tolong dong apk makin update makin lemot bgt, mode live sering macet eror sendiri dan jawabannya ngawur ga nyambung!!",
    star_rating=5, app_name="Google Gemini", artifact=artifact
)
scenarios.append(("2A. Ekstrem Slang + Discrepancy", r2a["status"], f"{r2a['predicted_sentiment']} ({r2a['confidence']*100:.1f}%)",
                  ", ".join(a["aspect_label"] for a in r2a["detected_aspects"]), r2a["discrepancy_status"]))

long_txt = "Aplikasi sering error macet dan kuota limit cepat habis padahal bayar mahal. " * 40
r2b = predict_single_review(long_txt, star_rating=2, app_name="Claude by Anthropic", artifact=artifact)
scenarios.append((f"2B. Ekstrem Teks Panjang ({len(long_txt):,} kar)", r2b["status"],
                  f"{r2b['predicted_sentiment']} ({r2b['confidence']*100:.1f}%)",
                  ", ".join(a["aspect_label"] for a in r2b["detected_aspects"]),
                  r2b.get("warning_message") or "Terpotong aman"))

r3a = predict_single_review("   ", star_rating=5, artifact=artifact)
scenarios.append(("3A. Input Kosong / Spasi", r3a["status"], "-", "-", r3a.get("error_message", "")))

r3b = predict_single_review("1234567890 ??? !!!", star_rating=3, artifact=artifact)
scenarios.append(("3B. Input Angka & Simbol", r3b["status"], "-", "-", r3b.get("error_message", "")))

bad_csv = pd.DataFrame({"id_transaksi": [1, 2], "nominal_harga": [50000, 75000]})
_, csv_err = predict_batch_dataframe(bad_csv, text_col="content", artifact=artifact)
scenarios.append(("3C. CSV Salah Format", "INVALID_CSV", "-", "-", csv_err))

df_scen = pd.DataFrame(scenarios, columns=["Skenario Pengujian", "Status", "Prediksi Sentimen", "Aspek Terdeteksi", "Catatan / Validasi / Discrepancy"])
print("\\n=== HASIL EKSEKUSI LIVE 7 SKENARIO PENGUJIAN PROTOTIPE ===")
display(df_scen)
"""
        ),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (.venv)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    execute_notebook_inprocess(nb, cwd=NB_DIR)

    with open(NB04_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"\\n[DONE] Notebook 04 selesai dieksekusi & disimpan lengkap: {NB04_PATH}", flush=True)


if __name__ == "__main__":
    build_and_run_notebook_04()
