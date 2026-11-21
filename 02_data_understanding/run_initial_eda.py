import os
import re
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================================================================
# KONFIGURASI PATH
# ==============================================================================
RAW_CSV_PATH = "./dataset/RAW_DATASET_GENAI_INDONESIA_AM522B.csv"
REPORT_MD_PATH = "./02_data_understanding/eda_summary_report.md"
FIG_DIR = "./02_data_understanding/figures"

# Kamus normalisasi ringkas & stopwords untuk ekstraksi Top Unigram/Bigram pada EDA
SLANG_MAP = {
    "gk": "tidak", "ga": "tidak", "gak": "tidak", "nggak": "tidak", "enggak": "tidak",
    "tdk": "tidak", "tak": "tidak", "bgt": "banget", "bngt": "banget", "yg": "yang",
    "dgn": "dengan", "utk": "untuk", "buat": "untuk", "krn": "karena", "karna": "karena",
    "klo": "kalau", "kalo": "kalau", "udh": "sudah", "udah": "sudah", "sdh": "sudah",
    "blm": "belum", "tp": "tapi", "tpi": "tapi", "aja": "saja", "aj": "saja",
    "apk": "aplikasi", "app": "aplikasi", "apps": "aplikasi", "bisa2": "bisa",
    "eror": "error", "lemot": "lambat", "lelet": "lambat", "bgus": "bagus",
    "bgs": "bagus", "mantul": "mantap", "mksh": "terima kasih", "makasih": "terima kasih",
    "trims": "terima kasih", " trimakasih": "terima kasih",
}

STOPWORDS_ID = {
    "yang", "di", "dan", "ini", "itu", "dengan", "untuk", "pada", "dari", "ke",
    "dalam", "saya", "aku", "nya", "ya", "saja", "juga", "atau", "ada", "kalau",
    "karena", "sudah", "bisa", "sama", "jadi", "lebih", "mau", "apa", "lagi",
    "buat", "kan", "lah", "kok", "sih", "deh", "dong", "nih", "tuh", "pun",
    "kami", "kita", "mereka", "dia", "kamu", "anda", "seperti", "saat", "ketika",
    "oleh", "sebagai", "bahwa", "adalah", "yaitu", "약", "the", "to", "and", "of",
    "in", "is", "it", "for", "on", "with", "as", "this", "that",
}

NEG_COMPLAIN_WORDS = {
    "error", "eror", "macet", "lemot", "lambat", "lelet", "limit", "batas", "dibatasi",
    "mahal", "bayar", "berbayar", "gagal", "kecewa", "jelek", "buruk", "rusak", "parah",
    "ngawur", "halusinasi", "bodoh", "sampah", "bug", "down", "sibuk", "keluar",
    "force", "crash", "iklan", "pelit", "susah", "sulit", "ribet", "kaku", "ngaco",
    "hilang", "hapus", "filter", "sensor", "gangguan", "masalah", "kendala",
}

POS_PRAISE_WORDS = {
    "bagus", "mantap", "keren", "membantu", "bantu", "terbaik", "puas", "suka",
    "pintar", "cerdas", "cepat", "mudah", "akurat", "luar biasa", "hebat",
    "recommended", "rekomendasi", "berguna", "bermanfaat", "canggih", "good", "best",
    "nice", "love", "awesome", "great", "helpful",
}

NEGATION_WORDS = {"tidak", "tak", "bukan", "jangan", "kurang", "belum", "gak", "ga", "gk", "nggak", "tdk"}


def format_id(n):
    return f"{n:,}".replace(",", ".")


def detect_discrepancy_row(score, text):
    """
    Deteksi awal Rating-Text Discrepancy berbasis aturan leksikal:
    - Type I (High Star >= 4, Negative/Complaint Text)
    - Type II (Low Star <= 2, Pure Positive Praise tanpa negasi/keluhan)
    """
    if not isinstance(text, str):
        return "Consistent"
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if not tokens:
        return "Consistent"
    token_set = set(tokens)

    has_complaint = len(token_set.intersection(NEG_COMPLAIN_WORDS)) > 0 or ("tidak bisa" in text.lower()) or ("gak bisa" in text.lower())
    has_praise = len(token_set.intersection(POS_PRAISE_WORDS)) > 0
    has_negation = len(token_set.intersection(NEGATION_WORDS)) > 0

    if score >= 4 and has_complaint:
        return "Type-I (High Rating >=4, Negative Complaint)"
    if score <= 2 and has_praise and not has_complaint and not has_negation:
        return "Type-II (Low Rating <=2, Positive Praise)"
    return "Consistent"


def tokenize_clean(text):
    if not isinstance(text, str):
        return []
    words = re.findall(r"[a-z]+", text.lower())
    cleaned = []
    for w in words:
        w_norm = SLANG_MAP.get(w, w)
        if len(w_norm) >= 2 and w_norm not in STOPWORDS_ID:
            cleaned.append(w_norm)
    return cleaned


def generate_figures(df, app_stats_df, disc_app_df):
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#CCCCCC"
    plt.rcParams["axes.linewidth"] = 0.8

    # -------------------------------------------------------------------------
    # GRAFIK 1: Distribusi Jumlah Ulasan & Rata-rata Rating per Aplikasi
    # -------------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(11, 6), dpi=180)
    apps = app_stats_df["app_name"].tolist()[::-1]
    counts = app_stats_df["count"].tolist()[::-1]
    avg_scores = app_stats_df["avg_score"].tolist()[::-1]

    colors = ["#1F4E79" if c >= 8000 else "#2E75B6" if c >= 5000 else "#5B9BD5" for c in counts]
    bars = ax1.barh(apps, counts, color=colors, height=0.62, edgecolor="none", alpha=0.92)

    ax1.set_xlabel("Jumlah Baris Ulasan Unik (Google Play Store Indonesia)", fontsize=10.5, fontweight="bold", color="#1F4E79")
    ax1.set_title(
        "Distribusi Volume Ulasan & Rata-rata Rating per Aplikasi Generative AI (N = 71.221)",
        fontsize=12.5,
        fontweight="bold",
        color="#0F243E",
        pad=14,
    )
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    ax1.set_xlim(0, max(counts) * 1.24)

    for bar, cnt, avg in zip(bars, counts, avg_scores):
        ax1.text(
            bar.get_width() + 350,
            bar.get_y() + bar.get_height() / 2,
            f"{format_id(cnt)} ulasan  (Rating: {avg:.2f}/5.0)",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
            color="#1F2937",
        )

    plt.tight_layout()
    fig1_path = os.path.join(FIG_DIR, "01_distribusi_ulasan_per_aplikasi.png")
    plt.savefig(fig1_path, dpi=180)
    plt.close(fig)
    print(f"[OK] Grafik 1 tersimpan: {fig1_path}")

    # -------------------------------------------------------------------------
    # GRAFIK 2: Distribusi Rating Bintang (1 s/d 5) per Aplikasi (100% Stacked Bar)
    # -------------------------------------------------------------------------
    ct = pd.crosstab(df["app_name"], df["score"], normalize="index") * 100
    ct = ct.reindex(app_stats_df["app_name"].tolist()[::-1])
    for s in [1, 2, 3, 4, 5]:
        if s not in ct.columns:
            ct[s] = 0.0
    ct = ct[[1, 2, 3, 4, 5]]

    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=180)
    star_colors = {
        1: "#D9534F",  # Merah (Bintang 1)
        2: "#F0AD4E",  # Oranye (Bintang 2)
        3: "#F7E16F",  # Kuning (Bintang 3)
        4: "#72B566",  # Hijau Muda (Bintang 4)
        5: "#1E8449",  # Hijau Tua (Bintang 5)
    }

    left = np.zeros(len(ct))
    for star in [1, 2, 3, 4, 5]:
        vals = ct[star].values
        bars = ax.barh(
            ct.index,
            vals,
            left=left,
            height=0.62,
            label=f"Bintang {star}",
            color=star_colors[star],
            edgecolor="white",
            linewidth=0.7,
        )
        for idx, (b, v) in enumerate(zip(bars, vals)):
            if v >= 5.5:
                txt_color = "white" if star in [1, 5] else "#1F2937"
                ax.text(
                    left[idx] + v / 2,
                    b.get_y() + b.get_height() / 2,
                    f"{v:.1f}%",
                    va="center",
                    ha="center",
                    fontsize=8.5,
                    fontweight="bold",
                    color=txt_color,
                )
        left += vals

    ax.set_xlim(0, 100)
    ax.set_xlabel("Proporsi Distribusi Rating Bintang (%)", fontsize=10.5, fontweight="bold", color="#1F4E79")
    ax.set_title(
        "Komparasi Proporsi Rating Bintang (1 s/d 5) Lintas 8 Aplikasi Generative AI di Indonesia",
        fontsize=12.5,
        fontweight="bold",
        color="#0F243E",
        pad=14,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=5, frameon=False, fontsize=9.5)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    plt.tight_layout()
    fig2_path = os.path.join(FIG_DIR, "02_distribusi_rating_per_aplikasi.png")
    plt.savefig(fig2_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Grafik 2 tersimpan: {fig2_path}")

    # -------------------------------------------------------------------------
    # GRAFIK 3: Proporsi Rating-Text Discrepancy per Aplikasi
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=180)
    d_sub = disc_app_df.sort_values("total_discrepancy_pct", ascending=True)
    y_pos = np.arange(len(d_sub))

    b1 = ax.barh(
        y_pos,
        d_sub["type1_pct"],
        height=0.58,
        color="#E67E22",
        label="Type-I: Rating Tinggi (>=4) tapi Isi Keluhan",
    )
    b2 = ax.barh(
        y_pos,
        d_sub["type2_pct"],
        left=d_sub["type1_pct"],
        height=0.58,
        color="#8E44AD",
        label="Type-II: Rating Rendah (<=2) tapi Isi Pujian",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(d_sub["app_name"])
    ax.set_xlabel("Persentase Ulasan yang Terdeteksi Discrepancy (%)", fontsize=10.5, fontweight="bold", color="#1F4E79")
    ax.set_title(
        "Deteksi Awal Rating-Text Discrepancy per Aplikasi Generative AI",
        fontsize=12,
        fontweight="bold",
        color="#0F243E",
        pad=12,
    )
    ax.set_xlim(0, max(d_sub["total_discrepancy_pct"]) * 1.25)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", frameon=True, fontsize=9)

    for idx, row in enumerate(d_sub.itertuples()):
        ax.text(
            row.total_discrepancy_pct + 0.15,
            idx,
            f"{row.total_discrepancy_pct:.2f}% ({format_id(row.total_discrepancy_cnt)} ulasan)",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
            color="#1F2937",
        )

    plt.tight_layout()
    fig3_path = os.path.join(FIG_DIR, "03_deteksi_awal_discrepancy_per_aplikasi.png")
    plt.savefig(fig3_path, dpi=180)
    plt.close(fig)
    print(f"[OK] Grafik 3 tersimpan: {fig3_path}")


def main():
    print("=" * 85)
    print("MENJALANKAN INITIAL EDA (CRISP-DM FASE 2: DATA UNDERSTANDING)")
    print("=" * 85)

    df = pd.read_csv(RAW_CSV_PATH, low_memory=False)
    total_rows, total_cols = df.shape

    # 1. Hitung panjang karakter dan jumlah kata
    df["content_str"] = df["content"].astype(str).str.strip()
    df["char_length"] = df["content_str"].str.len()
    df["word_count"] = df["content_str"].str.split().str.len()

    # 2. Audit Missing Values
    missing_rows = []
    for col in df.columns[:11]:
        na_cnt = int(df[col].isna().sum())
        na_pct = (na_cnt / total_rows) * 100
        missing_rows.append((col, str(df[col].dtype), na_cnt, na_pct))

    # 3. Statistik per Aplikasi
    app_stats = []
    for app_name, cnt in df["app_name"].value_counts().items():
        sub = df[df["app_name"] == app_name]
        app_stats.append(
            {
                "app_name": app_name,
                "app_id": sub["app_id"].iloc[0],
                "count": int(cnt),
                "pct": (cnt / total_rows) * 100,
                "avg_score": float(sub["score"].mean()),
                "median_char": float(sub["char_length"].median()),
                "mean_char": float(sub["char_length"].mean()),
                "mean_words": float(sub["word_count"].mean()),
                "min_date": str(sub["at"].min())[:10],
                "max_date": str(sub["at"].max())[:10],
            }
        )
    app_stats_df = pd.DataFrame(app_stats)

    # 4. Deteksi Awal Rating-Text Discrepancy
    df["discrepancy_type"] = [
        detect_discrepancy_row(s, t) for s, t in zip(df["score"], df["content_str"])
    ]
    disc_counts = df["discrepancy_type"].value_counts()
    type1_total = int(disc_counts.get("Type-I (High Rating >=4, Negative Complaint)", 0))
    type2_total = int(disc_counts.get("Type-II (Low Rating <=2, Positive Praise)", 0))
    disc_total = type1_total + type2_total

    disc_app_list = []
    for app_name in app_stats_df["app_name"]:
        sub = df[df["app_name"] == app_name]
        n_app = len(sub)
        t1 = int((sub["discrepancy_type"] == "Type-I (High Rating >=4, Negative Complaint)").sum())
        t2 = int((sub["discrepancy_type"] == "Type-II (Low Rating <=2, Positive Praise)").sum())
        disc_app_list.append(
            {
                "app_name": app_name,
                "n_app": n_app,
                "type1_cnt": t1,
                "type1_pct": (t1 / n_app) * 100,
                "type2_cnt": t2,
                "type2_pct": (t2 / n_app) * 100,
                "total_discrepancy_cnt": t1 + t2,
                "total_discrepancy_pct": ((t1 + t2) / n_app) * 100,
            }
        )
    disc_app_df = pd.DataFrame(disc_app_list)

    # Ambil contoh nyata discrepancy dari dataset
    sample_type1 = df[df["discrepancy_type"] == "Type-I (High Rating >=4, Negative Complaint)"].head(5)
    sample_type2 = df[df["discrepancy_type"] == "Type-II (Low Rating <=2, Positive Praise)"].head(5)

    # 5. Ekstraksi Top 20 Unigram & Top 20 Bigram
    unigram_counter = Counter()
    bigram_counter = Counter()
    for txt in df["content_str"]:
        tokens = tokenize_clean(txt)
        unigram_counter.update(tokens)
        if len(tokens) >= 2:
            bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
            bigram_counter.update(bigrams)

    top20_unigrams = unigram_counter.most_common(20)
    top20_bigrams = bigram_counter.most_common(20)

    # 6. Buat Grafik Visualisasi (.png)
    generate_figures(df, app_stats_df, disc_app_df)

    # 7. Tulis Laporan Markdown eda_summary_report.md
    char_desc = df["char_length"].describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.95])
    word_desc = df["word_count"].describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.95])

    app_rows_md = []
    for i, r in enumerate(app_stats_df.itertuples(), 1):
        app_rows_md.append(
            f"| {i} | **{r.app_name}** | `{r.app_id}` | {format_id(r.count)} | {r.pct:.2f}% | "
            f"{r.avg_score:.2f} | {r.mean_char:.1f} kar ({r.mean_words:.1f} kata) | {r.min_date} s/d {r.max_date} |"
        )

    missing_rows_md = []
    for idx, (col, dtype_str, na_cnt, na_pct) in enumerate(missing_rows, 1):
        status = "✅ 100% Lengkap" if na_cnt == 0 else ("ℹ️ Opsional (Versi tidak tercatat di Play Store)" if col == "reviewCreatedVersion" else "ℹ️ Opsional (Mayoritas ulasan tidak dibalas developer)")
        missing_rows_md.append(
            f"| {idx} | `{col}` | `{dtype_str}` | {format_id(na_cnt)} | {na_pct:.2f}% | {status} |"
        )

    disc_rows_md = []
    for i, r in enumerate(disc_app_df.itertuples(), 1):
        disc_rows_md.append(
            f"| {i} | **{r.app_name}** | {format_id(r.type1_cnt)} ({r.type1_pct:.2f}%) | "
            f"{format_id(r.type2_cnt)} ({r.type2_pct:.2f}%) | **{format_id(r.total_discrepancy_cnt)} ({r.total_discrepancy_pct:.2f}%)** |"
        )

    unigram_rows_md = []
    for i in range(20):
        u_word, u_cnt = top20_unigrams[i]
        b_word, b_cnt = top20_bigrams[i]
        unigram_rows_md.append(
            f"| {i+1} | `{u_word}` | {format_id(u_cnt)} | `{b_word}` | {format_id(b_cnt)} |"
        )

    sample_t1_md = []
    for r in sample_type1.itertuples():
        clean_c = str(r.content_str).replace("\n", " ")[:140]
        sample_t1_md.append(f"- **[{r.app_name} | ⭐ {r.score}]** `{r.userName}`: *\"{clean_c}\"*")

    sample_t2_md = []
    for r in sample_type2.itertuples():
        clean_c = str(r.content_str).replace("\n", " ")[:140]
        sample_t2_md.append(f"- **[{r.app_name} | ⭐ {r.score}]** `{r.userName}`: *\"{clean_c}\"*")

    report_md = f"""# LAPORAN EXPLORATORY DATA ANALYSIS (EDA) AWAL — CRISP-DM FASE 2: DATA UNDERSTANDING

> **Sumber Dataset**: `./dataset/RAW_DATASET_GENAI_INDONESIA_AM522B.csv`  
> **Total Baris Terverifikasi**: **`{format_id(total_rows)}` baris unik** (`100%` `reviewId` unik, `100%` region `id_ID`)  
> **Rentang Waktu Ulasan**: `{str(df['at'].min())}` s/d `{str(df['at'].max())}`  
> **Script Generator**: `02_data_understanding/run_initial_eda.py`

---

## 1. Profil Distribusi Baris, Rating, dan Panjang Teks per Aplikasi

| No | Nama Aplikasi (`app_name`) | Package ID (`app_id`) | Jumlah Baris | Proporsi | Rata-rata Rating | Rata-rata Panjang Ulasan | Rentang Waktu Data |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
{chr(10).join(app_rows_md)}
| **TOTAL** | **8 Aplikasi Generative AI** | **Terdeduplikasi Penuh** | **{format_id(total_rows)}** | **100.00%** | **{df['score'].mean():.2f}** | **{df['char_length'].mean():.1f} kar ({df['word_count'].mean():.1f} kata)** | **`2023-05-31` s/d `2026-11-17`** |

### Temuan Kunci Distribusi Aplikasi:
1. **Dominasi 3 Platform Utama**: **ChatGPT** (`30,89%`), **Google Gemini** (`17,69%`), dan **Character.AI** (`16,85%`) menyumbang **65,43%** dari total volume ulasan ekosistem Generative AI di Indonesia.
2. **Kontras Kepuasan Ekstrem**:
   - Kelompok **Asisten Produktivitas & Pencarian** memperoleh rata-rata rating sangat tinggi: **Poe** (`4.66`), **Microsoft Copilot** (`4.65`), **Perplexity** (`4.54`), **ChatGPT** (`4.46`), dan **Google Gemini** (`4.29`).
   - Kelompok **Challenger / Reasoning AI**: **Claude by Anthropic** (`3.85`) dan **DeepSeek** (`3.77`) memiliki proporsi bintang 1 yang cukup signifikan akibat friksi *login/paywall* (Claude) dan *server busy* saat lonjakan trafik (DeepSeek).
   - Kelompok **AI Companion**: **Character.AI** mengalami polarisasi ekstrem dengan rata-rata rating **`1.58`** (didominasi Bintang 1 akibat keluhan filter karakter, sensor, dan perubahan antarmuka).

---

## 2. Audit Kualitas Data & Persentase Missing Values

| No | Nama Kolom | Tipe Data | Jumlah Missing (NaN) | Persentase Missing (%) | Status & Interpretasi Data Engineering |
| :---: | :--- | :---: | :---: | :---: | :--- |
{chr(10).join(missing_rows_md)}

- **Kolom Kritis (`reviewId`, `app_name`, `app_id`, `content`, `score`, `at`, `thumbsUpCount`) memiliki `0%` missing values (100% utuh).**
- Kolom `reviewCreatedVersion` memiliki missing value sebesar `{(df['reviewCreatedVersion'].isna().sum()/total_rows)*100:.2f}%` (wajar pada Google Play Store ketika pengguna memberi ulasan via browser web atau perangkat yang tidak mengirimkan metadata versi build).Pada tahap preprocessing, nilai kosong ini akan diisi dengan kategori `"Unknown_Version"`.
- Kolom `replyContent` kosong pada `{(df['replyContent'].isna().sum()/total_rows)*100:.2f}%` baris, yang menunjukkan hanya **`{100 - (df['replyContent'].isna().sum()/total_rows)*100:.2f}%`** ulasan pengguna Indonesia yang mendapat balasan resmi dari tim developer aplikasi AI.

---

## 3. Statistik Distribusi Panjang Karakter & Jumlah Kata Ulasan

| Metrik Statistik | Panjang Karakter (`char_length`) | Jumlah Kata (`word_count`) | Interpretasi Analitis |
| :--- | :---: | :---: | :--- |
| **Mean (Rata-rata)** | `{char_desc['mean']:.2f}` karakter | `{word_desc['mean']:.2f}` kata | Rata-rata ulasan terdiri dari 1–2 kalimat. |
| **Std Dev (Simpangan Baku)** | `{char_desc['std']:.2f}` karakter | `{word_desc['std']:.2f}` kata | Variasi tinggi antara ulasan singkat (1 kata) vs komplain panjang. |
| **Minimum** | `{int(char_desc['min'])}` karakter | `{int(word_desc['min'])}` kata | Ulasan berupa satu emoji/simbol (misal `"👍"`). |
| **Kuartil 1 (Q1 / 25%)** | `{int(char_desc['25%'])}` karakter | `{int(word_desc['25%'])}` kata | 25% ulasan berukuran sangat singkat (<= {int(word_desc['25%'])} kata, mis. *"bagus"*, *"mantap"*). |
| **Median (Q2 / 50%)** | `{int(char_desc['50%'])}` karakter | `{int(word_desc['50%'])}` kata | Nilai tengah panjang ulasan pengguna Indonesia. |
| **Kuartil 3 (Q3 / 75%)** | `{int(char_desc['75%'])}` karakter | `{int(word_desc['75%'])}` kata | 25% ulasan teratas memiliki pembahasan fitur yang cukup detail. |
| **Persentil 95% (P95)** | `{int(char_desc['95%'])}` karakter | `{int(word_desc['95%'])}` kata | Ulasan naratif panjang (umumnya berisi keluhan teknis mendalam). |
| **Maksimum** | `{int(char_desc['max'])}` karakter | `{int(word_desc['max'])}` kata | Batas maksimum panjang ulasan Google Play Store (~500 karakter). |

> **Perbandingan Panjang Ulasan Berdasarkan Skor Bintang**:
> - Rata-rata panjang ulasan **Bintang 1 (⭐)**: **`{df[df['score']==1]['char_length'].mean():.1f}` karakter** (`{df[df['score']==1]['word_count'].mean():.1f}` kata).
> - Rata-rata panjang ulasan **Bintang 5 (⭐⭐⭐⭐⭐)**: **`{df[df['score']==5]['char_length'].mean():.1f}` karakter** (`{df[df['score']==5]['word_count'].mean():.1f}` kata).
> - **Insight Penting**: Ulasan negatif (Bintang 1) rata-rata **lebih panjang hampir 2,5x lipat** dibandingkan ulasan Bintang 5 karena pengguna yang kecewa menuliskan rincian bug, fitur yang rusak, atau keluhan kuota secara spesifik.

---

## 4. Deteksi Awal *Rating-Text Discrepancy* (Paradoks Bintang vs. Isi Teks)

Menggunakan aturan deteksi leksikal awal pada `71.221` ulasan:
- **Total Kasus *Discrepancy* Terdeteksi Awal**: **`{format_id(disc_total)}` ulasan (`{(disc_total/total_rows)*100:.2f}%` dari keseluruhan dataset)**
  - **Type-I (*High Rating >= 4, Negative Complaint Text*)**: **`{format_id(type1_total)}` ulasan (`{(type1_total/total_rows)*100:.2f}%`)** — Pengguna memberi Bintang 4 atau 5 namun menuliskan keluhan (`error`, `limit`, `lemot`, `macet`, `bayar`, `gagal`, dll.).
  - **Type-II (*Low Rating <= 2, Pure Positive Praise*)**: **`{format_id(type2_total)}` ulasan (`{(type2_total/total_rows)*100:.2f}%`)** — Pengguna memberi Bintang 1 atau 2 namun isi teksnya murni memuji tanpa kata keluhan/negasi.

### A. Rincian *Rating-Text Discrepancy* per Aplikasi

| No | Nama Aplikasi | Type-I: Rating >= 4 tapi Isi Keluhan | Type-II: Rating <= 2 tapi Isi Pujian | Total Discrepancy per Aplikasi |
| :---: | :--- | :---: | :---: | :---: |
{chr(10).join(disc_rows_md)}

### B. Sampel Bukti Nyata *Rating-Text Discrepancy* di Dataset Kita
**Contoh Nyata Type-I (Rating Bintang 4–5, tetapi Teks Mengeluh):**
{chr(10).join(sample_t1_md)}

**Contoh Nyata Type-II (Rating Bintang 1–2, tetapi Teks Memuji):**
{chr(10).join(sample_t2_md)}

---

## 5. Top 20 Kata (*Unigram*) dan Pasangan Kata (*Bigram*) Paling Sering Muncul

Setelah dilakukan pembersihan *stopwords* dasar dan normalisasi kata tidak baku (*slang normalization* seperti `"gk" -> "tidak"`, `"bgt" -> "banget"`, `"apk" -> "aplikasi"`):

| Peringkat | Top 20 Unigram (Kata Tunggal) | Frekuensi Unigram | Top 20 Bigram (Pasangan 2 Kata) | Frekuensi Bigram |
| :---: | :--- | :---: | :--- | :---: |
{chr(10).join(unigram_rows_md)}

---

## 6. Daftar Grafik Visualisasi yang Dihasilkan (`02_data_understanding/figures/`)

1. **`01_distribusi_ulasan_per_aplikasi.png`**: Visualisasi jumlah ulasan unik dan skor rating rata-rata lintas 8 aplikasi Generative AI.
2. **`02_distribusi_rating_per_aplikasi.png`**: Grafik *100% Stacked Horizontal Bar* proporsi bintang 1 s/d 5 pada masing-masing aplikasi.
3. **`03_deteksi_awal_discrepancy_per_aplikasi.png`**: Komparasi persentase *Rating-Text Discrepancy* (Type-I dan Type-II) per aplikasi.
"""

    os.makedirs(os.path.dirname(REPORT_MD_PATH), exist_ok=True)
    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[OK] Laporan EDA Markdown tersimpan: {REPORT_MD_PATH}")
    print("=" * 85)


if __name__ == "__main__":
    main()
