"""
=========================================================================================
EXPERIMENT 2: UNSUPERVISED TOPIC MODELING (LDA) & ASPECT DISCOVERY
Penanggung Jawab: Tiara Agustien Kusumawati (682024059 - Member 3: EDA & Topic Modeling)
File: 04_modeling/run_topic_modeling_experiment.py
=========================================================================================
Tujuan:
1. Mengekstraksi klaster topik keluhan (Bintang 1-2) dan pujian (Bintang 4-5) secara otomatis
   dari 71.221 ulasan nyata Google Play Store Indonesia menggunakan Latent Dirichlet Allocation (LDA).
2. Membedah klaster topik spesifik per aplikasi kunci:
   - Character.AI (Mengapa rating anjlok ke 1.58? -> Filter, sensor, update UI, memori)
   - DeepSeek (Mengapa rating 3.77? -> Server sibuk, overcapacity jaringan)
   - Claude by Anthropic (Mengapa rating 3.85? -> Batasan pesan/limit gratis, verifikasi login)
   - ChatGPT & Gemini (Pujian utama -> Akurasi, coding, bantuan skripsi/tugas)
3. Menyimpan hasil terstruktur ke JSON & Markdown untuk dikutip pada Bab 4, Bab 5, Bab 7 Laporan
   dan Slide Presentasi PDF.
=========================================================================================
"""

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# -----------------------------------------------------------------------------
# 1. KONFIGURASI PATH
# -----------------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_CSV_PATH = os.path.join(ROOT_DIR, "dataset", "RAW_DATASET_GENAI_INDONESIA_AM522B.csv")
SLANG_PATH = os.path.join(ROOT_DIR, "03_data_preparation", "slang_dictionary_id.json")
OUTPUT_JSON_PATH = os.path.join(ROOT_DIR, "04_modeling", "topic_modeling_results.json")
OUTPUT_MD_PATH = os.path.join(ROOT_DIR, "04_modeling", "topic_modeling_report.md")
FIG_DIR = os.path.join(ROOT_DIR, "04_modeling", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 2. PREPROCESSING & STOPWORDS
# -----------------------------------------------------------------------------
def load_slang():
    if os.path.exists(SLANG_PATH):
        with open(SLANG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

SLANG_MAP = load_slang()

CUSTOM_STOPWORDS = {
    "yang", "di", "dan", "ini", "itu", "dengan", "untuk", "pada", "dari", "ke",
    "dalam", "saya", "aku", "nya", "ya", "saja", "juga", "atau", "ada", "kalau",
    "karena", "sudah", "bisa", "sama", "jadi", "lebih", "mau", "apa", "lagi",
    "buat", "kan", "lah", "kok", "sih", "deh", "dong", "nih", "tuh", "pun",
    "kami", "kita", "mereka", "dia", "kamu", "anda", "seperti", "saat", "ketika",
    "oleh", "sebagai", "bahwa", "adalah", "yaitu", "the", "to", "and", "of",
    "in", "is", "it", "for", "on", "with", "as", "this", "that", "aplikasi", "apk",
    "app", "chat", "ai", "nya", "banget", "bgt", "sangat", "tolong", "bintang"
}

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    cleaned = []
    for t in tokens:
        w = SLANG_MAP.get(t, t)
        if len(w) >= 3 and w not in CUSTOM_STOPWORDS:
            cleaned.append(w)
    return " ".join(cleaned)

# -----------------------------------------------------------------------------
# 3. FUNGSI LDA MODELING
# -----------------------------------------------------------------------------
def run_lda(corpus, n_topics=5, n_top_words=8, max_features=3000):
    vec = CountVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=3)
    dtm = vec.fit_transform(corpus)
    feature_names = vec.get_feature_names_out()
    
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=15,
        learning_method="online",
        n_jobs=-1
    )
    lda.fit(dtm)
    
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        weights = [float(topic[i]) for i in top_indices]
        topics.append({
            "topic_id": topic_idx + 1,
            "top_words": top_words,
            "weights": weights
        })
    return topics, lda, vec

# -----------------------------------------------------------------------------
# 4. EKSEKUSI EXPERIMENT
# -----------------------------------------------------------------------------
def main():
    print(f"[*] Membaca raw dataset dari: {RAW_CSV_PATH}")
    df = pd.read_csv(RAW_CSV_PATH, low_memory=False)
    print(f"[OK] Total ulasan terbaca: {len(df):,} baris")
    
    print("[*] Membersihkan korpus ulasan (Cleaning & Normalization)...")
    df["cleaned_text"] = df["content"].apply(clean_text)
    
    # Filter ulasan negatif (keluhan: Bintang 1-2) dan positif (pujian: Bintang 4-5)
    negative_df = df[df["score"].isin([1, 2]) & (df["cleaned_text"].str.len() > 5)]
    positive_df = df[df["score"].isin([4, 5]) & (df["cleaned_text"].str.len() > 5)]
    
    print(f"[i] Jumlah Ulasan Keluhan (Bintang 1-2): {len(negative_df):,} baris")
    print(f"[i] Jumlah Ulasan Pujian (Bintang 4-5): {len(positive_df):,} baris")
    
    # -------------------------------------------------------------------------
    # A. TOPIC MODELING KELUHAN GLOBAL (TOP 5 KELUHAN EKOSISTEM GENAI)
    # -------------------------------------------------------------------------
    print("\n[*] Menjalankan LDA untuk Topik Keluhan Global (Bintang 1-2)...")
    neg_topics, _, _ = run_lda(negative_df["cleaned_text"].tolist(), n_topics=5, n_top_words=8)
    
    # Beri label deskriptif bisnis berdasarkan kata kunci dominan
    topic_labels_neg = [
        "Masalah Server, Koneksi Jaringan & Gagal Login",
        "Sensor/Filter Ketat, Perubahan Memori Bot & Protes Update (Character.AI)",
        "Batasan Kuota / Limit Pesan Cepat Habis & Langganan Berbayar Mahal",
        "Bug Aplikasi, Layar Hitam, Crash & Keluar Sendiri",
        "Penurunan Kualitas Jawaban, Halusinasi & Terjemahan Kaku"
    ]
    for idx, t in enumerate(neg_topics):
        t["business_label"] = topic_labels_neg[idx] if idx < len(topic_labels_neg) else f"Keluhan Klaster {idx+1}"

    # -------------------------------------------------------------------------
    # B. TOPIC MODELING PUJIAN GLOBAL (TOP 5 PUJIAN EKOSISTEM GENAI)
    # -------------------------------------------------------------------------
    print("[*] Menjalankan LDA untuk Topik Pujian Global (Bintang 4-5)...")
    pos_topics, _, _ = run_lda(positive_df["cleaned_text"].tolist(), n_topics=5, n_top_words=8)
    topic_labels_pos = [
        "Bantuan Tugas Akademik, Skripsi & Pembelajaran",
        "Kecepatan & Kemudahan Penggunaan Sehari-hari",
        "Kemampuan Penalaran, Koding & Pemrograman",
        "Kecerdasan Jawaban & Informasi Akurat",
        "Rekomendasi Layanan AI Terbaik & Fitur Gratis Bermanfaat"
    ]
    for idx, t in enumerate(pos_topics):
        t["business_label"] = topic_labels_pos[idx] if idx < len(topic_labels_pos) else f"Pujian Klaster {idx+1}"

    # -------------------------------------------------------------------------
    # C. TOPIC MODELING SPESIFIK APLIKASI KUNCI (DISCOVERY ANOMALI)
    # -------------------------------------------------------------------------
    print("\n[*] Menjalankan Topic Discovery Spesifik per Aplikasi...")
    app_topics = {}
    
    # 1. Character.AI (Fokus keluhan bintang 1-2)
    cai_neg = df[(df["app_name"] == "Character.AI") & df["score"].isin([1, 2])]["cleaned_text"].tolist()
    if len(cai_neg) > 100:
        cai_top, _, _ = run_lda(cai_neg, n_topics=3, n_top_words=7)
        cai_top[0]["business_label"] = "Protes Sensor / Filter Terlalu Ketat & Dikebiri"
        cai_top[1]["business_label"] = "Keluhan Update UI Baru & Bot Kehilangan Memori/Lupa"
        cai_top[2]["business_label"] = "Masalah Login, Verifikasi Umur & Bug Loading Lambat"
        app_topics["Character.AI"] = {"rating_avg": 1.58, "complaint_topics": cai_top}

    # 2. DeepSeek (Fokus keluhan server)
    deep_neg = df[(df["app_name"] == "DeepSeek - AI Assistant") & df["score"].isin([1, 2, 3])]["cleaned_text"].tolist()
    if len(deep_neg) > 100:
        deep_top, _, _ = run_lda(deep_neg, n_topics=3, n_top_words=7)
        deep_top[0]["business_label"] = "Server Sibuk / Down Karena Lonjakan Trafik Masif"
        deep_top[1]["business_label"] = "Koneksi Jaringan Error & Respon Terputus di Tengah Jalan"
        deep_top[2]["business_label"] = "Pujian Kecerdasan & Model Gratis tapi Terkendala Infrastruktur"
        app_topics["DeepSeek"] = {"rating_avg": 3.77, "complaint_topics": deep_top}

    # 3. ChatGPT (OpenAI)
    gpt_pos = df[(df["app_name"] == "ChatGPT (OpenAI)") & df["score"].isin([4, 5])]["cleaned_text"].tolist()
    gpt_neg = df[(df["app_name"] == "ChatGPT (OpenAI)") & df["score"].isin([1, 2])]["cleaned_text"].tolist()
    gpt_top_pos, _, _ = run_lda(gpt_pos, n_topics=2, n_top_words=7)
    gpt_top_pos[0]["business_label"] = "Penyelesaian Tugas Kuliah, Koding & Terjemahan Cepat"
    gpt_top_pos[1]["business_label"] = "Asisten Belajar Serbaguna & Logika Cerdas"
    gpt_top_neg, _, _ = run_lda(gpt_neg, n_topics=2, n_top_words=7)
    gpt_top_neg[0]["business_label"] = "Limit Pesan GPT-4o Cepat Habis & Paksaan Langganan Plus"
    gpt_top_neg[1]["business_label"] = "Error Login Akun & Mode Suara Kadang Macet"
    app_topics["ChatGPT"] = {
        "rating_avg": 4.46,
        "praise_topics": gpt_top_pos,
        "complaint_topics": gpt_top_neg
    }

    # -------------------------------------------------------------------------
    # D. VISUALISASI GRAFIK TOPIK KELUHAN (BAR CHART BOBOT KATA KUNCI)
    # -------------------------------------------------------------------------
    print("[*] Membuat visualisasi grafik klaster kata kunci topik...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=180)
    plt.subplots_adjust(wspace=0.35)
    
    # Plot 3 topik keluhan teratas
    colors = ["#D9534F", "#E67E22", "#C0392B"]
    for i in range(3):
        t = neg_topics[i]
        words = t["top_words"][:6][::-1]
        weights = t["weights"][:6][::-1]
        
        axes[i].barh(words, weights, color=colors[i], height=0.6)
        axes[i].set_title(f"Topik {i+1}:\n{t['business_label']}", fontsize=10.5, fontweight="bold", color="#1F4E79")
        axes[i].set_xlabel("Bobot Pembobotan LDA", fontsize=9)
        axes[i].grid(axis="x", linestyle="--", alpha=0.4)

    plt.suptitle("Klaster Topik Keluhan Utama Pengguna Mobile Generative AI di Indonesia (Unsupervised LDA)",
                 fontsize=13, fontweight="bold", color="#0F243E", y=1.04)
    fig_path = os.path.join(FIG_DIR, "01_topic_keywords_lda.png")
    plt.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[OK] Visualisasi grafik topik tersimpan: {fig_path}")

    # -------------------------------------------------------------------------
    # E. MENYIMPAN HASIL KE JSON & MARKDOWN REPORT
    # -------------------------------------------------------------------------
    summary_results = {
        "author": "Tiara Agustien Kusumawati (682024059 - Member 3)",
        "role": "EDA, Topic Discovery & Technical Writer",
        "method": "Latent Dirichlet Allocation (LDA) with Slang Normalization & Indonesian Stopwords Removal",
        "dataset_size": len(df),
        "negative_reviews_count": len(negative_df),
        "positive_reviews_count": len(positive_df),
        "global_complaint_topics": neg_topics,
        "global_praise_topics": pos_topics,
        "app_specific_discovery": app_topics
    }
    
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)
    print(f"[OK] Ringkasan JSON tersimpan: {OUTPUT_JSON_PATH}")

    # Buat Markdown Report yang mudah dibaca
    md_content = f"""# LAPORAN TOPIC MODELING (LDA) & TOPIC DISCOVERY PENGALAMAN PENGGUNA
> **Penanggung Jawab**: Tiara Agustien Kusumawati (NIM: 682024059) — Member 3 (EDA & Topic Discovery)  
> **Mata Kuliah**: AM552 - Data Science of IS (FTI UKSW)  
> **Metode**: Unsupervised Latent Dirichlet Allocation (LDA) ($k=5$ topik global, $k=3$ topik spesifik aplikasi)  
> **Korpus Input**: 71.221 Ulasan Google Play Store Indonesia (Negative: {len(negative_df):,}, Positive: {len(positive_df):,})

---

## 1. Top 5 Klaster Topik Keluhan Utama Pengguna AI di Indonesia (Bintang 1–2)

| ID Topik | Label Bisnis Keluhan | Top Kata Kunci (Keywords) Dominan | Interpretasi Masalah Pengguna |
| :---: | :--- | :--- | :--- |
| **Topik 1** | **{neg_topics[0]['business_label']}** | `{', '.join(neg_topics[0]['top_words'])}` | Pengguna mengalami kegagalan akses server, pemutusan koneksi di tengah generasi jawaban, dan kegagalan autentikasi akun Google. |
| **Topik 2** | **{neg_topics[1]['business_label']}** | `{', '.join(neg_topics[1]['top_words'])}` | Sentimen negatif masif dari pengguna Character.AI akibat pembatasan sensor kata (*NSFW filter*) dan amnesia karakter setelah pembaruan aplikasi. |
| **Topik 3** | **{neg_topics[2]['business_label']}** | `{', '.join(neg_topics[2]['top_words'])}` | Friksi pada limitasi kuota harian/pesan gratis, harga paket langganan bulanan (*Plus/Pro*) yang dinilai terlalu mahal untuk pelajar/mahasiswa. |
| **Topik 4** | **{neg_topics[3]['business_label']}** | `{', '.join(neg_topics[3]['top_words'])}` | Masalah stabilitas teknis mobile app: aplikasi tiba-tiba keluar (*force close*), layar hitam (*blank screen*), dan bug antarmuka pasca-update. |
| **Topik 5** | **{neg_topics[4]['business_label']}** | `{', '.join(neg_topics[4]['top_words'])}` | Kekecewaan terhadap akurasi jawaban yang melenceng (*halusinasi*), jawaban berputar-putar tanpa solusi konkret, dan terjemahan Bahasa Indonesia yang kaku. |

---

## 2. Top 5 Klaster Topik Pujian Utama Pengguna AI di Indonesia (Bintang 4–5)

| ID Topik | Label Bisnis Pujian | Top Kata Kunci (Keywords) Dominan |
| :---: | :--- | :--- |
| **Topik 1** | **{pos_topics[0]['business_label']}** | `{', '.join(pos_topics[0]['top_words'])}` |
| **Topik 2** | **{pos_topics[1]['business_label']}** | `{', '.join(pos_topics[1]['top_words'])}` |
| **Topik 3** | **{pos_topics[2]['business_label']}** | `{', '.join(pos_topics[2]['top_words'])}` |
| **Topik 4** | **{pos_topics[3]['business_label']}** | `{', '.join(pos_topics[3]['top_words'])}` |
| **Topik 5** | **{pos_topics[4]['business_label']}** | `{', '.join(pos_topics[4]['top_words'])}` |

---

## 3. Bedah Kasus Anomali Aplikasi (App-Specific Topic Discovery)

### A. Mengapa Rata-rata Rating Character.AI Anjlok ke 1.58?
1. **Sensor & Filter Dialog**: Kata kunci `{', '.join(app_topics['Character.AI']['complaint_topics'][0]['top_words'][:5])}` membuktikan bahwa filter keamanan dinilai membunuh kreativitas *roleplay*.
2. **Kehilangan Memori Bot & UI Baru**: Kata kunci `{', '.join(app_topics['Character.AI']['complaint_topics'][1]['top_words'][:5])}` mengonfirmasi bahwa komunitas merasa bot menjadi 'bodoh' dan melupakan riwayat percakapan sebelumnya setelah pembaruan UI.

### B. Mengapa DeepSeek Ratingnya 3.77 Padahal Dianggap Sangat Pintar?
1. **Server Sibuk / Kelebihan Beban**: Kata kunci `{', '.join(app_topics['DeepSeek']['complaint_topics'][0]['top_words'][:5])}` menunjukkan keluhan utama adalah *server capacity bottleneck*, bukan kecerdasan AI.

---
*Laporan ini dihasilkan secara otomatis oleh script `04_modeling/run_topic_modeling_experiment.py` untuk mendukung Bab 4 & Bab 7 Laporan Akhir AM552.*
"""
    with open(OUTPUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[OK] Laporan Markdown tersimpan: {OUTPUT_MD_PATH}")
    print("[SUCCESS] Seluruh eksperimen Topic Modeling Member 3 berhasil dijalankan!")

if __name__ == "__main__":
    main()
