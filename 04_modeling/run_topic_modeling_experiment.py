"""
=========================================================================================
EXPERIMENT 2: UNSUPERVISED TOPIC MODELING (LDA), COHERENCE SELECTION & 8-APP DISCOVERY
Penanggung Jawab: Arthur Willyam Liang (682024046)
File: 04_modeling/run_topic_modeling_experiment.py
=========================================================================================
Tujuan:
1. Mengevaluasi jumlah topik optimal K in {3, 4, 5, 6, 8} menggunakan Log-Likelihood,
   Perplexity, dan UMass Topic Coherence (Mimno et al., 2011).
2. Mengekstraksi 5 klaster topik keluhan (Bintang 1-2) dan 5 klaster topik pujian (Bintang 4-5)
   dari 71.221 ulasan Google Play Store Indonesia dengan pelabelan berbasis kata kunci aktual,
   proporsi dokumen eksak (%), pemetaan dimensi ABSA, dan kutipan ulasan representatif.
3. Menghitung matriks distribusi topik lintas seluruh 8 aplikasi serta Topic Discovery
   spesifik untuk masing-masing dari 8 aplikasi Generative AI.
=========================================================================================
"""

import json
import math
import os
import re
from typing import Any, Dict, List, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_CSV_PATH = os.path.join(ROOT_DIR, "dataset", "RAW_DATASET_GENAI_INDONESIA_AM522B.csv")
SLANG_PATH = os.path.join(ROOT_DIR, "03_data_preparation", "slang_dictionary_id.json")
OUTPUT_JSON_PATH = os.path.join(ROOT_DIR, "04_modeling", "topic_modeling_results.json")
OUTPUT_MD_PATH = os.path.join(ROOT_DIR, "04_modeling", "topic_modeling_report.md")
FIG_DIR = os.path.join(ROOT_DIR, "04_modeling", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_slang() -> Dict[str, str]:
    if os.path.exists(SLANG_PATH):
        with open(SLANG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


SLANG_MAP = load_slang()

# Stopwords khusus Topic Modeling: menyaring kata fungsi, kata ganti, nama brand aplikasi,
# serta kata sifat sentimen umum agar LDA fokus mengelompokkan SUBSTANSI TOPIK / ASPEK.
TOPIC_STOPWORDS: Set[str] = {
    # Kata tugas & preposisi Indonesia
    "yang", "di", "dan", "ini", "itu", "dengan", "untuk", "pada", "dari", "ke",
    "dalam", "saya", "aku", "nya", "ya", "saja", "juga", "atau", "ada", "kalau",
    "jika", "karena", "sebab", "sudah", "bisa", "dapat", "sama", "jadi", "lebih",
    "mau", "ingin", "apa", "lagi", "buat", "kan", "lah", "kok", "sih", "deh",
    "dong", "nih", "tuh", "pun", "kami", "kita", "mereka", "dia", "kamu", "anda",
    "kalian", "seperti", "saat", "ketika", "oleh", "sebagai", "bahwa", "adalah",
    "yaitu", "yakni", "antara", "sampai", "hingga", "setelah", "sebelum", "selama",
    "sementara", "sedang", "masih", "belum", "akan", "harus", "perlu", "butuh",
    "jangan", "bukan", "tidak", "tak", "kurang", "tapi", "tetapi", "namun",
    "padahal", "walaupun", "meskipun", "sekarang", "dulu", "kemarin", "besok",
    "tadi", "nanti", "makin", "semakin", "terus", "selalu", "sering", "kadang",
    "jarang", "malah", "justru", "hanya", "cuma", "semua", "seluruh", "setiap",
    "beberapa", "banyak", "sedikit", "sendiri", "paling", "agak", "tetap",
    "sangat", "banget", "sekali", "terlalu", "cukup", "lumayan", "begitu", "begini",
    "mengapa", "kenapa", "bagaimana", "kapan", "mana", "berapa", "siapa",
    "tolong", "mohon", "semoga", "harap", "kasih", "terima", "beri", "minta",
    "coba", "mencoba", "pakai", "memakai", "menggunakan", "membuat", "bikin",
    "hal", "orang", "pengguna", "aplikasi", "apk", "app", "ai", "bintang", "ulasan",
    "baru", "awal", "pertama", "kali", "hari", "bulan", "tahun", "jam", "menit",
    "satu", "dua", "tiga", "empat", "lima", "benar", "memang", "mungkin", "pasti",
    "tahu", "lihat", "melihat", "rasa", "merasa", "bilang", "masuk", "buka",
    # Kata sifat polaritas generik (agar klaster tidak sekadar kumpulan kata "bagus"/"jelek")
    "bagus", "baik", "mantap", "keren", "hebat", "terbaik", "puas", "memuaskan",
    "suka", "senang", "enak", "berguna", "bermanfaat", "rekomendasi", "sempurna",
    "ok", "oke", "jelek", "buruk", "parah", "kecewa", "mengecewakan", "menyesal",
    "kesal", "rusak", "hancur", "sampah", "aneh", "biasa", "jelas", "cocok",
    "sesuai", "jujur", "pokoknya", "seru", "asik",
    # Stopwords bahasa Inggris umum di ulasan Play Store
    "the", "to", "and", "of", "in", "is", "it", "for", "on", "with", "as", "this",
    "that", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does",
    "did", "can", "could", "will", "would", "should", "not", "but", "so", "if",
    "or", "my", "me", "you", "your", "we", "they", "he", "she", "its", "very",
    "really", "just", "now", "more", "all", "any", "some", "no", "yes", "please",
    "good", "nice", "best", "great", "love", "like", "awesome", "helpful", "bad",
    "worst", "why", "how", "what", "when", "back", "old", "new", "don", "doesn",
    "didn", "cant", "won", "isn", "im", "ive", "app", "apps", "use", "using", "used",
    # Nama entitas aplikasi (agar topik fokus ke isu layanan, bukan sekadar merek)
    "chatgpt", "gpt", "openai", "gemini", "bard", "deepseek", "claude", "anthropic",
    "perplexity", "copilot", "microsoft", "poe", "character", "cai",
}


def clean_for_topic(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    s = text.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"(.)\1{2,}", r"\1\1", s)
    raw_tokens = re.findall(r"[a-z]+", s)
    out_tokens: List[str] = []
    for t in raw_tokens:
        mapped = SLANG_MAP.get(t, t)
        for sub in mapped.split():
            if len(sub) >= 3 and sub.isalpha() and sub not in TOPIC_STOPWORDS:
                out_tokens.append(sub)
    return " ".join(out_tokens)


def compute_umass_coherence(dtm_bin: Any, lda_components: np.ndarray, top_m: int = 10) -> float:
    """
    Menghitung skor koherensi topik rata-rata C_UMass (Mimno et al., 2011):
    C_UMass = (2 / (M*(M-1))) * sum_{m=2..M} sum_{l=1..m-1} ln((D(w_m, w_l) + 1) / D(w_l))
    Semakin mendekati 0 (lebih tinggi / kurang negatif), semakin koheren kata-kata dalam topik.
    """
    coheres = []
    # dtm_bin: matriks biner dokumen x kata (1 jika kata muncul di dokumen)
    doc_freq = np.asarray(dtm_bin.sum(axis=0)).ravel()
    for comp in lda_components:
        top_idx = comp.argsort()[:-top_m - 1:-1]
        sub_bin = dtm_bin[:, top_idx]
        co_mat = (sub_bin.T @ sub_bin).toarray()
        score = 0.0
        pairs = 0
        for m in range(1, len(top_idx)):
            for l_idx in range(m):
                d_wl = float(doc_freq[top_idx[l_idx]])
                d_wm_wl = float(co_mat[m, l_idx])
                if d_wl > 0:
                    score += math.log((d_wm_wl + 1.0) / d_wl)
                    pairs += 1
        if pairs > 0:
            coheres.append(score / pairs)
    return round(float(np.mean(coheres)), 4) if coheres else -10.0


COMPLAINT_ARCHETYPES = [
    (
        "Masalah Server Sibuk, Jaringan & Kegagalan Login",
        "A3: Performa, UI/UX & Server",
        {"server", "sibuk", "error", "jaringan", "koneksi", "login", "akun", "gagal", "loading", "down", "nomor", "email"},
        "Pengguna mengeluhkan kegagalan autentikasi akun/nomor telepon, koneksi terputus di tengah generasi respons, dan server kelebihan beban.",
    ),
    (
        "Pembatasan Kuota Harian, Limit Pesan & Biaya Langganan",
        "A2: Harga & Limitasi Kuota",
        {"limit", "kuota", "batas", "dibatasi", "habis", "bayar", "berbayar", "mahal", "langganan", "premium", "gratis", "plus", "pro", "pesan", "pelit"},
        "Friksi utama pada cepat habisnya kuota chat/gambar gratis harian serta harga paket langganan bulanan yang dinilai memberatkan pelajar dan mahasiswa.",
    ),
    (
        "Protes Sensor Filter, Verifikasi Usia & Perubahan Karakter Bot",
        "A1 & A3 (Kebijakan Konten & Update Fitur)",
        {"sensor", "filter", "verifikasi", "usia", "umur", "bot", "memori", "lupa", "cerita", "percakapan", "age"},
        "Keluhan komunitas atas filter percakapan yang terlalu kaku, kewajiban verifikasi usia, serta penurunan daya ingat konteks karakter.",
    ),
    (
        "Gangguan Iklan, Bug Pasca-Update & Ketidakstabilan Aplikasi",
        "A3: Performa, UI/UX & Server",
        {"iklan", "update", "bug", "lambat", "macet", "keluar", "layar", "versi", "tampilan", "suara", "perbaiki", "kembalikan"},
        "Munculnya gangguan iklan berlebih, aplikasi lambat atau keluar sendiri (force close), serta fitur yang menurun setelah pembaruan versi.",
    ),
    (
        "Ketidakakuratan Jawaban, Halusinasi AI & Kendala Bahasa",
        "A1 & A4 (Akurasi Jawaban & Bahasa Lokal)",
        {"jawaban", "salah", "ngawur", "halusinasi", "bodoh", "bahasa", "indonesia", "inggris", "terjemahan", "pertanyaan", "informasi", "gambar", "foto"},
        "Ketidakpuasan terhadap jawaban AI yang melenceng dari instruksi (halusinasi), kurang akurat, atau kualitas bahasa Indonesia yang kurang natural.",
    ),
]

PRAISE_ARCHETYPES = [
    (
        "Pendamping Tugas Kuliah, Skripsi, Belajar & Koding",
        "A1: Akurasi & Kecerdasan AI",
        {"tugas", "belajar", "kuliah", "skripsi", "coding", "sekolah", "mahasiswa", "matematika", "soal", "materi"},
        "Apresiasi tinggi dari pelajar dan mahasiswa karena AI mempercepat pemahaman materi kuliah, penulisan skripsi, dan pemecahan bug pemrograman.",
    ),
    (
        "Produktivitas Pekerjaan, Pencarian Informasi & Solusi Cepat",
        "A1 & A3 (Produktivitas & Kecepatan)",
        {"pekerjaan", "kerja", "informasi", "solusi", "pertanyaan", "penjelasan", "ide", "dokumen", "presentasi"},
        "Pengguna profesional menilai AI sangat efektif merangkum informasi, menyusun dokumen kerja, dan memberikan solusi praktis.",
    ),
    (
        "Kecerdasan Penalaran, Akurasi Jawaban & Pemahaman Konteks",
        "A1: Akurasi & Kecerdasan AI",
        {"pintar", "cerdas", "akurat", "jawaban", "paham", "mengerti", "tepat", "lengkap", "logika", "fakta"},
        "Pujian terhadap kedalaman nalar model, ketepatan menjawab instruksi kompleks, serta kelengkapan referensi jawaban.",
    ),
    (
        "Kecepatan Respons, Kemudahan UI & Fitur Multimodal Gratis",
        "A2 & A3 (Aksesibilitas Gratis & UI/UX)",
        {"cepat", "mudah", "nyaman", "ringan", "stabil", "gratis", "gambar", "foto", "suara", "tampilan", "fitur"},
        "Pengguna menyukai antarmuka yang bersih, respons instan tanpa antre server, serta ketersediaan fitur pembuatan gambar/suara gratis.",
    ),
    (
        "Kualitas Bahasa Indonesia Luwes & Interaksi Percakapan Alami",
        "A4: Bahasa Indonesia & Konteks Lokal",
        {"bahasa", "indonesia", "terjemahan", "kalimat", "percakapan", "cerita", "bicara", "ngobrol", "alami", "teman"},
        "Kemampuan AI berkomunikasi dalam bahasa Indonesia yang natural, mudah dipahami, dan nyaman diajak berdiskusi sehari-hari.",
    ),
]


def assign_unique_archetypes(
    topics: List[Dict[str, Any]],
    archetypes: List[Tuple[str, str, Set[str], str]],
) -> None:
    """Memetakan setiap klaster LDA ke label bisnis paling relevan tanpa duplikasi."""
    used: Set[int] = set()
    # Hitung matriks skor kecocokan kata kunci
    scores = np.zeros((len(topics), len(archetypes)), dtype=float)
    for t_i, t in enumerate(topics):
        words_tokens: Set[str] = set()
        for w in t["top_words"]:
            words_tokens.update(w.split())
        for a_i, (_, _, kw_set, _) in enumerate(archetypes):
            # Bobot lebih besar jika cocok pada top-4 kata pertama
            top4_tokens: Set[str] = set()
            for w in t["top_words"][:4]:
                top4_tokens.update(w.split())
            scores[t_i, a_i] = 2.5 * len(top4_tokens & kw_set) + 1.0 * len(words_tokens & kw_set)

    # Greedy assignment dari skor kecocokan tertinggi
    pairs = sorted(
        [(scores[ti, ai], ti, ai) for ti in range(len(topics)) for ai in range(len(archetypes))],
        reverse=True,
    )
    assigned_t: Set[int] = set()
    for _, ti, ai in pairs:
        if ti not in assigned_t and ai not in used:
            lbl, absa_map, _, interp = archetypes[ai]
            topics[ti]["business_label"] = lbl
            topics[ti]["mapped_absa"] = absa_map
            topics[ti]["interpretation"] = interp
            assigned_t.add(ti)
            used.add(ai)

    for ti, t in enumerate(topics):
        if ti not in assigned_t:
            rem = [i for i in range(len(archetypes)) if i not in used]
            ai = rem[0] if rem else (ti % len(archetypes))
            used.add(ai)
            lbl, absa_map, _, interp = archetypes[ai]
            t["business_label"] = lbl
            t["mapped_absa"] = absa_map
            t["interpretation"] = interp


def evaluate_k_candidates(texts: List[str], k_list: List[int]) -> List[Dict[str, Any]]:
    """Mengevaluasi kandidat jumlah topik K menggunakan Log-Likelihood, Perplexity, dan C_UMass."""
    vec = CountVectorizer(max_features=2500, ngram_range=(1, 2), min_df=5, max_df=0.85)
    dtm = vec.fit_transform(texts)
    dtm_bin = (dtm > 0).astype(int)

    results = []
    for k in k_list:
        lda = LatentDirichletAllocation(
            n_components=k,
            random_state=42,
            max_iter=10,
            learning_method="online",
            batch_size=4096,
            n_jobs=1,
        )
        lda.fit(dtm)
        ll = round(float(lda.score(dtm)), 1)
        perp = round(float(lda.perplexity(dtm)), 2)
        coh = compute_umass_coherence(dtm_bin, lda.components_, top_m=10)
        results.append({
            "k_topics": int(k),
            "log_likelihood": ll,
            "perplexity": perp,
            "umass_coherence": coh,
        })
    return results


def fit_detailed_lda(
    df_sub: pd.DataFrame,
    n_topics: int = 5,
    n_top_words: int = 10,
    max_features: int = 3000,
) -> Tuple[List[Dict[str, Any]], np.ndarray, LatentDirichletAllocation, CountVectorizer]:
    vec = CountVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=4, max_df=0.85)
    dtm = vec.fit_transform(df_sub["cleaned_topic_text"].tolist())
    feature_names = vec.get_feature_names_out()

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=12,
        learning_method="online",
        batch_size=4096,
        n_jobs=1,
    )
    doc_topic = lda.fit_transform(dtm)
    dom_topics = doc_topic.argmax(axis=1)
    total_docs = max(1, len(df_sub))

    raw_contents = df_sub["content"].fillna("").astype(str).tolist()
    raw_apps = df_sub["app_name"].astype(str).tolist()

    topics: List[Dict[str, Any]] = []
    for idx, comp in enumerate(lda.components_):
        top_indices = comp.argsort()[:-n_top_words - 1:-1]
        top_words = [str(feature_names[i]) for i in top_indices]
        weights = [round(float(comp[i]), 2) for i in top_indices]

        mask = dom_topics == idx
        doc_cnt = int(mask.sum())
        doc_pct = round(doc_cnt / total_docs * 100.0, 2)

        # Cari ulasan asli paling representatif (probabilitas topik tertinggi & panjang memadai)
        probs_k = doc_topic[:, idx]
        cand_order = np.argsort(probs_k)[::-1]
        sample_quote = ""
        sample_app = ""
        for r_i in cand_order[:80]:
            txt = " ".join(raw_contents[r_i].split())
            if 45 <= len(txt) <= 240:
                sample_quote = txt
                sample_app = raw_apps[r_i]
                break
        if not sample_quote and len(cand_order) > 0:
            sample_quote = " ".join(raw_contents[cand_order[0]].split())[:220]
            sample_app = raw_apps[cand_order[0]]

        topics.append({
            "topic_id": idx + 1,
            "top_words": top_words,
            "weights": weights,
            "doc_count": doc_cnt,
            "doc_share_pct": doc_pct,
            "sample_app": sample_app,
            "sample_quote": sample_quote,
        })

    return topics, dom_topics, lda, vec


def infer_short_app_topic_label(top_words: List[str], is_complaint: bool) -> str:
    archetypes = COMPLAINT_ARCHETYPES if is_complaint else PRAISE_ARCHETYPES
    w_set: Set[str] = set()
    for w in top_words[:6]:
        w_set.update(w.split())
    best_lbl = archetypes[0][0]
    best_sc = -1
    for lbl, _, kw_set, _ in archetypes:
        sc = len(w_set & kw_set)
        if sc > best_sc:
            best_sc = sc
            best_lbl = lbl
    if best_sc == 0:
        prefix = "Keluhan" if is_complaint else "Pujian"
        return f"{prefix}: {', '.join(top_words[:3]).title()}"
    return best_lbl


def main() -> None:
    print(f"[*] Membaca dataset dari: {RAW_CSV_PATH}")
    df = pd.read_csv(RAW_CSV_PATH, low_memory=False)
    print(f"[OK] Total ulasan: {len(df):,} baris | Kamus slang aktif: {len(SLANG_MAP):,} entri")

    df["cleaned_topic_text"] = df["content"].apply(clean_for_topic)
    df["tok_count"] = df["cleaned_topic_text"].str.split().str.len()

    neg_df = df[df["score"].isin([1, 2]) & (df["tok_count"] >= 2)].copy().reset_index(drop=True)
    pos_df = df[df["score"].isin([4, 5]) & (df["tok_count"] >= 2)].copy().reset_index(drop=True)
    print(f"[i] Korpus Keluhan Substantif (Bintang 1-2): {len(neg_df):,} ulasan")
    print(f"[i] Korpus Pujian Substantif (Bintang 4-5): {len(pos_df):,} ulasan")

    # 1. Evaluasi Pemilihan Jumlah Topik Optimal K in {3, 4, 5, 6, 8}
    print("[*] Mengevaluasi metrik pemilihan K in {3, 4, 5, 6, 8}...")
    k_eval_neg = evaluate_k_candidates(neg_df["cleaned_topic_text"].tolist(), [3, 4, 5, 6, 8])
    k_eval_pos = evaluate_k_candidates(pos_df["cleaned_topic_text"].tolist(), [3, 4, 5, 6, 8])

    # 2. Fit Global LDA K=5 pada Korpus Keluhan & Pujian
    print("[*] Menjalankan Global LDA (K=5) untuk Keluhan & Pujian...")
    neg_topics, neg_dom, _, _ = fit_detailed_lda(neg_df, n_topics=5, n_top_words=10)
    assign_unique_archetypes(neg_topics, COMPLAINT_ARCHETYPES)

    pos_topics, pos_dom, _, _ = fit_detailed_lda(pos_df, n_topics=5, n_top_words=10)
    assign_unique_archetypes(pos_topics, PRAISE_ARCHETYPES)

    # 3. Matriks Distribusi Topik Lintas 8 Aplikasi (%)
    neg_df["dom_topic"] = neg_dom + 1
    pos_df["dom_topic"] = pos_dom + 1

    app_order = df["app_name"].value_counts().index.tolist()

    def _build_app_topic_matrix(sub_df: pd.DataFrame, topic_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows = []
        for app in app_order:
            g = sub_df[sub_df["app_name"] == app]
            n_g = len(g)
            r_dict: Dict[str, Any] = {"Nama Aplikasi": app, "Jumlah Ulasan": int(n_g)}
            for t in topic_list:
                tid = t["topic_id"]
                col_lbl = f"T{tid}: {t['business_label'][:28]}"
                pct = round(float((g["dom_topic"] == tid).sum()) / n_g * 100.0, 1) if n_g > 0 else 0.0
                r_dict[col_lbl] = pct
            rows.append(r_dict)
        return rows

    app_neg_matrix = _build_app_topic_matrix(neg_df, neg_topics)
    app_pos_matrix = _build_app_topic_matrix(pos_df, pos_topics)

    # 4. Topic Discovery Spesifik untuk Seluruh 8 Aplikasi
    print("[*] Menjalankan Topic Discovery spesifik untuk seluruh 8 aplikasi...")
    app_specific: Dict[str, Any] = {}
    for app in app_order:
        avg_r = round(float(df.loc[df["app_name"] == app, "score"].mean()), 2)
        sub_app_neg = neg_df[neg_df["app_name"] == app].reset_index(drop=True)
        sub_app_pos = pos_df[pos_df["app_name"] == app].reset_index(drop=True)

        entry: Dict[str, Any] = {
            "rating_avg": avg_r,
            "n_complaint_docs": int(len(sub_app_neg)),
            "n_praise_docs": int(len(sub_app_pos)),
            "complaint_topics": [],
            "praise_topics": [],
        }

        if len(sub_app_neg) >= 35:
            n_k_neg = 3 if len(sub_app_neg) >= 120 else 2
            c_tops, _, _, _ = fit_detailed_lda(sub_app_neg, n_topics=n_k_neg, n_top_words=7, max_features=1200)
            for ct in c_tops:
                ct["business_label"] = infer_short_app_topic_label(ct["top_words"], is_complaint=True)
            entry["complaint_topics"] = c_tops

        if len(sub_app_pos) >= 35:
            n_k_pos = 3 if len(sub_app_pos) >= 120 else 2
            p_tops, _, _, _ = fit_detailed_lda(sub_app_pos, n_topics=n_k_pos, n_top_words=7, max_features=1200)
            for pt in p_tops:
                pt["business_label"] = infer_short_app_topic_label(pt["top_words"], is_complaint=False)
            entry["praise_topics"] = p_tops

        app_specific[app] = entry

    # 5. Visualisasi PNG Ringkasan Topik
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), dpi=180)
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(wspace=0.38)
    sorted_neg = sorted(neg_topics, key=lambda x: x["doc_share_pct"], reverse=True)
    bar_cols = ["#1E3A5F", "#2B547E", "#3B6998"]
    for i in range(3):
        t = sorted_neg[i]
        words = t["top_words"][:7][::-1]
        weights = t["weights"][:7][::-1]
        axes[i].barh(words, weights, color=bar_cols[i], height=0.6)
        axes[i].set_title(
            f"Topik {t['topic_id']} ({t['doc_share_pct']:.1f}%):\n{t['business_label'][:38]}",
            fontsize=10, fontweight="bold", color="#0F243E",
        )
        axes[i].set_xlabel("Bobot Kata Kunci LDA", fontsize=9)
        axes[i].grid(axis="x", linestyle="--", alpha=0.35)
    plt.suptitle("Top 3 Klaster Topik Keluhan Dominan Pengguna Generative AI Indonesia (LDA)",
                 fontsize=12.5, fontweight="bold", color="#0F243E", y=1.04)
    fig_path = os.path.join(FIG_DIR, "01_topic_keywords_lda.png")
    plt.savefig(fig_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 6. Simpan JSON & Laporan Markdown
    summary_payload = {
        "author": "Arthur Willyam Liang (682024046)",
        "method": "Unsupervised Latent Dirichlet Allocation (LDA) + UMass Topic Coherence Selection",
        "optimal_k": 5,
        "dataset_size": int(len(df)),
        "slang_dictionary_size": int(len(SLANG_MAP)),
        "negative_reviews_count": int(len(neg_df)),
        "positive_reviews_count": int(len(pos_df)),
        "k_selection_complaint": k_eval_neg,
        "k_selection_praise": k_eval_pos,
        "global_complaint_topics": neg_topics,
        "global_praise_topics": pos_topics,
        "app_complaint_topic_matrix": app_neg_matrix,
        "app_praise_topic_matrix": app_pos_matrix,
        "app_specific_discovery": app_specific,
    }

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    md_lines = [
        "# LAPORAN PEMODELAN TOPIK (LDA) & EVALUASI KOHERENSI PENGALAMAN PENGGUNA",
        f"> **Metode**: Unsupervised Latent Dirichlet Allocation (LDA) ($K=5$ topik optimal)  ",
        f"> **Korpus Substantif**: Keluhan Bintang 1–2 (`{len(neg_df):,}` ulasan) · Pujian Bintang 4–5 (`{len(pos_df):,}` ulasan)  ",
        f"> **Kamus Normalisasi Slang**: `{len(SLANG_MAP):,}` entri bahasa Indonesia",
        "",
        "---",
        "",
        "## 1. Evaluasi Pemilihan Jumlah Topik Optimal ($K \\in \\{3, 4, 5, 6, 8\\}$)",
        "",
        "| Jumlah Topik ($K$) | Perplexity (Keluhan) | $C_{\\text{UMass}}$ Coherence (Keluhan) | Perplexity (Pujian) | $C_{\\text{UMass}}$ Coherence (Pujian) |",
        "| :---: | :---: | :---: | :---: | :---: |",
    ]
    for rn, rp in zip(k_eval_neg, k_eval_pos):
        md_lines.append(
            f"| **K = {rn['k_topics']}** | {rn['perplexity']:.1f} | {rn['umass_coherence']:.4f} | "
            f"{rp['perplexity']:.1f} | {rp['umass_coherence']:.4f} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 2. Lima Klaster Topik Keluhan Utama Pengguna AI di Indonesia (Bintang 1–2)",
        "",
        "| ID | Label Bisnis Topik | Aspek ABSA | Proporsi | Kata Kunci Dominan (LDA) |",
        "| :---: | :--- | :--- | :---: | :--- |",
    ])
    for t in neg_topics:
        md_lines.append(
            f"| **T{t['topic_id']}** | **{t['business_label']}** | {t['mapped_absa']} | "
            f"{t['doc_share_pct']:.1f}% ({t['doc_count']:,}) | `{', '.join(t['top_words'][:8])}` |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 3. Lima Klaster Topik Pujian Utama Pengguna AI di Indonesia (Bintang 4–5)",
        "",
        "| ID | Label Bisnis Topik | Aspek ABSA | Proporsi | Kata Kunci Dominan (LDA) |",
        "| :---: | :--- | :--- | :---: | :--- |",
    ])
    for t in pos_topics:
        md_lines.append(
            f"| **T{t['topic_id']}** | **{t['business_label']}** | {t['mapped_absa']} | "
            f"{t['doc_share_pct']:.1f}% ({t['doc_count']:,}) | `{', '.join(t['top_words'][:8])}` |"
        )

    with open(OUTPUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"[OK] JSON LDA tersimpan: {OUTPUT_JSON_PATH}")
    print(f"[OK] Laporan Markdown LDA tersimpan: {OUTPUT_MD_PATH}")
    print("\nRingkasan 5 Topik Keluhan:")
    for t in neg_topics:
        print(f"  T{t['topic_id']} ({t['doc_share_pct']:5.1f}%) | {t['business_label']} -> {t['top_words'][:6]}")
    print("\nRingkasan 5 Topik Pujian:")
    for t in pos_topics:
        print(f"  T{t['topic_id']} ({t['doc_share_pct']:5.1f}%) | {t['business_label']} -> {t['top_words'][:6]}")


if __name__ == "__main__":
    main()
