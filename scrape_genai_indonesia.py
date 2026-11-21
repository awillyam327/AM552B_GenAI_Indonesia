import os
import glob
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from google_play_scraper import Sort, reviews

# ==============================================================================
# KONFIGURASI TARGET APLIKASI GENERATIVE AI (GOOGLE PLAY STORE)
# ==============================================================================
PRIMARY_APPS = [
    {"app_id": "com.openai.chatgpt", "alt_ids": [], "app_name": "ChatGPT (OpenAI)", "cap": 22000},
    {"app_id": "com.google.android.apps.bard", "alt_ids": [], "app_name": "Google Gemini", "cap": 15000},
    {"app_id": "com.deepseek.chat", "alt_ids": [], "app_name": "DeepSeek - AI Assistant", "cap": 10000},
    {"app_id": "com.microsoft.copilot", "alt_ids": [], "app_name": "Microsoft Copilot", "cap": 10000},
    {"app_id": "ai.perplexity.app.android", "alt_ids": [], "app_name": "Perplexity - Ask Anything", "cap": 12000},
    {"app_id": "com.anthropic.claude", "alt_ids": [], "app_name": "Claude by Anthropic", "cap": 10000},
    {"app_id": "com.poe.android", "alt_ids": ["com.quora.poe"], "app_name": "Poe - Fast AI Chat", "cap": 8000},
    {"app_id": "ai.character.app", "alt_ids": ["com.character.ai"], "app_name": "Character.AI", "cap": 12000},
]

SECONDARY_APPS_ID = [
    {"app_id": "notion.id", "alt_ids": ["com.notion.id"], "app_name": "Notion - Notes, Tasks, AI", "cap": 8000},
    {"app_id": "ai.qwenlm.chat.android", "alt_ids": ["com.aliyun.tongyi"], "app_name": "Qwen (Alibaba AI)", "cap": 8000},
    {"app_id": "com.gamma.app", "alt_ids": [], "app_name": "Gamma AI", "cap": 6000},
]

CHECKPOINT_DIR = "./dataset/checkpoints"
FINAL_CSV_PATH = "./dataset/RAW_DATASET_GENAI_INDONESIA_AM522B.csv"
TARGET_MIN_ROWS = 48000
TARGET_MAX_ROWS = 62000

COLUMNS_ORDER = [
    "reviewId",
    "app_name",
    "app_id",
    "userName",
    "content",
    "score",
    "thumbsUpCount",
    "reviewCreatedVersion",
    "at",
    "replyContent",
    "language_region",
]

print_lock = threading.Lock()


def safe_print(msg):
    with print_lock:
        print(msg, flush=True)


def normalize_records(raw_list, app_name, app_id, language_region):
    """Normalisasi dictionary hasil google-play-scraper sesuai skema wajib."""
    cleaned = []
    for r in raw_list:
        rev_id = r.get("reviewId")
        content = r.get("content")
        if not rev_id or content is None or str(content).strip() == "":
            continue
        cleaned.append(
            {
                "reviewId": str(rev_id).strip(),
                "app_name": app_name,
                "app_id": app_id,
                "userName": r.get("userName", "Pengguna Google"),
                "content": str(content).strip(),
                "score": r.get("score"),
                "thumbsUpCount": r.get("thumbsUpCount", 0),
                "reviewCreatedVersion": r.get("reviewCreatedVersion"),
                "at": r.get("at"),
                "replyContent": r.get("replyContent"),
                "language_region": language_region,
            }
        )
    return cleaned


def fetch_reviews_with_retry(app_id, lang, country, sort_order, score_filter, continuation_token, max_retries=5):
    """Panggil google_play_scraper.reviews dengan retry otomatis bila koneksi terputus sementara."""
    for attempt in range(max_retries):
        try:
            return reviews(
                app_id,
                lang=lang,
                country=country,
                sort=sort_order,
                count=200,
                filter_score_with=score_filter,
                continuation_token=continuation_token,
            )
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1.0 * (attempt + 1))
            else:
                safe_print(f"    [WARN] Gagal setelah {max_retries} percobaan pada {app_id}: {e}")
                return [], None
    return [], None


def resolve_working_app_id(app_info, lang="id", country="id"):
    """Pastikan app_id utama memiliki ulasan; jika kosong/404, cek alt_ids."""
    candidates = [app_info["app_id"]] + app_info.get("alt_ids", [])
    for cid in candidates:
        res, _ = fetch_reviews_with_retry(cid, lang, country, Sort.NEWEST, None, None, max_retries=3)
        if res:
            return cid
    return app_info["app_id"]


def save_checkpoint(records, filename):
    if not records:
        return
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    df_cp = pd.DataFrame(records)
    df_cp.drop_duplicates(subset=["reviewId"], keep="first", inplace=True)
    df_cp = df_cp[COLUMNS_ORDER]
    cp_path = os.path.join(CHECKPOINT_DIR, filename)
    df_cp.to_csv(cp_path, index=False, encoding="utf-8-sig")
    safe_print(f"  [CHECKPOINT SAVED] {cp_path} ({len(df_cp):,} baris unik)")


def find_existing_checkpoint(app_info, lang_reg):
    """Cek apakah checkpoint untuk aplikasi ini sudah ada di disk."""
    candidates = [app_info["app_id"]] + app_info.get("alt_ids", [])
    for cid in candidates:
        cp_path = os.path.join(CHECKPOINT_DIR, f"{cid}_{lang_reg}.csv")
        if os.path.exists(cp_path):
            try:
                df_cp = pd.read_csv(cp_path)
                if len(df_cp) > 100:
                    safe_print(f"  [RESUME CHECKPOINT] Memuat {cp_path} ({len(df_cp):,} baris yang sudah tersimpan)")
                    return df_cp.to_dict("records")
            except Exception as e:
                safe_print(f"  [WARN] Gagal membaca checkpoint {cp_path}: {e}")
    return None


def scrape_app_stream(
    app_id,
    app_name,
    lang="id",
    country="id",
    sort_order=Sort.NEWEST,
    score_filter=None,
    seen_ids=None,
    max_new_for_stream=22000,
):
    """Menarik ulasan menggunakan continuation_token dalam batch=200."""
    if seen_ids is None:
        seen_ids = set()

    language_region = f"{lang}_{country.upper()}"
    collected = []
    continuation_token = None
    consecutive_zero_new = 0
    exhausted_naturally = False
    batch_idx = 0

    while len(collected) < max_new_for_stream:
        result, continuation_token = fetch_reviews_with_retry(
            app_id, lang, country, sort_order, score_filter, continuation_token
        )

        if not result:
            exhausted_naturally = True
            break

        batch_records = normalize_records(result, app_name, app_id, language_region)
        new_records = [rec for rec in batch_records if rec["reviewId"] not in seen_ids]

        for rec in new_records:
            seen_ids.add(rec["reviewId"])
        collected.extend(new_records)
        batch_idx += 1

        if batch_idx % 20 == 0:
            safe_print(
                f"    ... [{app_name} | {language_region} | score={score_filter}] Batch #{batch_idx}: "
                f"terkumpul {len(collected):,} ulasan di stream ini"
            )

        if len(new_records) == 0:
            consecutive_zero_new += 1
            if consecutive_zero_new >= 3:
                break
        else:
            consecutive_zero_new = 0

        if continuation_token is None:
            exhausted_naturally = True
            break

        time.sleep(0.06)

    return collected, exhausted_naturally


def scrape_full_app(app_info, lang, country, max_per_app=None):
    """
    Menarik ulasan maksimal untuk satu aplikasi dengan mengombinasikan
    Sort.NEWEST, Sort.MOST_RELEVANT, dan filter_score_with=1..5 bila batas paginasi tercapai.
    """
    if max_per_app is None:
        max_per_app = app_info.get("cap", 15000)

    lang_reg = f"{lang}_{country.upper()}"

    # Cek checkpoint terlebih dahulu agar bisa resume instan
    existing = find_existing_checkpoint(app_info, lang_reg)
    if existing is not None:
        return existing

    working_id = resolve_working_app_id(app_info, lang=lang, country=country)
    app_name = app_info["app_name"]
    cp_filename = f"{working_id}_{lang_reg}.csv"

    safe_print(f"\n>>> Memulai scraping: {app_name} ({working_id}) | Region: {lang_reg} | Cap: {max_per_app:,}")
    local_seen_ids = set()
    app_records = []

    # 1. Tarik dengan Sort.NEWEST tanpa filter skor
    recs, exhausted = scrape_app_stream(
        working_id,
        app_name,
        lang=lang,
        country=country,
        sort_order=Sort.NEWEST,
        score_filter=None,
        seen_ids=local_seen_ids,
        max_new_for_stream=max_per_app,
    )
    app_records.extend(recs)
    safe_print(f"  - [{app_name} | {lang_reg}] Sort.NEWEST (All): +{len(recs):,} ulasan unik")
    save_checkpoint(app_records, cp_filename)

    # 2. Hanya jalankan MOST_RELEVANT & Star Filter bila Sort.NEWEST terpotong batas paginasi Play Store
    # (misal terkumpul >= 4000 tapi belum exhausted, atau ingin menambah variasi)
    if len(app_records) < max_per_app and not exhausted and len(app_records) >= 3500:
        recs_rel, _ = scrape_app_stream(
            working_id,
            app_name,
            lang=lang,
            country=country,
            sort_order=Sort.MOST_RELEVANT,
            score_filter=None,
            seen_ids=local_seen_ids,
            max_new_for_stream=max_per_app - len(app_records),
        )
        app_records.extend(recs_rel)
        if len(recs_rel) > 0:
            safe_print(f"  - [{app_name} | {lang_reg}] Sort.MOST_RELEVANT (All): +{len(recs_rel):,} ulasan unik")
            save_checkpoint(app_records, cp_filename)

        for star in [1, 2, 3, 4, 5]:
            if len(app_records) >= max_per_app:
                break
            recs_star, _ = scrape_app_stream(
                working_id,
                app_name,
                lang=lang,
                country=country,
                sort_order=Sort.NEWEST,
                score_filter=star,
                seen_ids=local_seen_ids,
                max_new_for_stream=max_per_app - len(app_records),
            )
            app_records.extend(recs_star)
            if len(recs_star) > 0:
                safe_print(f"  - [{app_name} | {lang_reg}] Sort.NEWEST (Star={star}): +{len(recs_star):,} ulasan unik")
                save_checkpoint(app_records, cp_filename)

    return app_records


def deduplicate_records(records):
    seen = set()
    unique = []
    for r in records:
        rid = str(r["reviewId"]).strip()
        if rid not in seen:
            seen.add(rid)
            unique.append(r)
    return unique


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(FINAL_CSV_PATH), exist_ok=True)

    all_records = []

    print("=" * 85)
    print("TAHAP 1 & 2: HIGH-YIELD GOOGLE PLAY SCRAPER - EKOSISTEM GENERATIVE AI INDONESIA")
    print("=" * 85)

    # PASS 1: 8 Aplikasi Utama Generative AI dengan lang='id', country='id' (2 worker agar stabil)
    print("\n[PASS 1] Scraping/Resuming 8 Aplikasi Utama Generative AI (lang='id', country='id')...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(scrape_full_app, app_info, "id", "id", app_info.get("cap", 15000)): app_info
            for app_info in PRIMARY_APPS
        }
        for future in as_completed(futures):
            recs = future.result()
            all_records.extend(recs)
            all_records = deduplicate_records(all_records)
            safe_print(f"  => [PROGRESS PASS 1] Total Akumulasi Unik: {len(all_records):,} baris")

    # Evaluasi apakah Pass 1 sudah menembus TARGET_MIN_ROWS (48.000+)
    if len(all_records) < TARGET_MIN_ROWS:
        print("\n" + "=" * 85)
        print(
            f"[PASS 2A] Total ulasan lang='id' pada 8 aplikasi utama ({len(all_records):,}) "
            f"masih di bawah target {TARGET_MIN_ROWS:,}. Menjalankan ekspansi ke aplikasi AI produktivitas/asisten (lang='id')..."
        )
        print("=" * 85)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(scrape_full_app, app_info, "id", "id", app_info.get("cap", 8000)): app_info
                for app_info in SECONDARY_APPS_ID
            }
            for future in as_completed(futures):
                recs = future.result()
                all_records.extend(recs)
                all_records = deduplicate_records(all_records)
                safe_print(f"  => [PROGRESS PASS 2A] Total Akumulasi Unik: {len(all_records):,} baris")

    # PASS 2B: Jika masih di bawah TARGET_MIN_ROWS, tarik ulasan berbahasa Inggris dari Store Indonesia (lang='en', country='id')
    if len(all_records) < TARGET_MIN_ROWS:
        print("\n" + "=" * 85)
        print(
            f"[PASS 2B] Total ulasan saat ini ({len(all_records):,}) masih di bawah {TARGET_MIN_ROWS:,}. "
            f"Menarik ulasan regional/global pada aplikasi utama Generative AI (lang='en', country='id') "
            f"dengan pelabelan transparan pada kolom `language_region`..."
        )
        print("=" * 85)
        remaining_needed = TARGET_MAX_ROWS - len(all_records)
        per_app_cap = max(3000, int(remaining_needed / len(PRIMARY_APPS)) + 1000)

        for app_info in PRIMARY_APPS:
            if len(all_records) >= TARGET_MAX_ROWS:
                break
            allowance = min(per_app_cap, TARGET_MAX_ROWS - len(all_records))
            recs = scrape_full_app(app_info, "en", "id", max_per_app=allowance)
            all_records.extend(recs)
            all_records = deduplicate_records(all_records)
            safe_print(f"  => [PROGRESS PASS 2B] Total Akumulasi Unik: {len(all_records):,} baris")

    # Gabungkan dan deduplikasi akhir berdasarkan reviewId
    print("\n" + "=" * 85)
    print("MENGGABUNGKAN DAN MENDEDUPLIKASI SELURUH DATA...")
    df_final = pd.DataFrame(all_records)
    before_dedup = len(df_final)
    df_final.drop_duplicates(subset=["reviewId"], keep="first", inplace=True)
    df_final = df_final[df_final["content"].notna() & (df_final["content"].astype(str).str.strip() != "")]
    df_final = df_final[COLUMNS_ORDER]
    after_dedup = len(df_final)

    df_final.to_csv(FINAL_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"Sebelum deduplikasi akhir : {before_dedup:,} baris")
    print(f"Setelah deduplikasi akhir : {after_dedup:,} baris unik")
    print(f"File tersimpan di         : {FINAL_CSV_PATH}")
    print("=" * 85)


if __name__ == "__main__":
    main()
