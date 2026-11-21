import json
import pandas as pd

CSV_PATH = "./dataset/RAW_DATASET_GENAI_INDONESIA_AM522B.csv"
SUMMARY_JSON_PATH = "./dataset/audit_summary.json"


def main():
    print("=" * 85)
    print("AUDIT KUALITAS & VERIFIKASI RAW DATASET GENERATIVE AI INDONESIA (AM522B)")
    print("=" * 85)

    df = pd.read_csv(CSV_PATH)

    total_rows, total_cols = df.shape
    unique_review_ids = df["reviewId"].nunique()

    print(f"\n1. DIMENSI & INTEGRITAS DATASET:")
    print(f"   - TOTAL RAW DATA (Baris Eksak) : {total_rows:,} baris")
    print(f"   - Total Kolom                  : {total_cols} kolom ({', '.join(df.columns)})")
    print(f"   - Unique reviewId              : {unique_review_ids:,} (100% Unik: {total_rows == unique_review_ids})")
    print(f"   - Missing content              : {df['content'].isna().sum()} baris")

    print(f"\n2. BREAKDOWN JUMLAH BARIS PER APLIKASI (`value_counts()` pada `app_name`):")
    app_counts = df["app_name"].value_counts()
    app_breakdown = []
    for app_name, count in app_counts.items():
        sub = df[df["app_name"] == app_name]
        app_id = sub["app_id"].iloc[0]
        min_dt = str(sub["at"].min())[:10]
        max_dt = str(sub["at"].max())[:10]
        avg_score = float(sub["score"].mean())
        pct = (count / total_rows) * 100
        print(f"   - {app_name:<28} ({app_id:<28}): {count:>7,} baris ({pct:>5.2f}%) | Rating Rata-rata: {avg_score:.2f} | Periode: {min_dt} s/d {max_dt}")
        app_breakdown.append(
            {
                "app_name": app_name,
                "app_id": app_id,
                "count": int(count),
                "pct": round(pct, 2),
                "avg_score": round(avg_score, 2),
                "min_date": min_dt,
                "max_date": max_dt,
            }
        )

    if "language_region" in df.columns:
        print(f"\n   Breakdown berdasarkan `language_region`:")
        for lr, cnt in df["language_region"].value_counts().items():
            print(f"   - {lr:<10}: {cnt:>7,} baris ({cnt / total_rows * 100:.2f}%)")

    print(f"\n3. DISTRIBUSI RATING BINTANG (1 s/d 5):")
    score_counts = df["score"].value_counts().sort_index()
    score_dist = {}
    for star, count in score_counts.items():
        pct = (count / total_rows) * 100
        print(f"   - Bintang {star} : {count:>7,} baris ({pct:>5.2f}%)")
        score_dist[str(star)] = {"count": int(count), "pct": round(pct, 2)}

    min_date_global = str(df["at"].min())
    max_date_global = str(df["at"].max())
    print(f"\n4. RENTANG TANGGAL ULASAN (`min(at)` hingga `max(at)`):")
    print(f"   - Tanggal Ulasan Terlama (min) : {min_date_global}")
    print(f"   - Tanggal Ulasan Terbaru (max) : {max_date_global}")

    print(f"\n5. 10 SAMPEL BARIS ASLI PERTAMA (VALIDASI KEASLIAN DATA):")
    sample_cols = ["reviewId", "app_name", "userName", "score", "at", "reviewCreatedVersion", "thumbsUpCount", "content"]
    for idx, row in df[sample_cols].head(10).iterrows():
        content_preview = str(row["content"]).replace("\n", " ")[:110]
        print(
            f"   [{idx+1:02d}] ID={row['reviewId'][:12]}... | App={row['app_name']} | "
            f"User={row['userName']} | ⭐={row['score']} | Tgl={str(row['at'])[:19]} | "
            f"Ver={row['reviewCreatedVersion']} | 👍={row['thumbsUpCount']} | Ulasan: \"{content_preview}...\""
        )

    summary = {
        "total_rows": int(total_rows),
        "total_cols": int(total_cols),
        "unique_review_ids": int(unique_review_ids),
        "min_date": min_date_global,
        "max_date": max_date_global,
        "app_breakdown": app_breakdown,
        "score_distribution": score_dist,
        "language_region": {str(k): int(v) for k, v in df["language_region"].value_counts().items()} if "language_region" in df.columns else {},
    }
    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n" + "=" * 85)
    print(f"Ringkasan audit disimpan ke: {SUMMARY_JSON_PATH}")
    print("=" * 85)


if __name__ == "__main__":
    main()
