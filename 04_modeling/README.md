# CRISP-DM FASE 4: MODELING (DUAL-STAGE EXPERIMENT)

Folder ini digunakan oleh **Member 1**, **Member 2**, dan **Member 3** untuk menjalankan dua eksperimen pemodelan utama proyek akhir AM522B:

## 1. Experiment 1: Supervised Sentiment & Rating-Text Discrepancy Classification
- **Penanggung Jawab**: Member 1 (Project Lead, Modeling & XAI) & Member 2 (Feature Engineering)
- **Model yang Dibandingkan**:
  1. **Ensemble Gradient Boosting**: `CatBoost`, `XGBoost`, `LightGBM` (menggunakan fitur *TF-IDF Unigram-Trigram* + *Lexicon Score* + *Review Metadata* `char_length`, `thumbsUpCount`).
  2. **Pre-trained Language Model**: **IndoBERT** (`indobenchmark/indobert-base-p1`).
- **Target Klasifikasi**:
  - `Task 1A`: Polaritas Sentimen 3-Kelas (Positif, Netral, Negatif) & Skor Aspek (ABSA).
  - `Task 1B`: Deteksi *Rating-Text Discrepancy* (`Consistent` vs. `Discrepancy`).

## 2. Experiment 2: Unsupervised Topic Modeling & Aspect Discovery
- **Penanggung Jawab**: Member 3 (EDA & Topic Modeling)
- **Model yang Dibandingkan**:
  1. **Probabilistic Topic Model**: `Latent Dirichlet Allocation (LDA)` via `scikit-learn` / `gensim`.
  2. **Neural Topic Model**: `BERTopic` (Multilingual Sentence-Transformer Embeddings + UMAP + HDBSCAN + c-TF-IDF).
