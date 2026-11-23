# Bab 4: Pemodelan (Modeling)

Folder ini berisi skrip pelatihan model klasifikasi sentimen dan eksperimen pemodelan topik:

1. **Klasifikasi Sentimen 3-Kelas (`train_sentiment_models.py`)**:
   - **Baseline**: `LogisticRegression(class_weight='balanced')` dengan fitur `TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True)`.
   - **Model Alternatif 1**: `ComplementNB` (Multinomial Naive Bayes untuk data teks dengan kelas tidak seimbang).
   - **Model Alternatif 2**: `CalibratedClassifierCV(LinearSVC(class_weight='balanced'))`.
   - Penalaan parameter dilakukan menggunakan `GridSearchCV` 5-fold cross validation pada data latih (80%). Model terbaik disimpan ke `../models/genai_absa_pipeline.joblib`.

2. **Pemodelan Topik Keluhan & Pujian (`run_topic_modeling_experiment.py`)**:
   - Menggunakan algoritma *Latent Dirichlet Allocation* (LDA) untuk mengelompokkan topik bahasan utama pengguna ke dalam 5 klaster topik.
   - Hasil visualisasi kata kunci per topik disimpan pada `figures/01_topic_keywords_lda.png`.
