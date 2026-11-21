# CRISP-DM FASE 5: EVALUATION & EXPLAINABLE AI (SHAP)

Folder ini menyimpan hasil pengujian performa model dan analisis interpretabilitas fitur (**Penanggung Jawab: Member 1**):

1. **Tabel Komparasi Metrik Klasifikasi (`Experiment 1`)**:
   - *Accuracy*, *Precision (Macro)*, *Recall (Macro)*, ***Macro F1-Score*** (metrik utama untuk mengatasi ketimpangan jumlah kelas Bintang 5 vs Bintang 1–4), dan ***ROC-AUC (One-vs-Rest)***.
   - Visualisasi *Confusion Matrix* per algoritma (`CatBoost`, `XGBoost`, `LightGBM`, `IndoBERT`).
2. **Evaluasi Koherensi Topik (`Experiment 2`)**:
   - Perbandingan *Topic Coherence ($C_v$)* dan *Topic Diversity* antara `LDA` dan `BERTopic`.
3. **Explainable AI dengan SHAP (*SHapley Additive exPlanations*)**:
   - *SHAP Beeswarm / Summary Plot* untuk menampilkan 20 fitur kata kunci dan metadata yang paling memicu ketidakpuasan pengguna pada masing-masing aplikasi Generative AI.
