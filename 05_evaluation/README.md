# Bab 5: Evaluasi Model (Evaluation)

Folder ini menyimpan skrip dan hasil evaluasi performa model pada data uji (20% = 14.245 ulasan):

- `metric_sanity_audit.py`: Skrip pengujian metrik evaluasi (*Macro F1-Score*, *Weighted F1*, *Balanced Accuracy*, *ROC-AUC OvR*, dan *MCC*) serta pemeriksaan pencegahan *data leakage*.
- `audit_results.json` & `metric_sanity_audit_report.md`: Ringkasan hasil perbandingan model terhadap *Dummy Baseline*, evaluasi pada ulasan panjang ($\ge 5$ kata), serta pengujian lintas aplikasi (*Leave-One-App-Out*).
- `confident_errors_fp_fn.csv`: Sampel kasus ulasan di mana prediksi teks berbeda dengan rating bintang akibat fenomena *Rating-Text Discrepancy*.
- `figures/01_confusion_matrix_best_model.png`: Plot *Confusion Matrix* dari model terbaik (*Tuned Logistic Regression*).
- `figures/02_feature_attribution_shap_top_words.png`: Grafik bobot kata kunci teratas pendorong sentimen positif dan negatif.
