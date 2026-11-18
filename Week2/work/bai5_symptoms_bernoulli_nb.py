"""
Bài 5: Chẩn đoán 8 bệnh lý từ 30 triệu chứng lâm sàng bằng Bernoulli Naive Bayes
Họ và tên: Nguyễn Hải Anh - MSSV: 24001968
Môn học: Machine Learning - Bài thực hành số 2 (Naive Bayes)

Nội dung:
- Đọc tập dữ liệu bernoulli_nb_symptoms.csv (2500 bệnh nhân, 30 triệu chứng dạng nhị phân 0/1, 8 bệnh lý)
- Đặc trưng triệu chứng (30 binary features):
  fever, chills, runny_nose, sneezing, cough, sore_throat, headache, nausea, vomiting,
  diarrhea, stomach_pain, muscle_aches, fatigue, loss_appetite, loss_taste_smell,
  shortness_breath, chest_tightness, dizziness, light_sensitivity, noise_sensitivity,
  dehydration_signs, dry_mouth, dark_urine, rapid_heart_rate, sweating, itchy_eyes,
  watery_eyes, nasal_congestion, postnasal_drip, joint_pain
- Nhãn chẩn đoán (condition - 8 lớp):
  cold, flu, covid_like, stomach_bug, allergy, food_poison, dehydration, migraine
- Phân chia train:test = 6:4 (60% Train = 1500 mẫu, 40% Test = 1000 mẫu, stratified)
- Huấn luyện mô hình Bernoulli Naive Bayes (BernoulliNB với Laplace smoothing alpha=1.0)
- Phân tích xác suất điều kiện P(triệu chứng | bệnh lý) để hiểu các triệu chứng chỉ dấu đặc trưng
- Đánh giá toàn diện: Accuracy, Macro/Weighted Precision, Recall, F1-Score, Confusion Matrix
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def run_bai5():
    print("=" * 70)
    print(" BÀI 5: CHẨN ĐOÁN BỆNH TỪ TRIỆU CHỨNG LÂM SÀNG (BERNOULLI NAIVE BAYES)")
    print("=" * 70)

    # 1. Đọc dữ liệu
    csv_path = os.path.join(os.path.dirname(__file__), "..", "bernoulli_nb_symptoms.csv")
    if not os.path.exists(csv_path):
        csv_path = "bernoulli_nb_symptoms.csv"

    df = pd.read_csv(csv_path)
    target_col = 'condition'
    feature_cols = [c for c in df.columns if c != target_col]

    print(f"[*] Kích thước dữ liệu: {df.shape[0]} mẫu bệnh nhân, {len(feature_cols)} triệu chứng nhị phân")
    print(f"[*] Số lượng bệnh lý chẩn đoán: {df[target_col].nunique()}")
    print(f"[*] Phân bố bệnh nhân theo từng bệnh lý:\n{df[target_col].value_counts()}\n")

    X = df[feature_cols]
    y = df[target_col]

    # 2. Phân chia Train/Test theo tỉ lệ 6:4 (60% Train = 1500, 40% Test = 1000)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    print(f"[*] Tập Train: {X_train.shape[0]} mẫu")
    print(f"[*] Tập Test:  {X_test.shape[0]} mẫu\n")

    # 3. Huấn luyện mô hình Bernoulli Naive Bayes
    bnb = BernoulliNB(alpha=1.0)
    bnb.fit(X_train, y_train)

    y_pred = bnb.predict(X_test)
    y_proba = bnb.predict_proba(X_test)
    classes = bnb.classes_

    # ==============================================================
    # 4. ĐÁNH GIÁ VÀ BÁO CÁO KẾT QUẢ
    # ==============================================================
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average='macro')
    rec_macro = recall_score(y_test, y_pred, average='macro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    print("--- KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH BERNOULLI NAIVE BAYES ---")
    print(f"-> Độ chính xác tổng thể (Accuracy): {acc * 100:.2f}%")
    print(f"-> Precision (Macro Avg):  {prec_macro * 100:.2f}%")
    print(f"-> Recall (Macro Avg):     {rec_macro * 100:.2f}%")
    print(f"-> F1-Score (Macro Avg):   {f1_macro * 100:.2f}%")
    print(f"-> F1-Score (Weighted):    {f1_weighted * 100:.2f}%\n")

    print("[*] Nhận xét baseline ngẫu nhiên: Với 8 lớp, đoán ngẫu nhiên chỉ đạt 12.5%.")
    print(f"    Mô hình đạt {acc * 100:.2f}%, cao gấp 4 lần so với đoán ngẫu nhiên.\n")

    print("Báo cáo phân loại chi tiết từng bệnh lý:")
    print(classification_report(y_test, y_pred, digits=4))

    # 5. Phân tích các triệu chứng chỉ dấu nổi bật (Top Features per Disease)
    # feature_log_prob_ chứa log P(x_i = 1 | y = c)
    # Chuyển về xác suất thực tế P(x_i = 1 | y = c) = exp(log_prob)
    feature_probs = np.exp(bnb.feature_log_prob_)
    df_feat_prob = pd.DataFrame(feature_probs, index=classes, columns=feature_cols)

    print("\n--- TOP 3 TRIỆU CHỨNG CÓ XÁC SUẤT XUẤT HIỆN CAO NHẤT THEO TỪNG BỆNH ---")
    for disease in classes:
        top3 = df_feat_prob.loc[disease].sort_values(ascending=False).head(3)
        top_str = ", ".join([f"{feat} ({val*100:.1f}%)" for feat, val in top3.items()])
        print(f"[*] {disease:12s}: {top_str}")

    # Vẽ và lưu ma trận nhầm lẫn
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    save_fig_dir = os.path.dirname(__file__)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=classes, yticklabels=classes)
    plt.title('Ma trận nhầm lẫn - Chẩn đoán 8 bệnh lý (Bernoulli NB)')
    plt.xlabel('Chẩn đoán dự đoán')
    plt.ylabel('Bệnh lý thực tế')
    plt.xticks(rotation=30)
    plt.tight_layout()
    cm_path = os.path.join(save_fig_dir, "bai5_symptoms_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"\n[*] Đã lưu biểu đồ ma trận nhầm lẫn tại: {cm_path}")

    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'cm': cm
    }

if __name__ == "__main__":
    run_bai5()
