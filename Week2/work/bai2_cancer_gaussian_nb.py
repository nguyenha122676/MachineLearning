"""
Bài 2: Phân loại bệnh ung thư vú Wisconsin bằng Gaussian Naive Bayes (Bài tập tự giải)
Họ và tên: Nguyễn Hải Anh - MSSV: 24001968
Môn học: Machine Learning - Bài thực hành số 2 (Naive Bayes)

Nội dung:
- Đọc tập dữ liệu breast-cancer-wisconsin.data (699 mẫu, 10 thuộc tính y sinh, 2 lớp: 2=lành tính, 4=ác tính)
- Tiền xử lý dữ liệu:
  + Đặt tên cột chuẩn theo tài liệu (loại bỏ cột id không mang tính phân loại)
  + Xử lý 16 giá trị thiếu ('?') ở cột bare_nuclei bằng trung vị (median)
- Phân chia dữ liệu theo đúng yêu cầu đề bài:
  + Chọn ngẫu nhiên 80 mẫu lành tính (class 2) và 40 mẫu ác tính (class 4) làm Test (tổng 120 mẫu)
  + Toàn bộ mẫu còn lại (579 mẫu) làm Training
- Huấn luyện mô hình Gaussian Naive Bayes (cả tự cài đặt Scratch và Scikit-Learn)
- Đánh giá toàn diện: Accuracy, Precision, Recall, F1-Score (đặc biệt cho lớp ác tính - Malignant), Confusion Matrix
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def run_bai2():
    print("=" * 70)
    print(" BÀI 2: PHÂN LOẠI UNG THƯ VÚ WISCONSIN (GAUSSIAN NAIVE BAYES)")
    print("=" * 70)

    # 1. Đọc dữ liệu
    data_path = os.path.join(os.path.dirname(__file__), "..", "breast-cancer-wisconsin.data")
    if not os.path.exists(data_path):
        data_path = "breast-cancer-wisconsin.data"

    col_names = [
        'id', 'clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity',
        'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
        'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class'
    ]
    df = pd.read_csv(data_path, header=None, names=col_names)
    print(f"[*] Kích thước dữ liệu gốc: {df.shape[0]} dòng, {df.shape[1]} cột")

    # 2. Tiền xử lý dữ liệu
    # Kiểm tra giá trị thiếu '?' ở cột bare_nuclei
    missing_count = (df['bare_nuclei'] == '?').sum()
    print(f"[*] Số lượng giá trị thiếu ('?') trong cột bare_nuclei: {missing_count}")
    
    # Chuyển đổi '?' thành NaN và điền khuyết bằng trung vị (median) theo từng lớp
    df['bare_nuclei'] = pd.to_numeric(df['bare_nuclei'].replace('?', np.nan))
    df['bare_nuclei'] = df.groupby('class')['bare_nuclei'].transform(lambda s: s.fillna(s.median()))

    # Loại bỏ cột 'id' vì không có ý nghĩa phân loại bệnh học
    feature_cols = [c for c in col_names if c not in ['id', 'class']]

    print(f"[*] Số thuộc tính đặc trưng sử dụng: {len(feature_cols)}")
    print(f"[*] Danh sách đặc trưng: {feature_cols}")
    print(f"[*] Phân bố nhãn toàn bộ tập dữ liệu: {dict(df['class'].value_counts())}")
    print("    (2: Lành tính - Benign, 4: Ác tính - Malignant)\n")

    # 3. Phân chia Train/Test theo đúng yêu cầu đề bài:
    # Lấy 80 mẫu lành tính (class 2) và 40 mẫu ác tính (class 4) làm Test set
    test_benign = df[df['class'] == 2].sample(n=80, random_state=42)
    test_malignant = df[df['class'] == 4].sample(n=40, random_state=42)
    test_df = pd.concat([test_benign, test_malignant])

    # Tập train là phần còn lại
    train_df = df.drop(test_df.index)

    X_train = train_df[feature_cols].values
    y_train = train_df['class'].values

    X_test = test_df[feature_cols].values
    y_test = test_df['class'].values

    print(f"[*] Tập Train: {train_df.shape[0]} mẫu (Lành tính: {(y_train == 2).sum()}, Ác tính: {(y_train == 4).sum()})")
    print(f"[*] Tập Test:  {test_df.shape[0]} mẫu (Lành tính: {(y_test == 2).sum()}, Ác tính: {(y_test == 4).sum()})\n")

    # ==============================================================
    # 4. MÔ HÌNH HÓA VÀ DỰ ĐOÁN
    # ==============================================================
    # 4.1. Thư viện Scikit-Learn
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred_sk = gnb.predict(X_test)
    y_proba_sk = gnb.predict_proba(X_test)[:, 1] # xác suất lớp ác tính (4)

    # 4.2. Tự lập trình (Scratch) để đối chiếu
    classes = [2, 4]
    means = {c: np.mean(X_train[y_train == c], axis=0) for c in classes}
    stds = {c: np.std(X_train[y_train == c], axis=0, ddof=0) for c in classes}
    priors = {c: (y_train == c).sum() / len(y_train) for c in classes}

    y_pred_scratch = []
    for x in X_test:
        scores = {}
        for c in classes:
            log_prior = np.log(priors[c])
            log_lik = np.sum(norm.logpdf(x, loc=means[c], scale=stds[c]))
            scores[c] = log_prior + log_lik
        pred_c = max(scores, key=scores.get)
        y_pred_scratch.append(pred_c)
    y_pred_scratch = np.array(y_pred_scratch)

    # ==============================================================
    # 5. ĐÁNH GIÁ KẾT QUẢ VÀ CÁC ĐỘ ĐO
    # ==============================================================
    print("--- ĐÁNH GIÁ HIỆU NĂNG MÔ HÌNH TRÊN TẬP TEST (120 MẪU) ---")
    acc = accuracy_score(y_test, y_pred_sk)
    prec_pos = precision_score(y_test, y_pred_sk, pos_label=4)
    rec_pos = recall_score(y_test, y_pred_sk, pos_label=4)
    f1_pos = f1_score(y_test, y_pred_sk, pos_label=4)
    roc_auc = roc_auc_score(y_test == 4, y_proba_sk)

    print(f"-> Độ chính xác tổng thể (Accuracy): {acc * 100:.2f}%")
    print(f"-> Mức độ khớp giữa Scratch và Sklearn: {np.mean(y_pred_scratch == y_pred_sk) * 100:.2f}%")
    print("\nChỉ số dành riêng cho Lớp Ác tính (Malignant - Class 4):")
    print(f"   + Precision (Độ chuẩn xác): {prec_pos * 100:.2f}% (Tỷ lệ dự đoán ác tính thực sự bị ung thư)")
    print(f"   + Recall (Độ nhạy / Độ bao phủ): {rec_pos * 100:.2f}% (Tỷ lệ phát hiện được ca ác tính thực tế)")
    print(f"   + F1-Score: {f1_pos * 100:.2f}% (Trung bình điều hòa giữa Precision và Recall)")
    print(f"   + ROC-AUC:  {roc_auc:.4f}")

    cm = confusion_matrix(y_test, y_pred_sk, labels=[2, 4])
    print("\nMa trận nhầm lẫn (Confusion Matrix):")
    cm_df = pd.DataFrame(cm, index=['Thực tế: Lành tính (2)', 'Thực tế: Ác tính (4)'],
                         columns=['Dự đoán: Lành tính (2)', 'Dự đoán: Ác tính (4)'])
    print(cm_df)

    print("\nBáo cáo phân loại chi tiết (Classification Report):")
    print(classification_report(y_test, y_pred_sk, target_names=['Lành tính (2)', 'Ác tính (4)'], digits=4))

    # Vẽ và lưu ma trận nhầm lẫn
    save_fig_dir = os.path.dirname(__file__)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Lành tính (2)', 'Ác tính (4)'],
                yticklabels=['Lành tính (2)', 'Ác tính (4)'])
    plt.title('Ma trận nhầm lẫn - Breast Cancer Wisconsin (Gaussian NB)')
    plt.xlabel('Nhãn dự đoán')
    plt.ylabel('Nhãn thực tế')
    plt.tight_layout()
    cm_path = os.path.join(save_fig_dir, "bai2_cancer_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[*] Đã lưu biểu đồ ma trận nhầm lẫn tại: {cm_path}")

    return {
        'accuracy': acc,
        'precision_malignant': prec_pos,
        'recall_malignant': rec_pos,
        'f1_malignant': f1_pos,
        'roc_auc': roc_auc,
        'cm': cm
    }

if __name__ == "__main__":
    run_bai2()
