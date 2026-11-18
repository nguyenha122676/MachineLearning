"""
Bài 1: Phân loại hoa Iris bằng mô hình Gaussian Naive Bayes (Mở rộng 4 đặc trưng)
Họ và tên: Nguyễn Hải Anh - MSSV: 24001968
Môn học: Machine Learning - Bài thực hành số 2 (Naive Bayes)

Nội dung:
- Mở rộng phân loại hoa Iris với đầy đủ 4 thuộc tính (sepal length, sepal width, petal length, petal width)
- Tỉ lệ phân chia train:test = 4:1 (80% train, 20% test)
- Triển khai 2 phương pháp:
  1. Tự cài đặt từ đầu (From Scratch) bằng NumPy & Pandas theo công thức MLE
  2. Sử dụng thư viện Scikit-Learn (GaussianNB)
- Đánh giá toàn diện: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def run_bai1():
    print("=" * 70)
    print(" BÀI 1: PHÂN LOẠI HOA IRIS VỚI ĐẦY ĐỦ 4 ĐẶC TRƯNG (GAUSSIAN NAIVE BAYES)")
    print("=" * 70)

    # 1. Đọc dữ liệu Iris
    csv_path = os.path.join(os.path.dirname(__file__), "..", "iris.csv")
    if not os.path.exists(csv_path):
        csv_path = "iris.csv"
    
    df = pd.read_csv(csv_path)
    # Chuẩn hóa tên cột
    df.columns = [c.replace(".", "_") for c in df.columns]
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    target_col = 'variety'

    print(f"[*] Dữ liệu Iris: {df.shape[0]} mẫu, 4 đặc trưng liên tục.")
    print(f"[*] Phân bố các lớp: {dict(df[target_col].value_counts())}")

    X = df[feature_cols].values
    y_raw = df[target_col].values
    
    # Mã hóa nhãn dạng số: Setosa=0, Versicolor=1, Virginica=2
    unique_classes = np.unique(y_raw)
    class_map = {name: idx for idx, name in enumerate(unique_classes)}
    idx_to_class = {idx: name for name, idx in class_map.items()}
    y = np.array([class_map[val] for val in y_raw])

    # 2. Phân chia Train/Test theo tỉ lệ 4:1 (80% Train, 20% Test)
    # Stratified split đảm bảo mỗi lớp có đúng 40 train và 10 test (khớp tài liệu lab)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[*] Tập Train: {X_train.shape[0]} mẫu (40 mẫu/lớp)")
    print(f"[*] Tập Test:  {X_test.shape[0]} mẫu (10 mẫu/lớp)\n")

    # ==============================================================
    # PHƯƠNG PHÁP 1: TỰ CÀI ĐẶT (FROM SCRATCH) THEO CÔNG THỨC TOÁN HỌC
    # ==============================================================
    print("--- 1. HUẤN LUYỆN TỰ CÀI ĐẶT (SCRATCH GAUSSIAN NAIVE BAYES) ---")
    n_classes = len(unique_classes)
    n_features = X_train.shape[1]

    # Ước lượng tham số hợp lý cực đại (MLE):
    # mu_c: vector kỳ vọng từng thuộc tính với lớp c
    # std_c: vector độ lệch chuẩn từng thuộc tính với lớp c
    # pi_c: xác suất tiên nghiệm của lớp c
    means = np.zeros((n_classes, n_features))
    stds = np.zeros((n_classes, n_features))
    priors = np.zeros(n_classes)

    for c in range(n_classes):
        X_c = X_train[y_train == c]
        means[c, :] = np.mean(X_c, axis=0)
        # Sử dụng ddof=0 hoặc ddof=1 (ở đây dùng ddof=0 chuẩn MLE)
        stds[c, :] = np.std(X_c, axis=0, ddof=0)
        priors[c] = len(X_c) / len(X_train)

    print("Kỳ vọng (mu) từng lớp:\n", pd.DataFrame(means, index=unique_classes, columns=feature_cols))
    print("\nĐộ lệch chuẩn (std) từng lớp:\n", pd.DataFrame(stds, index=unique_classes, columns=feature_cols))
    print("\nXác suất tiên nghiệm (prior pi):", {unique_classes[i]: priors[i] for i in range(n_classes)})

    def predict_scratch(X_data):
        """Hàm dự đoán lớp theo Log-Likelihood để tránh tràn số thực (numerical underflow)"""
        preds = []
        for x in X_data:
            posteriors = []
            for c in range(n_classes):
                log_prior = np.log(priors[c])
                # Tổng log-likelihood trên 4 đặc trưng độc lập
                log_likelihood = np.sum(norm.logpdf(x, loc=means[c], scale=stds[c]))
                posteriors.append(log_prior + log_likelihood)
            preds.append(np.argmax(posteriors))
        return np.array(preds)

    y_pred_scratch = predict_scratch(X_test)

    # ==============================================================
    # PHƯƠNG PHÁP 2: SỬ DỤNG THƯ VIỆN SCIKIT-LEARN (GaussianNB)
    # ==============================================================
    print("\n--- 2. HUẤN LUYỆN BẰNG SCIKIT-LEARN (GaussianNB) ---")
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred_sk = gnb.predict(X_test)

    # ==============================================================
    # 3. ĐÁNH GIÁ VÀ ĐỐI CHIẾU KẾT QUẢ
    # ==============================================================
    print("\n--- 3. SO SÁNH VÀ ĐÁNH GIÁ CHỈ SỐ ---")
    acc_scratch = accuracy_score(y_test, y_pred_scratch)
    acc_sk = accuracy_score(y_test, y_pred_sk)
    match_rate = np.mean(y_pred_scratch == y_pred_sk) * 100

    print(f"-> Độ chính xác Scratch:     {acc_scratch * 100:.2f}%")
    print(f"-> Độ chính xác Scikit-Learn: {acc_sk * 100:.2f}%")
    print(f"-> Mức độ tương đồng giữa 2 cách: {match_rate:.2f}%")

    # Các chỉ số đa lớp (Macro & Weighted)
    prec_macro = precision_score(y_test, y_pred_sk, average='macro')
    rec_macro = recall_score(y_test, y_pred_sk, average='macro')
    f1_macro = f1_score(y_test, y_pred_sk, average='macro')

    prec_weighted = precision_score(y_test, y_pred_sk, average='weighted')
    rec_weighted = recall_score(y_test, y_pred_sk, average='weighted')
    f1_weighted = f1_score(y_test, y_pred_sk, average='weighted')

    print("\nBảng tổng hợp chỉ số đo lường (Tập Test - 30 mẫu):")
    metrics_df = pd.DataFrame({
        'Độ đo': ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)',
                  'Precision (Weighted)', 'Recall (Weighted)', 'F1-Score (Weighted)'],
        'Giá trị': [acc_sk, prec_macro, rec_macro, f1_macro, prec_weighted, rec_weighted, f1_weighted]
    })
    print(metrics_df.to_string(index=False))

    cm = confusion_matrix(y_test, y_pred_sk)
    print("\nMa trận nhầm lẫn (Confusion Matrix):")
    cm_df = pd.DataFrame(cm, index=[f"Thực tế: {c}" for c in unique_classes],
                         columns=[f"Dự đoán: {c}" for c in unique_classes])
    print(cm_df)

    print("\nBáo cáo chi tiết từng lớp (Classification Report):")
    print(classification_report(y_test, y_pred_sk, target_names=unique_classes, digits=4))

    # Vẽ và lưu biểu đồ Confusion Matrix
    save_fig_dir = os.path.dirname(__file__)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=unique_classes, yticklabels=unique_classes)
    plt.title('Ma trận nhầm lẫn - Iris 4 đặc trưng (Gaussian NB)')
    plt.xlabel('Nhãn dự đoán')
    plt.ylabel('Nhãn thực tế')
    plt.tight_layout()
    cm_path = os.path.join(save_fig_dir, "bai1_iris_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[*] Đã lưu biểu đồ ma trận nhầm lẫn tại: {cm_path}")

    return {
        'acc_scratch': acc_scratch,
        'acc_sk': acc_sk,
        'prec_macro': prec_macro,
        'rec_macro': rec_macro,
        'f1_macro': f1_macro,
        'cm': cm
    }

if __name__ == "__main__":
    run_bai1()
