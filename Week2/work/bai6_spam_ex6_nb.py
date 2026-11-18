"""
Bài tập bổ sung (Ví dụ 3 trong giáo trình): Phân loại Email Spam - Not Spam (Ling-Spam / ex6DataPrepared)
Họ và tên: Nguyễn Hải Anh - MSSV: 24001968
Mô hình: Multinomial Naive Bayes & Bernoulli Naive Bayes
Tập dữ liệu: ex6DataPrepared (2500 từ vựng, 700 train, 260 test)
"""

import os
import sys
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def get_data_dir():
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "ex6DataPrepared"),
        "ex6DataPrepared"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("Không tìm thấy thư mục ex6DataPrepared!")

def read_data(path, data_fn, label_fn, nwords=2500):
    with open(os.path.join(path, label_fn), 'r', encoding='utf-8') as f:
        labels = [int(line.strip()) for line in f if line.strip()]

    with open(os.path.join(path, data_fn), 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    dat = np.zeros((len(lines), 3), dtype=int)
    for i, line in enumerate(lines):
        parts = line.split()
        dat[i, :] = [int(parts[0]), int(parts[1]), int(parts[2])]

    # 1-indexed -> 0-indexed trong python
    # dat[:, 0] - 1: email index, dat[:, 1] - 1: word index, dat[:, 2]: count
    matrix = coo_matrix(
        (dat[:, 2], (dat[:, 0] - 1, dat[:, 1] - 1)),
        shape=(len(labels), nwords),
        dtype=int
    )
    return matrix, np.array(labels)

def run_bai6():
    print("=" * 70)
    print(" BÀI TẬP BỔ SUNG: PHÂN LOẠI EMAIL SPAM - LING-SPAM (ex6DataPrepared)")
    print("=" * 70)

    data_dir = get_data_dir()
    print(f"Thư mục dữ liệu: {data_dir}")

    # Đọc tập Full Train (700 mẫu) và Test (260 mẫu)
    train_data, train_label = read_data(data_dir, 'train-features.txt', 'train-labels.txt')
    test_data, test_label = read_data(data_dir, 'test-features.txt', 'test-labels.txt')

    print(f"Kích thước ma trận Train (Full): {train_data.shape} ({train_data.nnz} phần tử phi 0)")
    print(f"Kích thước ma trận Test:       {test_data.shape} ({test_data.nnz} phần tử phi 0)")
    print(f"Tỉ lệ nhãn Train (0: Ham, 1: Spam): {np.bincount(train_label)}")
    print(f"Tỉ lệ nhãn Test  (0: Ham, 1: Spam): {np.bincount(test_label)}")

    # 1. Multinomial Naive Bayes (Bag of Words tần suất từ)
    mnb = MultinomialNB(alpha=1.0)
    mnb.fit(train_data, train_label)
    y_pred_mnb = mnb.predict(test_data)
    acc_mnb = accuracy_score(test_label, y_pred_mnb)
    prec_mnb = precision_score(test_label, y_pred_mnb, pos_label=1)
    rec_mnb = recall_score(test_label, y_pred_mnb, pos_label=1)
    f1_mnb = f1_score(test_label, y_pred_mnb, pos_label=1)

    print("\n--- 1. MULTINOMIAL NAIVE BAYES (Full Train 700) ---")
    print(f"Độ chính xác (Accuracy):  {acc_mnb * 100:.2f}%")
    print(f"Precision (Spam = 1):     {prec_mnb * 100:.2f}%")
    print(f"Recall (Spam = 1):        {rec_mnb * 100:.2f}%")
    print(f"F1-Score (Spam = 1):      {f1_mnb * 100:.2f}%")
    print("Confusion Matrix:\n", confusion_matrix(test_label, y_pred_mnb))

    # 2. Bernoulli Naive Bayes (Nhị phân: Có xuất hiện hay không)
    # Convert sang nhị phân
    train_data_bin = (train_data > 0).astype(int)
    test_data_bin = (test_data > 0).astype(int)

    bnb = BernoulliNB(alpha=1.0)
    bnb.fit(train_data_bin, train_label)
    y_pred_bnb = bnb.predict(test_data_bin)
    acc_bnb = accuracy_score(test_label, y_pred_bnb)
    prec_bnb = precision_score(test_label, y_pred_bnb, pos_label=1)
    rec_bnb = recall_score(test_label, y_pred_bnb, pos_label=1)
    f1_bnb = f1_score(test_label, y_pred_bnb, pos_label=1)

    print("\n--- 2. BERNOULLI NAIVE BAYES (Full Train 700) ---")
    print(f"Độ chính xác (Accuracy):  {acc_bnb * 100:.2f}%")
    print(f"Precision (Spam = 1):     {prec_bnb * 100:.2f}%")
    print(f"Recall (Spam = 1):        {rec_bnb * 100:.2f}%")
    print(f"F1-Score (Spam = 1):      {f1_bnb * 100:.2f}%")
    print("Confusion Matrix:\n", confusion_matrix(test_label, y_pred_bnb))

    # 3. Khảo sát kích thước tập huấn luyện (Learning Curve: 50, 100, 400, 700)
    print("\n--- 3. KHẢO SÁT ẢNH HƯỞNG CỦA KÍCH THƯỚC TẬP HUẤN LUYỆN (MultinomialNB) ---")
    size_results = {}
    for size in [50, 100, 400]:
        feat_fn = f"train-features-{size}.txt"
        lbl_fn = f"train-labels-{size}.txt"
        sub_train_data, sub_train_label = read_data(data_dir, feat_fn, lbl_fn)
        clf = MultinomialNB(alpha=1.0)
        clf.fit(sub_train_data, sub_train_label)
        preds = clf.predict(test_data)
        acc = accuracy_score(test_label, preds)
        size_results[size] = acc
        print(f"- Tập Train {size:3d} mẫu: Accuracy = {acc * 100:.2f}%")
    size_results[700] = acc_mnb
    print(f"- Tập Train 700 mẫu: Accuracy = {acc_mnb * 100:.2f}%")

    return {
        'acc_mnb': acc_mnb,
        'prec_mnb': prec_mnb,
        'rec_mnb': rec_mnb,
        'f1_mnb': f1_mnb,
        'acc_bnb': acc_bnb,
        'size_results': size_results
    }

if __name__ == '__main__':
    run_bai6()
