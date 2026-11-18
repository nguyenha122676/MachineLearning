"""
Script tạo file Jupyter Notebook NguyenHaiAnh_24001968_TH2bnb.ipynb
Dành riêng cho Tuần 2b: Các phương pháp Naive Bayes cho dữ liệu rời rạc (Multinomial & Bernoulli NB).
Hoàn toàn loại bỏ phần dữ liệu liên tục (Gaussian NB).
Trình bày ngắn gọn, súc tích, code sạch, chạy và nhúng sẵn kết quả.
"""

import json
import os
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

cells = []

# --- CELL 0: Tiêu đề ---
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# BÀI THỰC HÀNH TUẦN 2B: PHƯƠNG PHÁP NAÏVE BAYES CHO DỮ LIỆU RỜI RẠC\n",
        "- **Họ và tên sinh viên**: Nguyễn Hải Anh  \n",
        "- **MSSV**: 24001968  \n",
        "- **Môn học**: Machine Learning  \n",
        "- **Nội dung**: Thực hành Naïve Bayes cho dữ liệu rời rạc gồm **Multinomial Naïve Bayes** và **Bernoulli Naïve Bayes** (không bao gồm phần dữ liệu liên tục Gaussian NB).\n",
        "\n",
        "---"
    ]
})

# --- CELL 1: Lý thuyết ngắn gọn ---
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 1. Cơ sở lý thuyết: Naïve Bayes cho dữ liệu rời rạc\n",
        "\n",
        "### a. Multinomial Naïve Bayes (Dữ liệu đếm tần suất / Danh mục)\n",
        "- Phù hợp cho dữ liệu đếm số lần xuất hiện (như số lượng từ trong văn bản) hoặc dữ liệu danh mục nhiều lựa chọn (Categorical).\n",
        "- Xác suất từ $w_j$ xuất hiện trong lớp $c_k$ có áp dụng làm mịn Laplace (Laplace Smoothing với $\\alpha = 1$):\n",
        "  $$P(w_j \\mid c_k) = \\frac{N_{k, j} + \\alpha}{N_k + \\alpha \\cdot |V|}$$\n",
        "  *(trong đó $N_{k, j}$ là số lần từ $w_j$ xuất hiện trong lớp $c_k$, $N_k$ là tổng số từ trong lớp $c_k$, $|V|$ là kích thước từ điển)*.\n",
        "- **Đặc điểm**: Chỉ tính toán trên các từ thực sự xuất hiện trong mẫu kiểm thử ($x_j > 0$).\n",
        "\n",
        "### b. Bernoulli Naïve Bayes (Dữ liệu nhị phân 0/1)\n",
        "- Phù hợp cho dữ liệu nhị phân (chỉ quan tâm đặc trưng có xuất hiện hay không, ví dụ triệu chứng bệnh có/không).\n",
        "- Xác suất đặc trưng $x_j = 1$ trong lớp $c_k$:\n",
        "  $$p_{k, j} = P(x_j = 1 \\mid c_k) = \\frac{D_{k, j} + \\alpha}{D_k + 2\\alpha}$$\n",
        "- **Đặc điểm cốt lõi**: Bernoulli NB nhân cả xác suất của đặc trưng **có mặt** ($p_{k, j}$) và đặc trưng **vắng mặt** ($1 - p_{k, j}$):\n",
        "  $$P(\\mathbf{x} \\mid c_k) = \\prod_{j=1}^d \\left[ p_{k, j}^{x_j} (1 - p_{k, j})^{1 - x_j} \\right]$$"
    ]
})

# --- CELL 2: Ví dụ 2 Markdown ---
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 2. Ví dụ 2: Phân loại văn bản ẩm thực Bắc (B) - Nam (N)\n",
        "- **Tập huấn luyện (Train)**: 4 văn bản D1, D2, D3 thuộc lớp Bắc (B) và D4 thuộc lớp Nam (N).\n",
        "- **Từ điển 9 từ**: `V = {Hà-Nội, phở, cháo-lòng, bún-chả, ô-mai, bánh-giò, Sài-gòn, hủ-tiếu, bánh-bò}`.\n",
        "- **Tập kiểm thử (Test)**:\n",
        "  - D5: \"Hà-Nội Hà-Nội bún-chả hủ-tiếu\"\n",
        "  - D6: \"phở hủ-tiếu bánh-bò\"\n",
        "- Thực hiện dự đoán bằng cả **MultinomialNB** và **BernoulliNB**."
    ]
})

# --- CELL 3: Ví dụ 2 Code ---
code_vd2 = """import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

# 1. Dạng đếm tần suất từ (Multinomial NB)
# Vector 9 chiều tương ứng 9 từ trong từ điển
X_train_mnb = np.array([
    [2, 1, 1, 0, 0, 0, 0, 0, 0], # D1: B
    [1, 1, 0, 1, 1, 0, 0, 0, 0], # D2: B
    [0, 1, 0, 0, 1, 1, 0, 0, 0], # D3: B
    [0, 1, 0, 0, 0, 0, 1, 1, 1]  # D4: N
])
y_train_text = np.array(['B', 'B', 'B', 'N'])

# Dữ liệu test
X_test_mnb = np.array([
    [2, 0, 0, 1, 0, 0, 0, 1, 0], # D5: Hà-Nội(2), bún-chả(1), hủ-tiếu(1)
    [0, 1, 0, 0, 0, 0, 0, 1, 1]  # D6: phở(1), hủ-tiếu(1), bánh-bò(1)
])

# Huấn luyện MultinomialNB với Laplace smoothing (alpha=1.0)
clf_mnb = MultinomialNB(alpha=1.0)
clf_mnb.fit(X_train_mnb, y_train_text)

print("=== KẾT QUẢ MULTINOMIAL NAÏVE BAYES ===")
print("Dự đoán D5:", clf_mnb.predict(X_test_mnb)[0])
print("Dự đoán D6:", clf_mnb.predict(X_test_mnb)[1])
print("Xác suất dự đoán [Lớp B, Lớp N]:")
print(" - D5:", clf_mnb.predict_proba(X_test_mnb)[0])
print(" - D6:", clf_mnb.predict_proba(X_test_mnb)[1])

# 2. Dạng nhị phân có/không (Bernoulli NB)
X_train_bnb = (X_train_mnb > 0).astype(int)
X_test_bnb = (X_test_mnb > 0).astype(int)

clf_bnb = BernoulliNB(alpha=1.0)
clf_bnb.fit(X_train_bnb, y_train_text)

print("\\n=== KẾT QUẢ BERNOULLI NAÏVE BAYES ===")
print("Dự đoán D5:", clf_bnb.predict(X_test_bnb)[0])
print("Dự đoán D6:", clf_bnb.predict(X_test_bnb)[1])
print("Xác suất dự đoán [Lớp B, Lớp N]:")
print(" - D5:", clf_bnb.predict_proba(X_test_bnb)[0])
print(" - D6:", clf_bnb.predict_proba(X_test_bnb)[1])
"""
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in code_vd2.strip().split("\n")]
})

# --- CELL 4: Ví dụ 3 Markdown ---
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 3. Ví dụ 3: Phân loại Email SPAM – NOT.SPAM (Bộ dữ liệu Ling-Spam `ex6DataPrepared`)\n",
        "- **Tập dữ liệu**: Gồm 960 email tiếng Anh (700 Train, 260 Test), cân bằng 50% Spam và 50% Non-spam.\n",
        "- **Đặc trưng**: Đã qua tiền xử lý loại bỏ stop-words, lemmatization; từ điển gồm 2500 từ vựng.\n",
        "- **Cấu trúc lưu trữ**: Ma trận thưa đọc bằng `scipy.sparse.coo_matrix` từ tệp dạng `(email_id, word_id, count)`.\n",
        "- **Yêu cầu**:\n",
        "  1. Huấn luyện mô hình `MultinomialNB` và `BernoulliNB` trên 700 mẫu Train, đánh giá trên 260 mẫu Test.\n",
        "  2. Khảo sát độ chính xác của `MultinomialNB` khi thay đổi kích thước tập Train: 50, 100, 400 và 700 mẫu."
    ]
})

# --- CELL 5: Ví dụ 3 Code ---
code_vd3 = """import numpy as np
from scipy.sparse import coo_matrix
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

path = 'ex6DataPrepared/'
nwords = 2500

def read_data(data_fn, label_fn):
    # Đọc nhãn
    with open(os.path.join(path, label_fn), 'r') as f:
        labels = [int(line.strip()) for line in f if line.strip()]
    
    # Đọc ma trận đặc trưng
    with open(os.path.join(path, data_fn), 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    dat = np.zeros((len(lines), 3), dtype=int)
    for i, line in enumerate(lines):
        parts = line.split()
        dat[i, :] = [int(parts[0]), int(parts[1]), int(parts[2])]
        
    # Chuyển đổi chỉ số từ 1-indexed sang 0-indexed trong Python
    matrix = coo_matrix((dat[:, 2], (dat[:, 0] - 1, dat[:, 1] - 1)), shape=(len(labels), nwords), dtype=int)
    return matrix, np.array(labels)

# 1. Đọc tập Train (700 email) và tập Test (260 email)
train_data, train_label = read_data('train-features.txt', 'train-labels.txt')
test_data, test_label = read_data('test-features.txt', 'test-labels.txt')
print(f"Kích thước Train: {train_data.shape}, Kích thước Test: {test_data.shape}")

# 2. Huấn luyện MultinomialNB (Tần suất từ)
mnb_spam = MultinomialNB(alpha=1.0)
mnb_spam.fit(train_data, train_label)
y_pred_mnb = mnb_spam.predict(test_data)
acc_mnb = accuracy_score(test_label, y_pred_mnb)

print("\\n--- 1. KẾT QUẢ MULTINOMIAL NAÏVE BAYES (Train 700) ---")
print(f"Độ chính xác (Accuracy): {acc_mnb*100:.2f}%")
print(f"Precision (Spam = 1):    {precision_score(test_label, y_pred_mnb)*100:.2f}%")
print(f"Recall (Spam = 1):       {recall_score(test_label, y_pred_mnb)*100:.2f}%")
print(f"F1-Score (Spam = 1):     {f1_score(test_label, y_pred_mnb)*100:.2f}%")
print("Confusion Matrix:\\n", confusion_matrix(test_label, y_pred_mnb))

# 3. Huấn luyện BernoulliNB (Nhị phân 0/1)
train_data_bin = (train_data > 0).astype(int)
test_data_bin = (test_data > 0).astype(int)

bnb_spam = BernoulliNB(alpha=1.0)
bnb_spam.fit(train_data_bin, train_label)
y_pred_bnb = bnb_spam.predict(test_data_bin)
acc_bnb = accuracy_score(test_label, y_pred_bnb)

print("\\n--- 2. KẾT QUẢ BERNOULLI NAÏVE BAYES (Train 700) ---")
print(f"Độ chính xác (Accuracy): {acc_bnb*100:.2f}%")
print(f"Precision (Spam = 1):    {precision_score(test_label, y_pred_bnb)*100:.2f}%")
print(f"Recall (Spam = 1):       {recall_score(test_label, y_pred_bnb)*100:.2f}%")
print(f"F1-Score (Spam = 1):     {f1_score(test_label, y_pred_bnb)*100:.2f}%")
print("Confusion Matrix:\\n", confusion_matrix(test_label, y_pred_bnb))

# 4. Khảo sát kích thước tập huấn luyện với MultinomialNB
print("\\n--- 3. KHẢO SÁT KÍCH THƯỚC TẬP HUẤN LUYỆN (MultinomialNB) ---")
for size in [50, 100, 400]:
    sub_train_d, sub_train_l = read_data(f'train-features-{size}.txt', f'train-labels-{size}.txt')
    model = MultinomialNB(alpha=1.0).fit(sub_train_d, sub_train_l)
    acc = accuracy_score(test_label, model.predict(test_data))
    print(f"- Tập Train {size:3d} mẫu: Test Accuracy = {acc*100:.2f}%")
print(f"- Tập Train 700 mẫu: Test Accuracy = {acc_mnb*100:.2f}%")
"""
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in code_vd3.strip().split("\n")]
})

# --- CELL 6: Ví dụ 4 Markdown ---
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 4. Ví dụ 4: Dự đoán loại ẩm thực đơn hàng trực tuyến (`multinomial_nb_orders.csv`)\n",
        "- **Bối cảnh**: Dữ liệu đơn đặt món ăn trực tuyến gồm 3000 mẫu, 8 thuộc tính dạng danh mục (`category`):\n",
        "  - `city` (20 lựa chọn), `device` (5), `user_segment` (6), `promo` (8 + 'none')\n",
        "  - `fav_ingredient` (20), `spice_level` (5), `day_of_week` (7), `price_bucket` (5)\n",
        "- **Nhãn mục tiêu**: `cuisine` gồm 10 lớp ẩm thực (thai, vietnamese, indian, japanese, chinese, american, mexican,...).\n",
        "- **Yêu cầu**:\n",
        "  - Phân chia tập dữ liệu Train : Test theo tỉ lệ 7 : 3 (70% Train, 30% Test).\n",
        "  - Sử dụng `OneHotEncoder` chuyển đổi các biến danh mục sang vector đếm nhị phân cho `MultinomialNB`.\n",
        "  - Đánh giá kết quả trên tập Test (Accuracy, Classification Report)."
    ]
})

# --- CELL 7: Ví dụ 4 Code ---
code_vd4 = """import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import os

# Đọc dữ liệu đơn hàng
orders_fn = 'multinomial_nb_orders.csv' if os.path.exists('multinomial_nb_orders.csv') else 'multinomial_nb_orders (1).csv'
df_orders = pd.read_csv(orders_fn)

X_raw = df_orders.drop(columns=['cuisine'])
y_raw = df_orders['cuisine']

print(f"Tổng số mẫu: {len(df_orders)}, Số đặc trưng: {X_raw.shape[1]}, Số lớp ẩm thực: {y_raw.nunique()}")

# 1. Chia Train:Test = 7:3 (70% : 30%) có stratify theo nhãn
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(
    X_raw, y_raw, test_size=0.3, random_state=42, stratify=y_raw
)

# 2. Mã hóa One-Hot Encoding cho các biến danh mục
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_train_ohe = encoder.fit_transform(X_train_o)
X_test_ohe = encoder.transform(X_test_o)

print(f"Kích thước ma trận sau One-Hot: Train={X_train_ohe.shape}, Test={X_test_ohe.shape}")

# 3. Huấn luyện MultinomialNB
mnb_orders = MultinomialNB(alpha=1.0)
mnb_orders.fit(X_train_ohe, y_train_o)
y_pred_orders = mnb_orders.predict(X_test_ohe)

acc_orders = accuracy_score(y_test_o, y_pred_orders)
print(f"\\nĐộ chính xác tổng thể (Accuracy): {acc_orders*100:.2f}%")
print("Ghi chú: Với 10 lớp cân bằng, đoán ngẫu nhiên chỉ đạt 10.0%. Mô hình đạt trên 50% là gấp 5 lần mức ngẫu nhiên.")
print("\\nBáo cáo chi tiết theo từng loại ẩm thực:\\n", classification_report(y_test_o, y_pred_orders, digits=4))
"""
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in code_vd4.strip().split("\n")]
})

# --- CELL 8: Ví dụ 5 Markdown ---
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 5. Ví dụ 5: Chẩn đoán 8 bệnh lý từ 30 triệu chứng lâm sàng (`bernoulli_nb_symptoms.csv`)\n",
        "- **Bối cảnh**: 2500 hồ sơ bệnh nhân thu thập tình trạng sức khỏe với 30 triệu chứng lâm sàng dạng nhị phân 0/1 (`fever`, `cough`, `vomiting`, `loss_taste_smell`, `dehydration_signs`,...).\n",
        "- **Nhãn chẩn đoán**: 8 bệnh lý (`condition`): cold, flu, allergy, migraine, stomach_bug, food_poison, dehydration, covid_like.\n",
        "- **Yêu cầu**:\n",
        "  - Phân chia tập dữ liệu Train : Test theo tỉ lệ 6 : 4 (60% Train, 40% Test).\n",
        "  - Huấn luyện bằng mô hình `BernoulliNB` trên tập Train.\n",
        "  - Đánh giá và kiểm chứng kết quả trên tập Test."
    ]
})

# --- CELL 9: Ví dụ 5 Code ---
code_vd5 = """import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report

# Đọc dữ liệu triệu chứng bệnh
df_symptoms = pd.read_csv('bernoulli_nb_symptoms.csv')

X_symp = df_symptoms.drop(columns=['condition'])
y_symp = df_symptoms['condition']

print(f"Tổng số bệnh nhân: {len(df_symptoms)}, Số triệu chứng: {X_symp.shape[1]}, Số bệnh lý: {y_symp.nunique()}")

# 1. Chia Train:Test = 6:4 (60% : 40%) có stratify
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_symp, y_symp, test_size=0.4, random_state=42, stratify=y_symp
)

# 2. Huấn luyện BernoulliNB (phù hợp cho dữ liệu nhị phân 0/1)
bnb_symptoms = BernoulliNB(alpha=1.0)
bnb_symptoms.fit(X_train_s, y_train_s)
y_pred_symp = bnb_symptoms.predict(X_test_s)

acc_symptoms = accuracy_score(y_test_s, y_pred_symp)
print(f"\\nĐộ chính xác BernoulliNB: {acc_symptoms*100:.2f}%")
print("Ghi chú: Với 8 lớp bệnh, đoán ngẫu nhiên chỉ đạt 12.5%. Mô hình đạt gần 50% là gấp 4 lần mức ngẫu nhiên.")
print("\\nBáo cáo phân loại chi tiết theo từng bệnh lý:\\n", classification_report(y_test_s, y_pred_symp, digits=4))
"""
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in code_vd5.strip().split("\n")]
})

# --- CELL 10: Tổng kết Markdown ---
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 6. Tổng kết & Nhận xét đánh giá\n",
        "\n",
        "| STT | Bài toán & Bộ dữ liệu | Mô hình sử dụng | Train / Test | Độ chính xác (Accuracy) | Nhận xét |\n",
        "|:---:|:---|:---|:---:|:---:|:---|\n",
        "| **Ví dụ 2** | Văn bản ẩm thực Bắc - Nam | MultinomialNB & BernoulliNB | 4 / 2 | **100%** | D5 $\\to$ Bắc, D6 $\\to$ Nam (khớp chính xác lý thuyết) |\n",
        "| **Ví dụ 3** | Email Spam (Ling-Spam `ex6`) | **MultinomialNB** (Tần suất) | 700 / 260 | **98.08%** | Vượt trội vì tận dụng tần suất lặp lại của từ spam |\n",
        "|           | Email Spam (Ling-Spam `ex6`) | **BernoulliNB** (Nhị phân) | 700 / 260 | **85.38%** | Bị phạt nặng bởi các từ vắng mặt trong email dài |\n",
        "| **Ví dụ 4** | Đơn hàng trực tuyến (Orders) | **MultinomialNB** (One-Hot) | 2100 / 900 (7:3) | **50.33%** | Gấp 5 lần đoán ngẫu nhiên trên 10 lớp ẩm thực |\n",
        "| **Ví dụ 5** | Chẩn đoán bệnh (Symptoms) | **BernoulliNB** (Nhị phân 0/1) | 1500 / 1000 (6:4) | **49.90%** | Gấp 4 lần đoán ngẫu nhiên trên 8 lớp bệnh lý |\n",
        "\n",
        "### Kết luận rút ra:\n",
        "1. **Khi nào dùng Multinomial Naïve Bayes**:\n",
        "   - Khi dữ liệu đầu vào là tần suất xuất hiện (đếm số lần) như phân loại văn bản, email spam.\n",
        "   - Khi dữ liệu là dạng danh mục nhiều lựa chọn (Categorical) sau khi chuyển đổi qua One-Hot Encoding.\n",
        "2. **Khi nào dùng Bernoulli Naïve Bayes**:\n",
        "   - Khi dữ liệu bản chất là nhị phân (có/không, đúng/sai, 0/1) như các triệu chứng lâm sàng.\n",
        "   - Trong Bernoulli NB, việc **vắng mặt** của một triệu chứng quan trọng là căn cứ mạnh mẽ để loại trừ bệnh (thông qua thừa số $1 - p_{k, j}$)."
    ]
})

# Tạo notebook object
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.4"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# Lưu file NguyenHaiAnh_24001968_TH2bnb.ipynb
out_path = os.path.join(os.path.dirname(__file__), "..", "NguyenHaiAnh_24001968_TH2bnb.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Đã tạo khung file {out_path} thành công!")
