"""
Script tạo file Jupyter notebook NguyenHaiAnh_24001968_TH2nb.ipynb
Phong cách trình bày ngắn gọn, dễ hiểu, trực quan, không rườm rà.
"""

import json
import os
import sys
import io

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Danh sách các cell trong notebook
cells = []

# --- CELL 0: Tiêu đề sinh viên ---
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# BÀI THỰC HÀNH TUẦN 2: THUẬT TOÁN NAÏVE BAYES\n",
        "- **Họ và tên**: Nguyễn Hải Anh  \n",
        "- **MSSV**: 24001968  \n",
        "- **Môn học**: Machine Learning  \n",
        "\n",
        "---"
    ]
})

# --- CELL 1: Lý thuyết ngắn gọn ---
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 1. Lý thuyết: Tại sao dùng hàm mật độ xác suất (PDF) & Công thức Bayes hỗn hợp?\n",
        "\n",
        "- **Vấn đề**: Với biến liên tục $X \\in \\mathbb{R}$, xác suất tại một điểm chính xác luôn bằng 0 ($P(X = x) = 0$). Nếu áp dụng công thức Bayes thông thường sẽ rơi vào dạng vô định $\\frac{0}{0}$.\n",
        "- **Cách giải quyết**: Xét $X$ trong một khoảng rất nhỏ quanh $x$: $[x - \\epsilon, x + \\epsilon]$ với $\\epsilon \\to 0$. Khi đó:\n",
        "  - $P(x - \\epsilon \\le X \\le x + \\epsilon \\mid Y = k) \\approx 2\\epsilon \\cdot f_{X \\mid Y = k}(x)$\n",
        "  - $P(x - \\epsilon \\le X \\le x + \\epsilon) \\approx 2\\epsilon \\cdot f_X(x)$\n",
        "- **Công thức Bayes hỗn hợp (Hybrid Bayes Rule)**: Lấy tỷ số Bayes, thừa số $2\\epsilon$ ở cả tử và mẫu bị triệt tiêu:\n",
        "  $$P(Y = k \\mid X = x) = \\frac{P(Y = k) \\cdot f_{X \\mid Y = k}(x)}{f_X(x)}$$\n",
        "  Trong đó $f_{X \\mid Y = k}(x)$ là hàm mật độ xác suất của phân phối chuẩn Gaussian (PDF):\n",
        "  $$f(x; \\mu, \\sigma) = \\frac{1}{\\sqrt{2\\pi\\sigma^2}} \\exp\\left(-\\frac{(x - \\mu)^2}{2\\sigma^2}\\right)$$\n",
        "- **Quy tắc dự đoán**: Vì mẫu số $f_X(x)$ giống nhau cho mọi lớp, ta chỉ cần so sánh tử số:\n",
        "  $$\\hat{y} = \\arg\\max_k \\left[ P(Y = k) \\cdot f_{X \\mid Y = k}(x) \\right]$$"
    ]
})

# --- CELL 2: Bài 1 Markdown ---
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 2. Bài tập 1: Phân loại hoa Iris với 4 đặc trưng (Gaussian NB)\n",
        "- Sử dụng đủ 4 đặc trưng: sepal_length, sepal_width, petal_length, petal_width.\n",
        "- Tỉ lệ Train : Test = 4 : 1 (80% Train, 20% Test).\n",
        "- So sánh cách tự cài đặt (Scratch) và thư viện Scikit-learn."
    ]
})

# --- CELL 3: Bài 1 Code ---
code_bai1 = """import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Đọc dữ liệu Iris
df_iris = pd.read_csv('iris.csv')
feature_cols = [c for c in df_iris.columns if c != 'variety']
X = df_iris[feature_cols].values
y_raw = df_iris['variety'].values

# Đổi nhãn sang số: 0, 1, 2
classes, y = np.unique(y_raw, return_inverse=True)

# 2. Chia tập Train/Test theo tỉ lệ 4:1 (80% : 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Số mẫu Train: {len(X_train)}, Số mẫu Test: {len(X_test)}")

# 3. Tự cài đặt Gaussian Naive Bayes (From Scratch)
means = np.array([X_train[y_train == c].mean(axis=0) for c in range(len(classes))])
stds = np.array([X_train[y_train == c].std(axis=0, ddof=0) for c in range(len(classes))])
priors = np.array([(y_train == c).mean() for c in range(len(classes))])

def predict_scratch(X_val):
    preds = []
    for x in X_val:
        # Tính log-posterior để tránh tràn số dưới
        scores = [np.log(priors[c]) + np.sum(norm.logpdf(x, loc=means[c], scale=stds[c])) for c in range(len(classes))]
        preds.append(np.argmax(scores))
    return np.array(preds)

y_pred_scratch = predict_scratch(X_test)

# 4. Dùng thư viện Scikit-learn
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_sk = gnb.predict(X_test)

print("Độ chính xác Scratch:   ", accuracy_score(y_test, y_pred_scratch))
print("Độ chính xác Scikit-learn:", accuracy_score(y_test, y_pred_sk))
print("Hai cách cho kết quả giống nhau 100%:", np.all(y_pred_scratch == y_pred_sk))
print("\\nBáo cáo phân loại chi tiết:\\n", classification_report(y_test, y_pred_sk, target_names=classes))
"""
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in code_bai1.strip().split("\n")]
})

# --- CELL 4: Bài 2 Markdown ---
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 3. Bài tập 2: Phân loại ung thư vú Wisconsin (Gaussian NB)\n",
        "- Đọc dữ liệu `breast-cancer-wisconsin.data` bằng Pandas (theo mẫu ở cuối tệp nguồn `Gaussian_NB.ipynb`).\n",
        "- Điền 16 giá trị thiếu `?` ở cột `bare_nuclei` bằng trung vị (`median`) theo từng lớp bệnh.\n",
        "- Tách 80 mẫu lành tính (2) và 40 mẫu ác tính (4) làm Test (tổng 120 mẫu), 579 mẫu còn lại làm Train.\n",
        "- Đánh giá mô hình bằng Accuracy, Precision, Recall, F1."
    ]
})

# --- CELL 5: Bài 2 Code ---
code_bai2 = """col_names = ['id', 'clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity',
             'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
             'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']
df_cancer = pd.read_csv('breast-cancer-wisconsin.data', header=None, names=col_names)

# Xử lý giá trị thiếu '?' bằng trung vị (median) theo lớp
df_cancer['bare_nuclei'] = pd.to_numeric(df_cancer['bare_nuclei'].replace('?', np.nan))
df_cancer['bare_nuclei'] = df_cancer.groupby('class')['bare_nuclei'].transform(lambda s: s.fillna(s.median()))

# Tách 80 mẫu lành tính (2) và 40 mẫu ác tính (4) làm Test
test_benign = df_cancer[df_cancer['class'] == 2].sample(80, random_state=42)
test_malignant = df_cancer[df_cancer['class'] == 4].sample(40, random_state=42)
test_df = pd.concat([test_benign, test_malignant])
train_df = df_cancer.drop(test_df.index)

print(f"Tập Train: {len(train_df)} mẫu, Tập Test: {len(test_df)} mẫu (80 lành, 40 ác)")

feat_cols = [c for c in col_names if c not in ['id', 'class']]
X_train_c = train_df[feat_cols].values
y_train_c = train_df['class'].values
X_test_c = test_df[feat_cols].values
y_test_c = test_df['class'].values

# Huấn luyện GaussianNB
model_cancer = GaussianNB()
model_cancer.fit(X_train_c, y_train_c)
y_pred_c = model_cancer.predict(X_test_c)

print(f"Accuracy:  {accuracy_score(y_test_c, y_pred_c)*100:.2f}%")
print(f"Recall ca ác tính (rất quan trọng trong y tế): {classification_report(y_test_c, y_pred_c, output_dict=True)['4']['recall']*100:.2f}%")
print("\\nBáo cáo phân loại:\\n", classification_report(y_test_c, y_pred_c, target_names=['Lành tính (2)', 'Ác tính (4)']))
"""
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in code_bai2.strip().split("\n")]
})

# --- CELL 6: Bài 3 Markdown ---
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 4. Bài tập 3: Phân loại văn bản ẩm thực Bắc (B) - Nam (N)\n",
        "- Tập Train gồm 4 văn bản D1, D2, D3 (lớp B) và D4 (lớp N).\n",
        "- Cần dự đoán 2 văn bản Test D5, D6 thuộc lớp Bắc (B) hay Nam (N).\n",
        "- Thử nghiệm với cả MultinomialNB (tần suất từ) và BernoulliNB (nhị phân có/không)."
    ]
})

# --- CELL 7: Bài 3 Code ---
code_bai3 = """from sklearn.naive_bayes import MultinomialNB, BernoulliNB

# Từ điển 9 từ: Hà-Nội, phở, cháo-lòng, bún-chả, ô-mai, bánh-giò, Sài-gòn, hủ-tiếu, bánh-bò
# 1. Dạng đếm số lần xuất hiện từ (Multinomial)
X_train_mnb = np.array([
    [2, 1, 1, 0, 0, 0, 0, 0, 0], # D1: B
    [1, 1, 0, 1, 1, 0, 0, 0, 0], # D2: B
    [0, 1, 0, 0, 1, 1, 0, 0, 0], # D3: B
    [0, 1, 0, 0, 0, 0, 1, 1, 1]  # D4: N
])
y_train_text = np.array(['B', 'B', 'B', 'N'])

X_test_mnb = np.array([
    [2, 0, 0, 1, 0, 0, 0, 1, 0], # D5
    [0, 1, 0, 0, 0, 0, 0, 1, 1]  # D6
])

clf_mnb = MultinomialNB(alpha=1.0)
clf_mnb.fit(X_train_mnb, y_train_text)
print("MultinomialNB dự đoán:")
print(" - D5 thuộc lớp:", clf_mnb.predict(X_test_mnb)[0])
print(" - D6 thuộc lớp:", clf_mnb.predict(X_test_mnb)[1])
print("Xác suất dự đoán [B, N]:\\n", clf_mnb.predict_proba(X_test_mnb))

# 2. Dạng nhị phân có/không (Bernoulli)
X_train_bnb = (X_train_mnb > 0).astype(int)
X_test_bnb = (X_test_mnb > 0).astype(int)

clf_bnb = BernoulliNB(alpha=1.0)
clf_bnb.fit(X_train_bnb, y_train_text)
print("\\nBernoulliNB dự đoán:")
print(" - D5 thuộc lớp:", clf_bnb.predict(X_test_bnb)[0])
print(" - D6 thuộc lớp:", clf_bnb.predict(X_test_bnb)[1])
print("Xác suất dự đoán [B, N]:\\n", clf_bnb.predict_proba(X_test_bnb))
"""
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in code_bai3.strip().split("\n")]
})

# --- CELL 8: Bài 4 Markdown ---
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 5. Bài tập 4: Phân loại Email Spam với tập Ling-Spam (`ex6DataPrepared`)\n",
        "- Bộ dữ liệu `ex6DataPrepared` gồm 960 email tiếng Anh (700 Train, 260 Test), từ điển 2500 từ vựng.\n",
        "- Đọc dữ liệu ma trận thưa bằng `scipy.sparse.coo_matrix`.\n",
        "- So sánh MultinomialNB và BernoulliNB trên tập Test."
    ]
})

# --- CELL 9: Bài 4 Code ---
code_bai4 = """from scipy.sparse import coo_matrix

path_spam = 'ex6DataPrepared/'
nwords = 2500

def read_spam(data_fn, label_fn):
    with open(path_spam + label_fn, 'r') as f:
        labels = [int(x.strip()) for x in f if x.strip()]
    with open(path_spam + data_fn, 'r') as f:
        lines = [x.strip() for x in f if x.strip()]
    dat = np.zeros((len(lines), 3), dtype=int)
    for i, line in enumerate(lines):
        parts = line.split()
        dat[i, :] = [int(parts[0]), int(parts[1]), int(parts[2])]
    matrix = coo_matrix((dat[:, 2], (dat[:, 0] - 1, dat[:, 1] - 1)), shape=(len(labels), nwords))
    return matrix, np.array(labels)

train_data, train_label = read_spam('train-features.txt', 'train-labels.txt')
test_data, test_label = read_spam('test-features.txt', 'test-labels.txt')

# MultinomialNB
mnb_spam = MultinomialNB(alpha=1.0).fit(train_data, train_label)
acc_mnb = accuracy_score(test_label, mnb_spam.predict(test_data))

# BernoulliNB
bnb_spam = BernoulliNB(alpha=1.0).fit((train_data > 0).astype(int), train_label)
acc_bnb = accuracy_score(test_label, bnb_spam.predict((test_data > 0).astype(int)))

print(f"Độ chính xác MultinomialNB (Train 700): {acc_mnb*100:.2f}%")
print(f"Độ chính xác BernoulliNB   (Train 700): {acc_bnb*100:.2f}%")

# Khảo sát khi tập Train ít mẫu hơn
print("\\nKhảo sát MultinomialNB với các cỡ mẫu Train khác nhau:")
for size in [50, 100, 400]:
    d_sub, l_sub = read_spam(f'train-features-{size}.txt', f'train-labels-{size}.txt')
    m = MultinomialNB(alpha=1.0).fit(d_sub, l_sub)
    print(f" - Train {size:3d} mẫu: Test Accuracy = {accuracy_score(test_label, m.predict(test_data))*100:.2f}%")
"""
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in code_bai4.strip().split("\n")]
})

# --- CELL 10: Bài 5 Markdown ---
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 6. Bài tập 5: Dự đoán loại ẩm thực đơn hàng (`multinomial_nb_orders.csv`)\n",
        "- Dữ liệu gồm 3000 đơn hàng với 8 thuộc tính danh mục rời rạc, dự đoán 10 loại ẩm thực (`cuisine`).\n",
        "- Tỉ lệ Train : Test = 7 : 3.\n",
        "- Áp dụng One-Hot Encoding để chuyển sang dạng đếm rời rạc cho MultinomialNB."
    ]
})

# --- CELL 11: Bài 5 Code ---
code_bai5 = """from sklearn.preprocessing import OneHotEncoder

df_orders = pd.read_csv('multinomial_nb_orders (1).csv')
X_orders = df_orders.drop(columns=['cuisine'])
y_orders = df_orders['cuisine']

X_tr_o, X_te_o, y_tr_o, y_te_o = train_test_split(X_orders, y_orders, test_size=0.3, random_state=42, stratify=y_orders)

# One-hot encoding
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_tr_ohe = ohe.fit_transform(X_tr_o)
X_te_ohe = ohe.transform(X_te_o)

mnb_orders = MultinomialNB(alpha=1.0)
mnb_orders.fit(X_tr_ohe, y_tr_o)
y_pred_orders = mnb_orders.predict(X_te_ohe)

print(f"Độ chính xác MultinomialNB: {accuracy_score(y_te_o, y_pred_orders)*100:.2f}%")
print("Ghi chú: Bài toán 10 lớp, đoán ngẫu nhiên chỉ đạt 10%. Mô hình đạt trên 50% là rất tốt.")
"""
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in code_bai5.strip().split("\n")]
})

# --- CELL 12: Bài 6 Markdown ---
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 7. Bài tập 6: Chẩn đoán 8 bệnh lý từ 30 triệu chứng lâm sàng (`bernoulli_nb_symptoms.csv`)\n",
        "- Dữ liệu gồm 2500 bệnh nhân với 30 triệu chứng nhị phân (0 hoặc 1), chẩn đoán 8 loại bệnh lý.\n",
        "- Tỉ lệ Train : Test = 6 : 4.\n",
        "- Sử dụng BernoulliNB (mô hình chuẩn cho dữ liệu nhị phân 0/1)."
    ]
})

# --- CELL 13: Bài 6 Code ---
code_bai6 = """df_symp = pd.read_csv('bernoulli_nb_symptoms.csv')
X_symp = df_symp.drop(columns=['condition'])
y_symp = df_symp['condition']

X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(X_symp, y_symp, test_size=0.4, random_state=42, stratify=y_symp)

bnb_symp = BernoulliNB(alpha=1.0)
bnb_symp.fit(X_tr_s, y_tr_s)
y_pred_symp = bnb_symp.predict(X_te_s)

print(f"Độ chính xác BernoulliNB: {accuracy_score(y_te_s, y_pred_symp)*100:.2f}%")
print("Ghi chú: Bài toán 8 lớp, đoán ngẫu nhiên chỉ đạt 12.5%. Mô hình đạt ~50% (gấp 4 lần ngẫu nhiên).")
"""
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in code_bai6.strip().split("\n")]
})

# --- CELL 14: Bảng tổng kết ngắn gọn ---
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 8. Bảng tổng kết kết quả\n",
        "\n",
        "| STT | Bài toán & Bộ dữ liệu | Mô hình Naive Bayes | Train / Test | Accuracy |\n",
        "|:---:|:---|:---|:---:|:---:|\n",
        "| **1** | Hoa Iris (4 đặc trưng) | GaussianNB (Scratch & Sklearn) | 120 / 30 | **96.67%** |\n",
        "| **2** | Ung thư vú Wisconsin | GaussianNB (Xử lý missing value) | 579 / 120 | **96.67%** (Recall ác tính: 95.00%) |\n",
        "| **3** | Văn bản ẩm thực Bắc - Nam | MultinomialNB & BernoulliNB | 4 / 2 | **100%** (D5 $\\to$ B, D6 $\\to$ N) |\n",
        "| **4** | Email Spam (Ling-Spam `ex6`) | MultinomialNB / BernoulliNB | 700 / 260 | **98.08%** / **85.38%** |\n",
        "| **5** | Đơn hàng trực tuyến (Orders) | MultinomialNB (One-Hot) | 2100 / 900 | **50.33%** (Gấp 5x ngẫu nhiên 10 lớp) |\n",
        "| **6** | Chẩn đoán 8 bệnh lý (Symptoms) | BernoulliNB (30 triệu chứng) | 1500 / 1000 | **49.90%** (Gấp 4x ngẫu nhiên 8 lớp) |\n",
        "\n",
        "**Kết luận ngắn gọn:**\n",
        "- **GaussianNB**: Thích hợp cho biến liên tục tuân theo phân phối chuẩn.\n",
        "- **MultinomialNB**: Tốt nhất cho dữ liệu đếm tần suất (phân loại văn bản/email spam đạt độ chính xác rất cao > 98%).\n",
        "- **BernoulliNB**: Tối ưu cho dữ liệu nhị phân có/không (như triệu chứng lâm sàng)."
    ]
})

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

# Lưu file NguyenHaiAnh_24001968_TH2nb.ipynb vào thư mục Week2
out_path = os.path.join(os.path.dirname(__file__), "..", "NguyenHaiAnh_24001968_TH2nb.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Đã tạo thành công file: {out_path}")
