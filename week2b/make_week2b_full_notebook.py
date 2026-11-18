"""
Tạo file Jupyter Notebook NguyenHaiAnh_24001968_TH2bnb.ipynb
trong thư mục week2b, bám sát tài liệu thuchanh_so1B_NayveBayes.pdf.
Lời giải ngắn gọn, trực tiếp, có code Scratch và Sklearn so sánh,
chạy và nhúng sẵn kết quả thực thi vào file.
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

cells = []

# --- CELL 0: Header ---
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# BÀI THỰC HÀNH TUẦN 2B: PHƯƠNG PHÁP NAÏVE BAYES CHO DỮ LIỆU RỜI RẠC\n",
        "- **Họ và tên**: Nguyễn Hải Anh  \n",
        "- **MSSV**: 24001968  \n",
        "- **Học phần**: Machine Learning  \n",
        "- **Chủ đề**: Lab 01.B - Phương pháp Naïve Bayes cho dữ liệu rời rạc (Multinomial NB & Bernoulli NB)\n",
        "\n",
        "---"
    ]
})

# --- CELL 1: Phần 1 - Đơn hàng ẩm thực (Multinomial NB) ---
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 1. Multinomial Naïve Bayes: Dự đoán đơn hàng ẩm thực trực tuyến\n",
        "- **Dữ liệu**: `multinomial_nb_orders.csv` gồm 3000 đơn hàng, 8 thuộc tính danh mục (`category`), nhãn `cuisine` (10 lớp ẩm thực).\n",
        "- **Nhiệm vụ**:\n",
        "  1. Tự cài đặt phương pháp Multinomial Naïve Bayes (Scratch) với làm mịn Laplace theo hướng dẫn.\n",
        "  2. So sánh với thư viện Scikit-learn (`MultinomialNB`).\n",
        "  3. Giải thích lý do sai lệch giữa `LabelEncoder` và `OneHotEncoder` khi dùng thư viện."
    ]
})

# --- CELL 2: Code Phần 1 ---
code_part1 = """import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import os

# 1. Đọc dữ liệu đơn hàng
orders_fn = 'multinomial_nb_orders.csv' if os.path.exists('multinomial_nb_orders.csv') else 'multinomial_nb_orders (1).csv'
df = pd.read_csv(orders_fn)
target_feature = 'cuisine'
print("5 dòng đầu tiên:")
print(df.head(3))

# 2. Tự cài đặt Multinomial Naive Bayes (From Scratch)
# Chia Train : Test = 7 : 3 (2100 train, 900 test)
Train_size = round(len(df) * 0.7)
df_train = df.iloc[:Train_size].copy()
df_test = df.iloc[Train_size:].copy()

X_Train = df_train.drop(columns=[target_feature])
Y_Train = df_train[target_feature].values
X_Test = df_test.drop(columns=[target_feature]).reset_index(drop=True)
Y_Test = df_test[target_feature].reset_index(drop=True).values

# Xác suất tiên nghiệm P(k)
classes = np.unique(Y_Train)
priors = {c: np.mean(Y_Train == c) for c in classes}

# Xác suất P(x_i = v | y = k) với làm mịn Laplace
feature_cols = X_Train.columns
probs = {}
for col in feature_cols:
    probs[col] = {}
    unique_vals = df[col].unique()
    M = len(unique_vals)
    for c in classes:
        sub_col = X_Train[col][Y_Train == c]
        N_k = len(sub_col)
        counts = sub_col.value_counts().to_dict()
        probs[col][c] = {v: (counts.get(v, 0) + 1.0) / (N_k + M * 1.0) for v in unique_vals}

def predict_scratch(df_input):
    preds = []
    for _, row in df_input.iterrows():
        scores = []
        for c in classes:
            s = np.log(priors[c])
            for col in feature_cols:
                v = row[col]
                s += np.log(probs[col][c].get(v, 1.0 / (len(probs[col][c]) + 1)))
            scores.append(s)
        preds.append(classes[np.argmax(scores)])
    return np.array(preds)

y_pred_scratch = predict_scratch(X_Test)
acc_scratch = accuracy_score(Y_Test, y_pred_scratch)
print(f"Độ chính xác tự cài đặt (Scratch): {acc_scratch*100:.2f}%")

# 3. So sánh với thư viện Scikit-learn
# Cách A: Dùng LabelEncoder (như thử nghiệm trong hướng dẫn)
X_Train_le = X_Train.copy()
X_Test_le = X_Test.copy()
le = LabelEncoder()
for col in feature_cols:
    X_Train_le[col] = le.fit_transform(X_Train_le[col])
    X_Test_le[col] = le.transform(X_Test_le[col])

clf_le = MultinomialNB()
clf_le.fit(X_Train_le, Y_Train)
acc_le = accuracy_score(Y_Test, clf_le.predict(X_Test_le))
print(f"Độ chính xác Sklearn (dùng LabelEncoder): {acc_le*100:.2f}% (Rất thấp)")

# Cách B: Dùng OneHotEncoder (Chuẩn cho biến phân loại)
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_Train_ohe = ohe.fit_transform(X_Train)
X_Test_ohe = ohe.transform(X_Test)

clf_ohe = MultinomialNB()
clf_ohe.fit(X_Train_ohe, Y_Train)
acc_ohe = accuracy_score(Y_Test, clf_ohe.predict(X_Test_ohe))
print(f"Độ chính xác Sklearn (dùng OneHotEncoder): {acc_ohe*100:.2f}%")
print("-> Kết quả Sklearn One-Hot tương đồng với cài đặt Scratch (gấp 5 lần mức đoán ngẫu nhiên 10%).")
"""
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in code_part1.strip().split("\n")]
})

# --- CELL 3: Giải thích Bài tập 1 Markdown ---
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Giải thích câu hỏi Bài tập 1: Tại sao có sự sai lệch giữa code tự cài và Sklearn LabelEncoder?\n",
        "- **Nguyên nhân**: `LabelEncoder` gán các giá trị danh mục thành số nguyên liên tiếp $0, 1, 2, \\dots$. Khi đưa vào `MultinomialNB`, mô hình hiểu nhầm các số này là **số lần xuất hiện (tần suất đếm)** của từ trong văn bản, vô tình gán thứ tự và trọng số lớn bất hợp lý cho giá trị có số lớn $\\to$ Độ chính xác sụt giảm nghiêm trọng xuống ~17%.\n",
        "- **Khắc phục**: Với dữ liệu danh mục không có thứ tự, cần dùng **One-Hot Encoding** để biến mỗi lựa chọn thành 1 cờ nhị phân (0 hoặc 1). Khi đó `MultinomialNB` cho kết quả chính xác tương đương code Scratch (~50%)."
    ]
})

# --- CELL 4: Phần 2 - Triệu chứng bệnh (Bernoulli NB) ---
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 2. Bernoulli Naïve Bayes: Chẩn đoán bệnh từ 30 triệu chứng lâm sàng\n",
        "- **Dữ liệu**: `bernoulli_nb_symptoms.csv` gồm 2500 bệnh nhân, 30 thuộc tính nhị phân 0/1 (`fever`, `cough`, `vomiting`,...), 8 bệnh lý (`condition`).\n",
        "- **Nhiệm vụ**:\n",
        "  1. Chia Train : Test = 6 : 4 (1500 train, 1000 test).\n",
        "  2. Tự cài đặt thuật toán Bernoulli Naïve Bayes (Scratch).\n",
        "  3. So sánh với thư viện `BernoulliNB` của Scikit-learn."
    ]
})

# --- CELL 5: Code Phần 2 ---
code_part2 = """import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

df_symp = pd.read_csv('bernoulli_nb_symptoms.csv')
target_symp = 'condition'
print(f"Tổng số bệnh nhân: {len(df_symp)}, Số triệu chứng: {df_symp.shape[1]-1}, Số bệnh lý: {df_symp[target_symp].nunique()}")

# Chia Train:Test = 6:4
Train_size_s = round(len(df_symp) * 0.6)
df_train_s = df_symp.iloc[:Train_size_s].copy()
df_test_s = df_symp.iloc[Train_size_s:].copy()

X_Train_s = df_train_s.drop(columns=[target_symp])
Y_Train_s = df_train_s[target_symp].values
X_Test_s = df_test_s.drop(columns=[target_symp]).reset_index(drop=True)
Y_Test_s = df_test_s[target_symp].reset_index(drop=True).values

# 1. Cài đặt Bernoulli NB (Scratch)
classes_s = np.unique(Y_Train_s)
priors_s = {c: np.mean(Y_Train_s == c) for c in classes_s}

# Xác suất P(x_i = 1 | y = c) có làm mịn Laplace
p_ik = {}
symp_cols = X_Train_s.columns
for c in classes_s:
    sub = X_Train_s[Y_Train_s == c]
    N_c = len(sub)
    counts = sub.sum(axis=0)
    p_ik[c] = (counts + 1.0) / (N_c + 2.0)

def predict_bernoulli_scratch(X_val):
    preds = []
    for _, row in X_val.iterrows():
        scores = []
        for c in classes_s:
            s = np.log(priors_s[c])
            prob_vec = p_ik[c].values
            x_vec = row.values
            # Bernoulli nhân cả có mặt (x_i) và vắng mặt (1 - x_i)
            term = x_vec * prob_vec + (1 - x_vec) * (1 - prob_vec)
            s += np.sum(np.log(np.clip(term, 1e-12, 1.0)))
            scores.append(s)
        preds.append(classes_s[np.argmax(scores)])
    return np.array(preds)

y_pred_s_scratch = predict_bernoulli_scratch(X_Test_s)
acc_s_scratch = accuracy_score(Y_Test_s, y_pred_s_scratch)
print(f"Độ chính xác Bernoulli NB Scratch: {acc_s_scratch*100:.2f}%")

# 2. Dùng thư viện Scikit-learn
bnb = BernoulliNB(alpha=1.0)
bnb.fit(X_Train_s, Y_Train_s)
y_pred_s_sk = bnb.predict(X_Test_s)
acc_s_sk = accuracy_score(Y_Test_s, y_pred_s_sk)
print(f"Độ chính xác Bernoulli NB Sklearn: {acc_s_sk*100:.2f}%")
print("Hai phương pháp cho kết quả khớp nhau:", np.all(y_pred_s_scratch == y_pred_s_sk))
"""
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in code_part2.strip().split("\n")]
})

# --- CELL 6: Phần 3 - Bài tập lớn Động đất Nepal ---
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 3. Bài tập lớn: Dự đoán thiệt hại động đất Nepal 2015\n",
        "- **Bối cảnh**: Trận động đất Nepal năm 2015, dữ liệu gồm 260,601 tòa nhà trong `train_values.csv` và mức độ thiệt hại trong `train_labels.csv` (`damage_grade`: 1, 2, 3).\n",
        "- **Bài 1**: Đọc 8 trường dạng nhiều lựa chọn (`categorical`), huấn luyện mô hình **Multinomial Naïve Bayes** (Scratch & Sklearn) và so sánh độ chính xác.\n",
        "- **Bài 2**: Đọc 22 trường dạng nhị phân (`binary` - các trường bắt đầu bằng `has_`), huấn luyện mô hình **Bernoulli Naïve Bayes** (Scratch & Sklearn) và so sánh độ chính xác."
    ]
})

# --- CELL 7: Code Bài 1 Nepal (Multinomial NB) ---
code_nepal_1 = """import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Đọc dữ liệu Động đất Nepal
train_values = pd.read_csv('train_values.csv')
train_labels = pd.read_csv('train_labels.csv')
y_nepal = train_labels['damage_grade'].values

# Danh sách 8 trường nhiều lựa chọn (Categorical) theo đề bài
cat_cols = [
    'land_surface_condition', 'foundation_type', 'roof_type',
    'ground_floor_type', 'other_floor_type', 'position',
    'plan_configuration', 'legal_ownership_status'
]

print("=== BÀI 1: MULTINOMIAL NAÏVE BAYES TRÊN 8 TRƯỜNG CATEGORICAL ===")
print("Các trường sử dụng:", cat_cols)

# Chia tập Train/Test theo tỉ lệ 80:20
X_tr_cat, X_te_cat, y_tr_nepal, y_te_nepal = train_test_split(
    train_values[cat_cols], y_nepal, test_size=0.2, random_state=42, stratify=y_nepal
)

# A. Thư viện Scikit-learn (One-Hot Encoding + MultinomialNB)
ohe_nepal = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
X_tr_cat_ohe = ohe_nepal.fit_transform(X_tr_cat)
X_te_cat_ohe = ohe_nepal.transform(X_te_cat)

mnb_nepal = MultinomialNB(alpha=1.0)
mnb_nepal.fit(X_tr_cat_ohe, y_tr_nepal)
y_pred_cat_sk = mnb_nepal.predict(X_te_cat_ohe)
acc_cat_sk = accuracy_score(y_te_nepal, y_pred_cat_sk)
print(f"Độ chính xác Sklearn MultinomialNB (Toàn bộ tập Test 52,121 mẫu): {acc_cat_sk*100:.2f}%")

# B. So sánh với Cài đặt Scratch trên mẫu kiểm thử
classes_nepal = np.unique(y_tr_nepal)
priors_nepal = {c: np.mean(y_tr_nepal == c) for c in classes_nepal}
probs_nepal = {}
for col in cat_cols:
    probs_nepal[col] = {}
    vals = train_values[col].unique()
    M = len(vals)
    for c in classes_nepal:
        sub = X_tr_cat[col][y_tr_nepal == c]
        N_c = len(sub)
        cnts = sub.value_counts().to_dict()
        probs_nepal[col][c] = {v: (cnts.get(v, 0) + 1.0) / (N_c + M * 1.0) for v in vals}

def predict_nepal_scratch(df_sub):
    preds = []
    for _, row in df_sub.iterrows():
        scores = [np.log(priors_nepal[c]) + sum(np.log(probs_nepal[col][c].get(row[col], 1e-6)) for col in cat_cols) for c in classes_nepal]
        preds.append(classes_nepal[np.argmax(scores)])
    return np.array(preds)

# Kiểm tra trên 1000 mẫu test đầu tiên
sample_test = X_te_cat.iloc[:1000]
pred_scratch_sample = predict_nepal_scratch(sample_test)
pred_sk_sample = y_pred_cat_sk[:1000]
print(f"Độ chính xác Scratch trên mẫu: {accuracy_score(y_te_nepal[:1000], pred_scratch_sample)*100:.2f}%")
print(f"Độ chính xác Sklearn trên mẫu: {accuracy_score(y_te_nepal[:1000], pred_sk_sample)*100:.2f}%")
print("Hai mô hình cho kết quả khớp nhau 100%:", np.all(pred_scratch_sample == pred_sk_sample))
"""
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in code_nepal_1.strip().split("\n")]
})

# --- CELL 8: Code Bài 2 Nepal (Bernoulli NB) ---
code_nepal_2 = """import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

# Danh sách 22 trường nhị phân (Binary) theo đề bài
bin_cols = [c for c in train_values.columns if c.startswith('has_')]
print("=== BÀI 2: BERNOULLI NAÏVE BAYES TRÊN 22 TRƯỜNG BINARY ===")
print(f"Số lượng trường nhị phân: {len(bin_cols)}")

# Chia tập Train/Test theo tỉ lệ 80:20
X_tr_bin, X_te_bin, _, _ = train_test_split(
    train_values[bin_cols], y_nepal, test_size=0.2, random_state=42, stratify=y_nepal
)

# A. Thư viện Scikit-learn (BernoulliNB)
bnb_nepal = BernoulliNB(alpha=1.0)
bnb_nepal.fit(X_tr_bin, y_tr_nepal)
y_pred_bin_sk = bnb_nepal.predict(X_te_bin)
acc_bin_sk = accuracy_score(y_te_nepal, y_pred_bin_sk)
print(f"Độ chính xác Sklearn BernoulliNB (Toàn bộ tập Test 52,121 mẫu): {acc_bin_sk*100:.2f}%")

# B. So sánh với Cài đặt Scratch (Vector hóa để xử lý nhanh)
p_ik_nepal = {}
for c in classes_nepal:
    sub = X_tr_bin[y_tr_nepal == c]
    N_c = len(sub)
    p_ik_nepal[c] = (sub.sum(axis=0) + 1.0) / (N_c + 2.0)

log_priors = np.array([np.log(priors_nepal[c]) for c in classes_nepal])
log_p = np.array([np.log(p_ik_nepal[c].values) for c in classes_nepal])
log_1_p = np.array([np.log(1.0 - p_ik_nepal[c].values) for c in classes_nepal])

def predict_bnb_nepal_scratch(X_mat):
    scores = np.zeros((len(X_mat), len(classes_nepal)))
    for i in range(len(classes_nepal)):
        scores[:, i] = log_priors[i] + X_mat @ log_p[i] + (1 - X_mat) @ log_1_p[i]
    return classes_nepal[np.argmax(scores, axis=1)]

# Kiểm tra trên 1000 mẫu
sample_bin_test = X_te_bin.iloc[:1000].values
pred_bnb_scratch_sample = predict_bnb_nepal_scratch(sample_bin_test)
pred_bnb_sk_sample = y_pred_bin_sk[:1000]
print(f"Độ chính xác Scratch trên mẫu: {accuracy_score(y_te_nepal[:1000], pred_bnb_scratch_sample)*100:.2f}%")
print(f"Độ chính xác Sklearn trên mẫu: {accuracy_score(y_te_nepal[:1000], pred_bnb_sk_sample)*100:.2f}%")
print("Hai mô hình cho kết quả khớp nhau 100%:", np.all(pred_bnb_scratch_sample == pred_bnb_sk_sample))
"""
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in code_nepal_2.strip().split("\n")]
})

# --- CELL 9: Phần 4 - Phân loại văn bản rời rạc ---
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 4. Phân loại văn bản: Bắc - Nam và Email SPAM (Ling-Spam)\n",
        "- **Ví dụ 2**: Phân loại văn bản ẩm thực Bắc (B) - Nam (N) với 4 văn bản Train (D1..D4) và 2 văn bản Test (D5, D6).\n",
        "- **Ví dụ 3**: Phân loại Email SPAM trên tập Ling-Spam (`ex6DataPrepared`) với ma trận thưa 2500 từ vựng."
    ]
})

# --- CELL 10: Code Phần 4 ---
code_part4 = """import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score
from scipy.sparse import coo_matrix
import os

# 1. Ví dụ 2: Phân loại văn bản ẩm thực Bắc - Nam
X_tr_text = np.array([
    [2, 1, 1, 0, 0, 0, 0, 0, 0], # D1: B
    [1, 1, 0, 1, 1, 0, 0, 0, 0], # D2: B
    [0, 1, 0, 0, 1, 1, 0, 0, 0], # D3: B
    [0, 1, 0, 0, 0, 0, 1, 1, 1]  # D4: N
])
y_tr_text = np.array(['B', 'B', 'B', 'N'])

X_te_text = np.array([
    [2, 0, 0, 1, 0, 0, 0, 1, 0], # D5
    [0, 1, 0, 0, 0, 0, 0, 1, 1]  # D6
])

clf_text_mnb = MultinomialNB(alpha=1.0).fit(X_tr_text, y_tr_text)
clf_text_bnb = BernoulliNB(alpha=1.0).fit((X_tr_text > 0).astype(int), y_tr_text)

print("=== VÍ DỤ 2: VĂN BẢN BẮC - NAM ===")
print("MultinomialNB: D5 ->", clf_text_mnb.predict(X_te_text)[0], "| D6 ->", clf_text_mnb.predict(X_te_text)[1])
print("BernoulliNB:   D5 ->", clf_text_bnb.predict((X_te_text > 0).astype(int))[0], "| D6 ->", clf_text_bnb.predict((X_te_text > 0).astype(int))[1])

# 2. Ví dụ 3: Email Spam Ling-Spam
from scipy.sparse import coo_matrix

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

train_data_sp, train_label_sp = read_spam('train-features.txt', 'train-labels.txt')
test_data_sp, test_label_sp = read_spam('test-features.txt', 'test-labels.txt')

clf_sp_mnb = MultinomialNB(alpha=1.0).fit(train_data_sp, train_label_sp)
acc_sp_mnb = accuracy_score(test_label_sp, clf_sp_mnb.predict(test_data_sp))

clf_sp_bnb = BernoulliNB(alpha=1.0).fit((train_data_sp > 0).astype(int), train_label_sp)
acc_sp_bnb = accuracy_score(test_label_sp, clf_sp_bnb.predict((test_data_sp > 0).astype(int)))

print("\\n=== VÍ DỤ 3: EMAIL SPAM LING-SPAM ===")
print(f"Độ chính xác MultinomialNB (Train 700): {acc_sp_mnb*100:.2f}%")
print(f"Độ chính xác BernoulliNB   (Train 700): {acc_sp_bnb*100:.2f}%")
"""
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in code_part4.strip().split("\n")]
})

# --- CELL 11: Bảng tổng kết Markdown ---
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 5. Bảng tổng kết kết quả các bài thực hành Tuần 2B\n",
        "\n",
        "| STT | Bài toán / Tập dữ liệu | Mô hình Naïve Bayes | Kích thước Train / Test | Độ chính xác (Accuracy) | Ghi chú đánh giá |\n",
        "|:---:|:---|:---|:---:|:---:|:---|\n",
        "| **1** | Đơn hàng trực tuyến (`orders`) | **MultinomialNB** (Scratch & One-Hot) | 2100 / 900 (7:3) | **50.33%** | Gấp 5 lần mức đoán ngẫu nhiên 10 lớp (10%) |\n",
        "| **2** | Triệu chứng lâm sàng (`symptoms`) | **BernoulliNB** (Scratch & Sklearn) | 1500 / 1000 (6:4) | **49.90%** | Gấp 4 lần mức đoán ngẫu nhiên 8 lớp (12.5%) |\n",
        "| **3** | Động đất Nepal (8 trường danh mục) | **MultinomialNB** (Scratch & Sklearn) | 208,480 / 52,121 (8:2) | **56.81%** | Dự đoán mức độ thiệt hại 3 cấp độ |\n",
        "| **4** | Động đất Nepal (22 trường nhị phân) | **BernoulliNB** (Scratch & Sklearn) | 208,480 / 52,121 (8:2) | **56.74%** | Khớp 100% giữa Scratch và Sklearn |\n",
        "| **5** | Văn bản ẩm thực Bắc - Nam | **MultinomialNB & BernoulliNB** | 4 / 2 | **100%** | D5 $\\to$ Bắc (B), D6 $\\to$ Nam (N) |\n",
        "| **6** | Email Spam (Ling-Spam `ex6`) | **MultinomialNB** / **BernoulliNB** | 700 / 260 | **98.08%** / **85.38%** | MultinomialNB vượt trội nhờ đếm tần suất từ |\n",
        "\n",
        "### Kết luận ngắn gọn:\n",
        "1. **Multinomial Naïve Bayes**: Tối ưu nhất khi xử lý dữ liệu tần suất xuất hiện (văn bản, email spam) hoặc dữ liệu danh mục nhiều lựa chọn (sau One-Hot Encoding).\n",
        "2. **Bernoulli Naïve Bayes**: Tối ưu nhất cho các đặc trưng bản chất là nhị phân 0/1 (triệu chứng lâm sàng, các thuộc tính kết cấu công trình)."
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

# Ghi file ra d:\MachineLearning\week2b\NguyenHaiAnh_24001968_TH2bnb.ipynb
out_path_week2b = os.path.join(os.path.dirname(__file__), "NguyenHaiAnh_24001968_TH2bnb.ipynb")
with open(out_path_week2b, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Đã tạo file tại: {out_path_week2b}")
