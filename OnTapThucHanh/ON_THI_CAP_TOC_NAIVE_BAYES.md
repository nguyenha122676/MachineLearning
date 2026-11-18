# CẨM NANG ÔN THI NAIVE BAYES: TỰ CÀI TỪ ĐẦU (FROM SCRATCH) BẰNG NUMPY & PANDAS
*(Dành cho người học lại từ số 0 — Không phụ thuộc `sklearn`, bám sát 100% bài thực hành & khung code của thầy)*

---

## PHẦN 1: HIỂU 1 CÔNG THỨC GỐC ĐỂ TỰ SUY RA MỌI DÒNG CODE

Đừng cố học thuộc code như vẹt! Chỉ cần hiểu câu chuyện bên dưới, bạn sẽ **tự nhớ ra phải gõ gì**.

### 1.1. Công thức gốc duy nhất của cả chương Naive Bayes

Bài toán phân loại hỏi: *"Cho một mẫu mới có các đặc trưng $\mathbf{x} = (x_1, x_2, \dots, x_d)$, nó thuộc lớp $c$ nào?"*

Theo Định lý Bayes + giả thiết **"Ngây thơ" (Naive)** (coi các cột $x_1, x_2, \dots, x_d$ **độc lập nhau** khi biết lớp $c$), điểm số để chọn lớp $c$ là:

$$\text{Điểm}(c) = P(y = c) \times \prod_{j=1}^{d} P(x_j \mid y = c)$$

- **$P(y = c)$ (*Prior / Tiên nghiệm*)**: Tỷ lệ mẫu thuộc lớp $c$ trong tập Train (`số mẫu lớp c / tổng số mẫu`).
- **$P(x_j \mid y = c)$ (*Likelihood*)**: Khả năng cột thứ $j$ nhận giá trị $x_j$ trong riêng các mẫu thuộc lớp $c$.
- **$\arg\max_c$**: Tính $\text{Điểm}(c)$ cho **từng lớp**, lớp nào điểm cao nhất thì chọn lớp đó!

### 1.2. Tại sao trong code luôn lấy `np.log(...)` rồi cộng (`+`)?

1. **Chống tràn số về `0.0` (*Underflow*)**: Mỗi xác suất đều nhỏ hơn `1` (vd `0.05`). Nhân 20–30 số nhỏ hơn `1` lại sẽ ra số bé tí xíu, máy tính làm tròn thành `0.0` $\rightarrow$ không so sánh được lớp nào lớn hơn!
2. **Giữ nguyên lớp chiến thắng**: Vì $\ln(A \times B) = \ln(A) + \ln(B)$ và hàm $\ln$ tăng đồng biến, nên đổi phép **NHÂN** thành phép **CỘNG** các `log`:
   $$\text{Log-Điểm}(c) = \ln P(y = c) + \sum_{j=1}^{d} \ln P(x_j \mid y = c)$$
   Lớp nào có `Log-Điểm(c)` lớn nhất vẫn là **đúng lớp đó**!

---

### 1.3. Nhìn dữ liệu 10 giây — Phân biệt 4 dạng bài tự cài From Scratch

Cả 4 dạng bài chỉ khác nhau ở **duy nhất 1 chỗ: Cách tính $P(x_j \mid y = c)$**:

| Dạng bài | Nhìn cột của `X` thấy gì? | Bài mẫu trong Workspace | Công thức tính $P(x_j \mid y=c)$ |
|---|---|---|---|
| **1. Gaussian NB** | **Số thực đo đạc** (`5.1`, `3.5`, `1.4`...) | [iris.csv](file:///d:/MachineLearning/Week2/iris.csv), [breast-cancer-wisconsin.data](file:///d:/MachineLearning/Week2/breast-cancer-wisconsin.data) | Hình chuông Gauss với trung bình $\mu_{cj}$ và phương sai $\sigma_{cj}^2$: `norm.logpdf(...)` hoặc tự viết hàm $\ln$ Gauss |
| **2. Bernoulli NB** | **Chỉ có `0` và `1`** (Có / Không triệu chứng, Có / Không từ) | [bernoulli_nb_symptoms.csv](file:///d:/MachineLearning/Week2/bernoulli_nb_symptoms.csv), cột `has_...` Nepal | Gọi $p_{cj} = \frac{\text{số ca lớp } c \text{ có } x_j=1 + \alpha}{N_c + 2\alpha}$.<br>Đoán: $x_j \cdot p_{cj} + (1 - x_j)(1 - p_{cj})$ |
| **3. Multinomial NB (Văn bản)** | **Số đếm tần suất từ** (`0, 1, 2, 3...`) | Văn bản Bắc–Nam (`D1..D6`), Email Spam `ex6DataPrepared` | $p_{cj} = \frac{\text{tổng số lần từ } j \text{ ở lớp } c + \alpha}{\text{tổng lượt từ ở lớp } c + \alpha \cdot d}$.<br>Đoán: $x_j \cdot \ln(p_{cj})$ |
| **4. Categorical / Multinomial NB (Bảng chữ)** | **Chữ nhiều lựa chọn** (`ios`, `web`, `city_1`...) | [multinomial_nb_orders.csv](file:///d:/MachineLearning/Week2/multinomial_nb_orders.csv), 8 cột chữ Nepal | Cột $i$ có $M_i$ lựa chọn:<br>$P(x_i = v \mid c) = \frac{\text{số mẫu lớp } c \text{ có cột } i = v + \alpha}{N_c + M_i \cdot \alpha}$ |

> **Tại sao luôn cộng thêm $\alpha = 1$ (*Laplace Smoothing*)?**  
> Để tránh trường hợp một từ/giá trị ở tập Test chưa bao giờ gặp trong lớp `c` ở tập Train $\rightarrow$ số lần đếm `= 0` $\rightarrow$ xác suất `= 0` $\rightarrow$ $\ln(0) = -\infty$ làm hỏng cả phép tính! Cộng $\alpha = 1$ vào tử số và $M \cdot \alpha$ vào mẫu số giúp xác suất luôn dương.

---

## PHẦN 2: CHUẨN BỊ DỮ LIỆU KHÔNG CẦN `sklearn` (CHỈ DÙNG `pandas` & `numpy`)

### 2.1. Đọc CSV, dọn ô lỗi `?` và tách `X, y`
```python
import numpy as np
import pandas as pd

# 1. Đọc file có tiêu đề cột
df = pd.read_csv('iris.csv')

# Hoặc đọc file KHÔNG có tiêu đề (như breast-cancer-wisconsin.data)
# df = pd.read_csv('breast-cancer-wisconsin.data', header=None, names=col_names)

# 2. Nếu có cột chứa dấu '?' (thiếu dữ liệu) -> đổi thành NaN, ép sang số, điền trung vị
df['bare_nuclei'] = pd.to_numeric(df['bare_nuclei'].replace('?', np.nan))
df['bare_nuclei'] = df['bare_nuclei'].fillna(df['bare_nuclei'].median())

# 3. Tách đề bài X (bỏ cột nhãn và cột id nếu có) và đáp án y
X = df.drop(columns=['variety']).to_numpy(dtype=float)   # chuyển sang mảng numpy 2D
y = df['variety'].to_numpy()                             # chuyển sang mảng numpy 1D
```

### 2.2. Tự chia Train / Test bằng 2 cách (không cần `train_test_split`)

**Cách 1: Cắt thẳng phần đầu làm Train, phần cuối làm Test (hay gặp trong notebook thầy)**
```python
train_size = round(len(df) * 0.7)    # ví dụ 70% Train, 30% Test
X_train = df.iloc[:train_size].drop(columns=['target']).reset_index(drop=True)
y_train = df['target'].iloc[:train_size].reset_index(drop=True)
X_test  = df.iloc[train_size:].drop(columns=['target']).reset_index(drop=True)
y_test  = df['target'].iloc[train_size:].reset_index(drop=True)
```

**Cách 2: Xáo trộn ngẫu nhiên bằng NumPy rồi mới cắt (công bằng hơn khi dữ liệu bị xếp theo nhóm như Iris)**
```python
np.random.seed(42)                       # cố định hạt giống ngẫu nhiên
idx = np.random.permutation(len(X))      # xáo trộn thứ tự chỉ số 0..N-1
split = int(0.8 * len(X))                # 80% Train, 20% Test

X_train, X_test = X[idx[:split]], X[idx[split:]]
y_train, y_test = y[idx[:split]], y[idx[split:]]
```

---

## PHẦN 3: 4 BỘ CODE TỰ CÀI ĐẦY ĐỦ TỪ A ĐẾN Z (ĐÃ KIỂM CHỨNG KHỚP `sklearn`)

Tất cả 4 bộ code dưới đây **đều viết theo đúng 1 khuôn gồm 2 phần**:
1. **Học từ tập Train** (lưu `classes`, `priors`, và tham số từng lớp).
2. **Dự đoán trên tập Test** (với mỗi mẫu `x`: lặp từng lớp $\rightarrow$ tính `log(prior) + sum(log(likelihood))` $\rightarrow$ chọn `argmax`).

---

### MẪU 1: Gaussian Naive Bayes (Dữ liệu số thực liên tục)

#### Bản 1A — Thuần NumPy 100% (Chạy được ngay cả khi máy chấm CẤM cả `scipy`!)
```python
import numpy as np

# === BƯỚC 1: HỌC THAM SỐ TỪ TẬP TRAIN ===
classes = np.unique(y_train)                               # danh sách các lớp, vd ['Setosa', 'Versicolor', 'Virginica']
n_classes = len(classes)
n_features = X_train.shape[1]

means  = np.zeros((n_classes, n_features), dtype=float)    # trung bình từng cột theo từng lớp
vars_  = np.zeros((n_classes, n_features), dtype=float)    # phương sai từng cột theo từng lớp
priors = np.zeros(n_classes, dtype=float)                  # tỷ lệ ban đầu của từng lớp

for idx, c in enumerate(classes):
    X_c = X_train[y_train == c]                            # lọc lấy các dòng thuộc lớp c
    means[idx]  = np.mean(X_c, axis=0)
    vars_[idx]  = np.var(X_c, axis=0) + 1e-9               # + 1e-9 để chống chia cho 0!
    priors[idx] = len(X_c) / len(X_train)

# === BƯỚC 2: DỰ ĐOÁN CHO TẬP TEST ===
def predict_gaussian_nb(X_data):
    y_pred = []
    for x in X_data:                                       # xét từng mẫu x (1 hàng)
        scores = []
        for idx in range(n_classes):
            # Công thức log mật độ Gauss: -0.5 * ln(2*pi*var) - (x - mean)^2 / (2*var)
            log_pdf = -0.5 * np.log(2 * np.pi * vars_[idx]) - ((x - means[idx]) ** 2) / (2 * vars_[idx])
            score = np.log(priors[idx]) + np.sum(log_pdf)
            scores.append(score)
        y_pred.append(classes[np.argmax(scores)])
    return np.array(y_pred)

y_pred = predict_gaussian_nb(X_test)
```
> **Ghi chú**: Bản thuần NumPy này đã chạy kiểm chứng trên `iris.csv`, cho kết quả **khớp 100%** với `GaussianNB()` của `sklearn`!

#### Bản 1B — Theo khung `Gaussian_NB.ipynb` của thầy (dùng `scipy.stats.norm`)
```python
from scipy.stats import norm
import numpy as np

classes  = np.unique(y_train)
mu_list  = [np.mean(X_train[y_train == c], axis=0) for c in classes]
std_list = [np.std(X_train[y_train == c], axis=0, ddof=1) for c in classes]
pi_list  = [np.mean(y_train == c) for c in classes]

def predict_NB_gaussian_class(x, mu_list, std_list, pi_list):
    scores_list = []
    for c in range(len(mu_list)):
        mu_c  = np.asarray(mu_list[c]).ravel()
        std_c = np.asarray(std_list[c]).ravel()
        x_vec = np.asarray(x).ravel()
        # Dùng norm.logpdf cộng với np.log(pi) nếu nhiều cột; hoặc nhân trực tiếp nếu ít cột:
        score = np.log(pi_list[c]) + np.sum(norm.logpdf(x_vec, loc=mu_c, scale=std_c))
        scores_list.append(score)
    return classes[np.argmax(scores_list)]   # trả về tên lớp (nếu nhãn là 0,1,2 thì chính là np.argmax)
```

---

### MẪU 2: Bernoulli Naive Bayes (Dữ liệu nhị phân `0` / `1`)

Dùng cho bài **Triệu chứng bệnh** (`bernoulli_nb_symptoms.csv`), **Văn bản 0/1**, **Động đất Nepal (22 cột `has_...`)**.

```python
import numpy as np

# === BƯỚC 1: HỌC THAM SỐ TỪ TẬP TRAIN ===
# X_train, X_test chứa toàn số 0 và 1
classes = np.unique(y_train)
n_classes = len(classes)
n_features = X_train.shape[1]
alpha = 1.0                                                # làm trơn Laplace (nếu đề không dùng Laplace thì đặt 0)

priors = np.zeros(n_classes, dtype=float)
p_ik   = np.zeros((n_classes, n_features), dtype=float)    # P(cột j = 1 | lớp c)

for idx, c in enumerate(classes):
    X_c = X_train[y_train == c]
    N_c = len(X_c)
    priors[idx] = N_c / len(X_train)
    count_1 = np.sum(X_c == 1, axis=0)                     # đếm số ca bằng 1 ở từng cột
    p_ik[idx] = (count_1 + alpha) / (N_c + 2.0 * alpha)    # mẫu + 2*alpha vì chỉ có 2 giá trị 0 và 1

# === BƯỚC 2: DỰ ĐOÁN CHO TẬP TEST ===
def predict_bernoulli_nb(X_data):
    y_pred = []
    for x in X_data:
        scores = []
        for idx in range(n_classes):
            # Công thức Bernoulli: nếu x_j = 1 lấy p_ik; nếu x_j = 0 lấy (1 - p_ik)
            prob = x * p_ik[idx] + (1 - x) * (1 - p_ik[idx])
            score = np.log(priors[idx]) + np.sum(np.log(np.clip(prob, 1e-12, 1.0)))
            scores.append(score)
        y_pred.append(classes[np.argmax(scores)])
    return np.array(y_pred)

y_pred = predict_bernoulli_nb(X_test)
```
> **Khung hàm lẻ trong [NaiveBayes_Bernoulli.ipynb](file:///d:/MachineLearning/week2b/NaiveBayes_Bernoulli.ipynb) của thầy**: Nếu đề cho sẵn khung `def Pik_feature_per_class(X):`, bạn chỉ cần viết:
> ```python
> def Pik_feature_per_class(X_col_k):
>     X_arr = np.asarray(X_col_k)
>     return (X_arr == 1).sum() / len(X_arr)
> ```

---

### MẪU 3: Multinomial Naive Bayes — Văn bản đếm tần suất từ (Bag-of-Words)

Dùng cho bài **Văn bản Bắc–Nam (`D1..D6`)** hoặc **Email Spam (`ex6DataPrepared`)**, nơi `X_train` là ma trận đếm số lần từ xuất hiện (vd `[2, 1, 1, 0, ...]`).

```python
import numpy as np

# === BƯỚC 1: HỌC THAM SỐ TỪ TẬP TRAIN ===
classes = np.unique(y_train)
n_classes = len(classes)
d = X_train.shape[1]                                       # kích thước từ điển |V| (tổng số cột từ)
alpha = 1.0                                                # làm trơn Laplace

priors = np.zeros(n_classes, dtype=float)
p_word = np.zeros((n_classes, d), dtype=float)             # xác suất xuất hiện của từng từ trong lớp c

for idx, c in enumerate(classes):
    X_c = X_train[y_train == c]
    priors[idx] = len(X_c) / len(X_train)

    word_counts_c = np.sum(X_c, axis=0)                    # tổng số lần từng từ xuất hiện ở lớp c -> mảng (d,)
    total_words_c = np.sum(X_c)                            # tổng số lượt từ trong toàn bộ lớp c -> 1 con số!
    p_word[idx] = (word_counts_c + alpha) / (total_words_c + alpha * d)

# === BƯỚC 2: DỰ ĐOÁN CHO TẬP TEST ===
def predict_multinomial_text_nb(X_data):
    y_pred = []
    for x in X_data:
        scores = []
        for idx in range(n_classes):
            # Từ j xuất hiện x[j] lần -> cộng x[j] * ln(p_word)
            # (Từ nào không xuất hiện trong văn bản test thì x[j] = 0 -> tự động bỏ qua!)
            score = np.log(priors[idx]) + np.sum(x * np.log(p_word[idx]))
            scores.append(score)
        y_pred.append(classes[np.argmax(scores)])
    return np.array(y_pred)

y_pred = predict_multinomial_text_nb(X_test)
```
> **Tại sao có `x * np.log(p_word[idx])`?**  
> Vì trong văn bản `"Hà-Nội Hà-Nội bún-chả"`, từ `"Hà-Nội"` xuất hiện `2` lần (`x[0] = 2`), theo công thức xác suất phải nhân $p_{\text{Hà-Nội}}^2 \xrightarrow{\text{lấy log}} 2 \cdot \ln(p_{\text{Hà-Nội}})$. Còn từ không xuất hiện có `x[j] = 0`, nhân `0` sẽ tự mất đi!

---

### MẪU 4: Categorical / Multinomial Naive Bayes — Bảng chữ nhiều lựa chọn

Dùng cho bài **Đơn hàng ẩm thực (`multinomial_nb_orders.csv`)** hoặc **Động đất Nepal (8 cột chữ)**, nơi `X_train` là DataFrame chứa chữ (`'city_1'`, `'ios'`, `'Tue'`...).

```python
import numpy as np
import pandas as pd

# X_train, X_test là DataFrame (đã bỏ cột nhãn); y_train là mảng 1D
classes = np.unique(y_train)
feature_cols = X_train.columns
alpha = 1.0

# === BƯỚC 1: HỌC XÁC SUẤT CHO TỪNG CỘT, TỪNG LỚP ===
priors = {c: np.mean(y_train == c) for c in classes}
probs = {}                                                 # bảng tra: probs[tên_cột][lớp_c][giá_trị]

for col in feature_cols:
    probs[col] = {}
    unique_vals = X_train[col].unique()                    # các lựa chọn có thể có trong cột
    M_col = len(unique_vals)                               # số lựa chọn M của cột này
    for c in classes:
        sub_col = X_train[col][y_train == c]
        N_c = len(sub_col)
        counts = sub_col.value_counts().to_dict()          # đếm nhanh tần suất từng giá trị
        probs[col][c] = {v: (counts.get(v, 0) + alpha) / (N_c + M_col * alpha) for v in unique_vals}
        probs[col][c]["__UNSEEN__"] = alpha / (N_c + M_col * alpha)  # phòng giá trị lạ ở Test!

# === BƯỚC 2: DỰ ĐOÁN CHO TẬP TEST ===
def predict_categorical_nb(X_df):
    y_pred = []
    for _, row in X_df.iterrows():
        scores = []
        for c in classes:
            score = np.log(priors[c])
            for col in feature_cols:
                p_val = probs[col][c].get(row[col], probs[col][c]["__UNSEEN__"])
                score += np.log(p_val)
            scores.append(score)
        y_pred.append(classes[np.argmax(scores)])
    return np.array(y_pred)

y_pred = predict_categorical_nb(X_test)
```
> **Khung hàm lẻ trong [NaiveBayes_Multinomial.ipynb](file:///d:/MachineLearning/week2b/NaiveBayes_Multinomial.ipynb) của thầy**: Nếu đề cho sẵn khung `def Pxik_feature_per_class(X_col_k, xi_val, num_nomials_M, alpha=1.0):`:
> ```python
> def Pxik_feature_per_class(X_col_k, xi_val, num_nomials_M, alpha=1.0):
>     X_arr = np.asarray(X_col_k)
>     count = (X_arr == xi_val).sum()
>     return (count + alpha) / (len(X_arr) + num_nomials_M * alpha)
> ```

---

## PHẦN 4: TỰ CÀI CÁC ĐỘ ĐO ĐÁNH GIÁ KHÔNG CẦN `sklearn.metrics`

Nếu máy chấm cấm luôn `sklearn.metrics`, bạn chỉ cần nhớ 5 công thức NumPy siêu ngắn này:

```python
import numpy as np

# 1. Accuracy (cho cả 2 lớp và nhiều lớp)
acc = np.mean(y_pred == y_true)

# 2. Precision, Recall, F1 cho bài toán 2 lớp (gọi pos là nhãn lớp dương, vd 1 hoặc 4)
def calc_binary_metrics(y_true, y_pred, pos=1):
    TP = np.sum((y_true == pos) & (y_pred == pos))   # Thật = 1, Đoán = 1
    FP = np.sum((y_true != pos) & (y_pred == pos))   # Thật = 0, Đoán = 1 (báo nhầm)
    FN = np.sum((y_true == pos) & (y_pred != pos))   # Thật = 1, Đoán = 0 (bỏ sót)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1
```

---

## PHẦN 5: PHỤ LỤC NHANH — NẾU ĐỀ CHO PHÉP DÙNG `sklearn`

Chỉ cần 4 bước cố định (đưa `X_train, y_train` vào `.fit()`, đưa `X_test` vào `.predict()`):
- Số thực liên tục: `from sklearn.naive_bayes import GaussianNB` $\rightarrow$ `clf = GaussianNB()`
- Nhị phân `0/1`: `from sklearn.naive_bayes import BernoulliNB` $\rightarrow$ `clf = BernoulliNB(alpha=1.0)`
- Văn bản số đếm: `from sklearn.naive_bayes import MultinomialNB` $\rightarrow$ `clf = MultinomialNB(alpha=1.0)`
- Bảng chữ nhiều lựa chọn: Qua `OneHotEncoder(sparse_output=False, handle_unknown='ignore')` (`fit_transform` trên Train, `transform` trên Test) rồi đưa vào `MultinomialNB(alpha=1.0)`! *(Không dùng `LabelEncoder` cho `X` khi chạy `MultinomialNB`!)*

---

## PHẦN 6: 5 BẪY HAY MẤT ĐIỂM KHI MÁY CHẤM TỰ ĐỘNG

1. **Quên `+ 1e-9` vào phương sai `vars_` ở Gaussian NB**: Nếu trong 1 lớp nào đó có 1 cột toàn số giống hệt nhau (phương sai `= 0`), công thức sẽ bị chia cho `0` và ra `NaN`! Luôn cộng `+ 1e-9`.
2. **Quên `np.clip(prob, 1e-12, 1.0)` khi lấy `np.log` ở Bernoulli NB**: Nếu không dùng Laplace (`alpha=0`) mà `prob` rơi vào `0.0`, `np.log(0)` ra `-inf` làm chết cả bài.
3. **Truyền sai thứ tự `(y_true, y_pred)` hoặc sai kiểu dữ liệu**:
   - Đảm bảo `y_true` và `y_pred` cùng kiểu (đừng để `y_true` là chữ `'Setosa'` còn `y_pred` là số `0` $\rightarrow$ `y_pred == y_true` trả về `False` hết!).
   - Luôn gọi `classes[np.argmax(scores)]` thay vì trả về thẳng chỉ số `np.argmax(scores)` nếu nhãn của đề là chữ (`'B'`, `'N'`) hoặc số không bắt đầu từ `0` (như nhãn `2`, `4` trong bài Ung thư vú!).
4. **Bẫy chỉ số bắt đầu từ 1 trong file Spam `ex6DataPrepared` (`coo_matrix`)**: File `.txt` đánh số từ `1`, Python đếm từ `0` $\rightarrow$ nhớ `- 1`: `(dat[:, 0] - 1, dat[:, 1] - 1)`.
5. **Quên `.reset_index(drop=True)` sau khi cắt `.iloc[train_size:]` bằng Pandas**: Làm lệch chỉ số dòng `2100..2999`, gây `KeyError` khi vòng lặp lấy phần tử đầu tiên.
