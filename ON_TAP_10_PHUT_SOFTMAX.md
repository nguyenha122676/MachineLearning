# ÔN TẬP CẤP TỐC SOFTMAX TRONG 10 PHÚT (CHUẨN THI MÁY TEST CASE)
> **Mục tiêu:** Đọc hết tờ này trong 10 phút trước khi vào phòng thi để nhớ **1 khung chung**, thuộc **cả 2 chế độ (Không thư viện & Có thư viện)**, và né sạch **bẫy test case**.  
> **File thực hành đi kèm:** [OnTap_Softmax_ThiThuongXuyen.ipynb](file:///D:/MachineLearning/OnTap_Softmax_ThiThuongXuyen.ipynb)

---

## PHÚT 1–2: 1 KHUNG 7 BƯỚC CHO MỌI BÀI SOFTMAX

Mọi đề thi Softmax (2D Gauss, Iris, Glass `[1,2,3,5,6,7]`, hay ảnh MNIST 784 chiều) đều đi đúng **7 bước** này:

```text
[1] ĐỌC & MÃ HÓA NHÃN   : Chuyển y bất kỳ -> 0..C-1 (lưu classes để map ngược)
[2] CHIA TRAIN / TEST   : Chia TRƯỚC khi scale (stratify=y) -> chống Data Leakage
[3] CHUẨN HÓA X         : Fit mean/std CHỈ trên X_train -> transform Train & Test (std==0 -> 1)
[4] CHỌN QUY ƯỚC AXIS   : Mẫu theo hàng (N, d) -> axis=1  |  Mẫu theo cột (d+1, N) -> axis=0
[5] HUẤN LUYỆN          :
    - Không thư viện: one_hot -> Z -> softmax_stable -> loss (clip) -> grad (P-Y) -> cập nhật W, b
    - Có thư viện   : Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(...))])
[6] DỰ ĐOÁN             : predict_proba -> argmax -> classes[idx]
[7] ĐÁNH GIÁ            : accuracy, confusion_matrix, precision/recall/f1 (macro, zero_division=0)
```

---

## PHÚT 3–5: KHÔNG DÙNG THƯ VIỆN ML (ĐIỀN TỪNG HÀM NHỎ)

### A. Quy ước Hàng `(N, d)` — Chuẩn phổ biến khi đề chia nhiều hàm (`axis=1`)
```python
import numpy as np

# 1. Mã hóa nhãn bất kỳ (chữ hoặc nhảy cóc 1,2,3,5,6,7) -> 0..C-1 + One-hot (N, C)
def encode_labels(y):
    classes, y_idx = np.unique(np.asarray(y).ravel(), return_inverse=True)
    return y_idx.astype(np.int64), classes

def one_hot(y_idx, C):
    Y = np.zeros((len(y_idx), C), dtype=np.float64)
    Y[np.arange(len(y_idx)), y_idx] = 1.0
    return Y

# 2. Scale không rò rỉ + chống chia 0
def scale_no_leak(X_tr, X_te):
    mean = X_tr.mean(axis=0)
    std = np.where(X_tr.std(axis=0) == 0, 1.0, X_tr.std(axis=0))
    return (X_tr - mean) / std, (X_te - mean) / std

# 3. Softmax ổn định (hỗ trợ cả 1D (C,) và 2D (N, C))
def softmax(Z):
    Z = np.asarray(Z, dtype=np.float64)
    ax = 0 if Z.ndim == 1 else 1
    e = np.exp(Z - np.max(Z, axis=ax, keepdims=True))
    return e / np.sum(e, axis=ax, keepdims=True)

# 4. Loss Cross-Entropy & Gradient (W shape (d, C), b shape (C,))
def loss_and_grad(X, Y, W, b, l2_reg=0.0):
    N = X.shape[0]
    P = softmax(X @ W + b)
    loss = float(-np.sum(Y * np.log(np.clip(P, 1e-15, 1.0))) / N) + 0.5 * l2_reg * float(np.sum(W * W))
    diff = (P - Y) / N
    return loss, X.T @ diff + l2_reg * W, np.sum(diff, axis=0)

# 5. Mini-batch GD & Dự đoán
def fit_softmax(X, y_idx, C, lr=0.1, epochs=400, batch_size=32, seed=42):
    N, d = X.shape
    Y = one_hot(y_idx, C)
    W, b = np.zeros((d, C)), np.zeros(C)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        perm = rng.permutation(N)
        for s in range(0, N, batch_size):
            idx = perm[s:s + batch_size]
            _, dW, db = loss_and_grad(X[idx], Y[idx], W, b)
            W -= lr * dW
            b -= lr * db
    return W, b

def predict(X, W, b, classes=None):
    idx = np.argmax(softmax(X @ W + b), axis=1)
    return classes[idx] if classes is not None else idx
```

### B. Nếu đề dùng Quy ước Cột `(d+1, N)` theo code mẫu Tuần 4 (`axis=0`)
Chỉ cần đổi **3 chỗ**:
1. Ghép hàng 1 lên đầu: `X_bar = np.vstack([np.ones((1, X.shape[0])), X.T])` → `(d+1, N)`.
2. Đổi `axis=1` thành **`axis=0`**: `softmax_col` trừ `np.max(Z, axis=0, keepdims=True)`, `pred_col` dùng `np.argmax(A, axis=0)`.
3. Cập nhật SGD 1 mẫu `xi = X_bar[:, [i]]`, `yi = Y[:, [i]]`:
   `ai = softmax_col(W.T @ xi)` → `W = W + eta * (xi @ (yi - ai).T)`.

### C. Nếu đề cấm cả NumPy (`math` only)
```python
import math
def py_softmax(logits):
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    s = sum(exps)
    return [e / s for e in exps]
```

---

## PHÚT 6–7: TỰ VIẾT ĐỘ ĐO KHÔNG DÙNG `sklearn.metrics`

```python
def my_confusion_matrix(y_true, y_pred, labels=None):
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([yt, yp]))
    m = {lbl: i for i, lbl in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for a, b in zip(yt, yp):
        if a in m and b in m:
            cm[m[a], m[b]] += 1
    return cm

def my_acc_and_macro_f1(y_true, y_pred, labels=None):
    cm = my_confusion_matrix(y_true, y_pred, labels)
    acc = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
    tp = np.diag(cm).astype(np.float64)
    fp, fn = cm.sum(axis=0) - tp, cm.sum(axis=1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec  = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1   = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return acc, float(np.mean(f1))
```

---

## PHÚT 8–9: ĐƯỢC DÙNG THƯ VIỆN (`sklearn` + MNIST `.gz`)

### 1. Pipeline chuẩn `scikit-learn` (chạy trên mọi phiên bản `sklearn`)
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000, random_state=42)),
])
pipe.fit(X_tr, y_tr)
y_pred, y_proba = pipe.predict(X_te), pipe.predict_proba(X_te)
classes = pipe.named_steps["clf"].classes_

print("Acc:", accuracy_score(y_te, y_pred))
print("Macro-F1:", f1_score(y_te, y_pred, labels=classes, average="macro", zero_division=0))
print("LogLoss:", log_loss(y_te, y_proba, labels=classes))
```
> **Nhớ nhanh về `coef_` của `sklearn`:** khi $C \ge 3$, `clf.coef_` có shape `(C, d)` và `clf.intercept_` có shape `(C,)`.  
> Do đó logit là **`X_scaled @ clf.coef_.T + clf.intercept_`**.

### 2. Đọc file nhị phân MNIST `.gz` chuẩn IDX
```python
import gzip, struct

def read_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, r, c = struct.unpack(">IIII", f.read(16))  # magic ảnh = 2051
        if magic != 2051: raise ValueError("Sai file ảnh IDX")
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r * c).astype(np.float32) / 255.0

def read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic, _ = struct.unpack(">II", f.read(8))           # magic nhãn = 2049
        if magic != 2049: raise ValueError("Sai file nhãn IDX")
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)
```

---

## PHÚT 10: CHECKLIST 8 BẪY TEST CASE HAY MẤT ĐIỂM OAN

- [ ] **Quên trừ `max` trong `exp`** → gặp logit `1000.0` ra `NaN`.
- [ ] **Quên `keepdims=True`** → lỗi shape `(N, C) - (N,)`.
- [ ] **Quên `np.clip(P, 1e-15, 1.0)` trước `np.log`** → `0 * (-inf) = NaN`.
- [ ] **Dùng thẳng `y` làm chỉ số One-hot** → lỗi khi `y` là chữ hoặc nhảy cóc (`1, 2, 3, 5, 6, 7`).
- [ ] **Scale trước khi `train_test_split` (hoặc `fit_transform` trên `X_test`)** → Data Leakage!
- [ ] **Quên chặn `std == 0` khi chuẩn hóa** → cột hằng số chia 0 ra `NaN`.
- [ ] **Truyền `multi_class='multinomial'` trên `sklearn >= 1.8`** → `TypeError` (hãy bỏ tham số `multi_class`, dùng `solver='lbfgs'`).
- [ ] **Nhầm quy ước `(N, d)` vs `(d+1, N)`** → luôn kiểm tra `shape` ở dòng đầu hàm: `N` theo hàng dùng `axis=1`, `N` theo cột dùng `axis=0`.
