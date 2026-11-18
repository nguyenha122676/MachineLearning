# Week 4 — K-NN nhiều lớp, Softmax và MNIST cho người bắt đầu từ số 0

> Week 3 hỏi “0 hay 1?”. Week 4 hỏi “trong nhiều lớp, lớp nào phù hợp nhất?”. Mục tiêu là hiểu cách mô hình tạo điểm cho từng lớp, đổi thành xác suất và chọn đáp án.

---

## 0. Cách học tài liệu này

1. Đọc mục 1–8 để hiểu multiclass và softmax bằng lời.
2. Tính ví dụ ba lớp ở mục 4 và 8 bằng máy tính.
3. Chạy Iris ở mục 14.
4. Chỉ sang MNIST sau khi hiểu shape `(n, d)` và `(n, C)`.

Sau Week 4, bạn cần:

- Phân biệt binary, multiclass và multilabel.
- Tự làm K-NN nhiều lớp.
- Hiểu logit/score và softmax.
- Hiểu one-hot và categorical cross-entropy.
- Theo dõi đúng shape khi tự code.
- Đọc dữ liệu MNIST và phân tích lỗi theo từng chữ số.

---

## 1. Từ hai lớp đến nhiều lớp

Week 3 có hai lựa chọn:

```text
0 = trượt
1 = đỗ
```

Week 4 có từ ba lớp trở lên:

```text
Iris:  Setosa, Versicolor, Virginica
MNIST: 0, 1, 2, ..., 9
Glass: nhiều loại kính
```

Đây là **multiclass classification**: mỗi mẫu thuộc đúng một lớp.

### Multiclass khác multilabel

- Multiclass: ảnh chữ số chỉ là một số, ví dụ `7`.
- Multilabel: một ảnh có thể đồng thời mang nhãn `người`, `xe`, `ngoài trời`.

Softmax phù hợp khi các lớp cạnh tranh và chỉ một lớp đúng. Multilabel thường dùng sigmoid riêng cho từng nhãn.

---

## 2. Dữ liệu nhìn như thế nào?

Với Iris:

- Một hàng: một bông hoa.
- Bốn feature: chiều dài/rộng đài hoa và cánh hoa.
- Một label: tên một trong ba loài.

```python
import pandas as pd

df = pd.read_csv("week4/iris.csv")
df.columns = [name.replace(".", "_") for name in df.columns]

X = df[[
    "sepal_length", "sepal_width",
    "petal_length", "petal_width",
]]
y = df["variety"]

print(X.shape)  # (150, 4)
print(y.shape)  # (150,)
```

Trong ký hiệu chung:

- `n`: số mẫu.
- `d`: số feature.
- `C`: số lớp.
- `X` có shape `(n, d)`.

---

## 3. K-NN nhiều lớp

Quy trình không khác K-NN nhị phân:

1. Tính khoảng cách tới mọi mẫu train.
2. Lấy `k` láng giềng gần nhất.
3. Đếm số phiếu của từng lớp.
4. Chọn lớp nhiều phiếu nhất.

Không còn chỉ đếm 0 và 1; có thể có lớp 0, 1, 2, ..., `C-1`.

---

## 4. Ví dụ K-NN ba lớp bằng tay

Ta có các điểm 1D:

| `x` | Lớp |
|---:|---:|
| 1.0 | 0 |
| 1.5 | 0 |
| 3.0 | 1 |
| 3.5 | 1 |
| 5.0 | 2 |
| 5.5 | 2 |

Dự đoán `x_new=4.2`, `k=3`.

Khoảng cách gần nhất:

- Tới `3.5`, lớp 1: `0.7`.
- Tới `5.0`, lớp 2: `0.8`.
- Tới `3.0`, lớp 1: `1.2`.

Phiếu:

```text
lớp 0: 0 phiếu
lớp 1: 2 phiếu
lớp 2: 1 phiếu
```

Dự đoán lớp 1.

```python
import numpy as np

def knn_multiclass_predict(X_train, y_train, X_test, k=5):
    predictions = []

    for x in X_test:
        distances = np.linalg.norm(X_train - x, axis=1)
        nearest = np.argsort(distances)[:k]
        votes = np.bincount(y_train[nearest])
        predictions.append(np.argmax(votes))

    return np.asarray(predictions)
```

`np.bincount` phù hợp khi nhãn là số nguyên không âm. `np.argmax(votes)` trả chỉ số lớp có nhiều phiếu nhất.

---

## 5. Hòa phiếu, scaling và nhiều chiều

### Hòa phiếu

Nếu các lớp hòa, `np.argmax` chọn vị trí xuất hiện đầu tiên — thường là lớp có số nhỏ nhất. Đây là một quy tắc kỹ thuật, không có nghĩa lớp nhỏ hơn hợp lý hơn.

Có thể giảm hòa bằng `k` lẻ, nhưng với ba lớp vẫn có thể hòa kiểu `1–1–1`.

### Scaling

K-NN dựa vào khoảng cách nên feature có đơn vị lớn sẽ lấn át. Luôn cân nhắc `StandardScaler`.

### Curse of dimensionality

Khi số chiều rất cao, hầu hết các điểm đều trở nên “xa”, và khoảng cách gần/xa kém khác biệt. MNIST có 784 feature, nên K-NN vẫn dùng làm baseline nhưng dự đoán chậm và có thể khó mở rộng.

---

## 6. Tại sao cần một score cho mỗi lớp?

Với ba lớp, mô hình tạo ba score:

```text
Setosa:     4.2
Versicolor: 1.1
Virginica: -0.7
```

Score lớn nhất nghiêng về Setosa. Nhưng các score chưa phải xác suất:

- Có thể âm.
- Không bị giới hạn ở 1.
- Không có tổng bằng 1.

Ta cần softmax để đổi toàn bộ dãy score thành phân phối xác suất.

Score trước softmax thường gọi là **logit**.

---

## 7. Từ sigmoid đến softmax

Binary Logistic Regression dùng một sigmoid để tạo xác suất lớp 1; lớp 0 có xác suất `1-p`.

Với nhiều lớp, mô hình tính:

$$
Z=XW+b
$$

- `X`: `(n, d)`.
- `W`: `(d, C)` — mỗi cột là trọng số của một lớp.
- `b`: `(C,)`.
- `Z`: `(n, C)` — mỗi hàng chứa `C` score của một mẫu.

Softmax biến mỗi hàng score thành xác suất:

$$
P(y=c\mid x)=\frac{e^{z_c}}{\sum_{k=1}^{C}e^{z_k}}
$$

Mọi xác suất dương và tổng mỗi hàng bằng 1.

---

## 8. Softmax tính bằng tay

Giả sử logits của một mẫu là `[2, 1, 0]`.

### Bước 1 — lấy exponential

```text
exp(2) ≈ 7.389
exp(1) ≈ 2.718
exp(0) = 1
```

### Bước 2 — tính tổng

```text
tổng ≈ 11.107
```

### Bước 3 — chia từng số cho tổng

```text
lớp 0: 7.389 / 11.107 ≈ 0.665
lớp 1: 2.718 / 11.107 ≈ 0.245
lớp 2: 1     / 11.107 ≈ 0.090
```

Tổng bằng 1. Mô hình dự đoán lớp 0 vì xác suất lớn nhất.

Điều quan trọng là chênh lệch logits, không phải giá trị tuyệt đối. `[102,101,100]` cho cùng softmax với `[2,1,0]`.

---

## 9. Stable softmax — tránh máy tính bị tràn số

`exp(1000)` quá lớn và có thể thành vô cực. Ta trừ logit lớn nhất của **từng hàng**:

```python
def softmax(scores):
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)
```

Với `[1002,1001,1000]`, sau khi trừ max trở thành `[0,-1,-2]`, an toàn hơn nhưng xác suất không đổi.

`axis=1` nghĩa là thao tác theo từng hàng/mẫu. `keepdims=True` giữ shape `(n,1)` để broadcasting đúng.

Tự kiểm tra:

```python
probabilities = softmax(np.array([[1000.0, 1001.0, 1002.0]]))
assert np.isfinite(probabilities).all()
assert np.allclose(probabilities.sum(axis=1), 1.0)
```

---

## 10. One-hot encoding là gì?

Nhãn lớp dạng số:

```text
y = [0, 2, 1]
```

Nếu có ba lớp, one-hot là:

```text
lớp 0 → [1, 0, 0]
lớp 2 → [0, 0, 1]
lớp 1 → [0, 1, 0]
```

Mỗi hàng có đúng một số 1 tại lớp đúng.

```python
def one_hot(y, n_classes):
    encoded = np.zeros((len(y), n_classes))
    encoded[np.arange(len(y)), y] = 1
    return encoded
```

Nếu nhãn là chuỗi, cần mã hóa thành `0..C-1` trước khi tự tạo one-hot.

---

## 11. Cross-entropy nhiều lớp

Giả sử lớp đúng là 2, one-hot `[0,0,1]`.

- Mô hình A dự đoán `[0.1,0.2,0.7]` → loss `-log(0.7)≈0.357`.
- Mô hình B dự đoán `[0.8,0.1,0.1]` → loss `-log(0.1)≈2.303`.

Mô hình đặt xác suất thấp cho lớp đúng bị phạt cao.

$$
L=-\frac{1}{n}\sum_i\sum_c y_{ic}\log p_{ic}
$$

Vì one-hot chỉ có một số 1, với mỗi mẫu loss thực chất là âm log xác suất của lớp đúng.

---

## 12. Mô hình học như thế nào?

Sai số theo từng lớp có dạng rất gọn:

$$
P-Y
$$

Trực giác:

- Lớp đúng: `Y=1`, nên `P-Y` âm nếu xác suất chưa đủ cao; cập nhật sẽ tăng logit lớp đúng.
- Lớp sai: `Y=0`, nên `P-Y` dương; cập nhật sẽ giảm logit lớp sai.

Gradient:

$$
\frac{\partial L}{\partial W}=\frac{1}{n}X^T(P-Y)
$$

$$
\frac{\partial L}{\partial b}=mean(P-Y)
$$

Gradient descent trừ các gradient này khỏi `W` và `b`.

---

## 13. Softmax Regression từ đầu

```python
import numpy as np

def softmax(scores):
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)

def one_hot(y, n_classes):
    encoded = np.zeros((len(y), n_classes))
    encoded[np.arange(len(y)), y] = 1
    return encoded

def fit_softmax(X, y, learning_rate=0.1, epochs=2000):
    n, d = X.shape
    classes = np.unique(y)

    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("Nhãn phải liên tiếp từ 0 đến C-1")

    C = len(classes)
    Y = one_hot(y, C)
    W = np.zeros((d, C))
    b = np.zeros(C)

    for _ in range(epochs):
        P = softmax(X @ W + b)
        W -= learning_rate * (X.T @ (P - Y) / n)
        b -= learning_rate * (P - Y).mean(axis=0)

    return W, b

def predict_softmax(X, W, b):
    probabilities = softmax(X @ W + b)
    return np.argmax(probabilities, axis=1)
```

Shape map với `n=150`, `d=4`, `C=3`:

| Biến | Shape |
|---|---|
| `X` | `(150,4)` |
| `W` | `(4,3)` |
| `b` | `(3,)` |
| `Z` | `(150,3)` |
| `P` | `(150,3)` |
| `Y` | `(150,3)` |
| gradient `W` | `(4,3)` |

Nếu shape khác bảng, sửa shape trước khi chỉnh learning rate.

---

## 14. Iris multiclass với sklearn

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("week4/iris.csv")
df.columns = [name.replace(".", "_") for name in df.columns]

features = [
    "sepal_length", "sepal_width",
    "petal_length", "petal_width",
]
X = df[features]
y = df["variety"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000),
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

Sklearn mã hóa lớp chuỗi nội bộ. Xem thứ tự lớp bằng estimator cuối:

```python
print(model[-1].classes_)
print(model.predict_proba(X_test.iloc[:2]))
```

Mỗi hàng probability có ba cột theo thứ tự `classes_`.

---

## 15. Đánh giá nhiều lớp

Confusion matrix có `C` hàng và `C` cột:

- Hàng: lớp thật.
- Cột: lớp dự đoán.
- Đường chéo: đúng.
- Ngoài đường chéo: nhầm từ lớp nào sang lớp nào.

### Macro average

Tính metric cho từng lớp rồi trung bình đều. Lớp ít mẫu quan trọng ngang lớp nhiều mẫu.

### Weighted average

Trung bình theo số mẫu mỗi lớp. Lớp lớn có ảnh hưởng nhiều hơn.

Nếu dữ liệu mất cân bằng, báo cáo Accuracy cùng macro F1 và metric từng lớp.

---

## 16. MNIST là gì?

MNIST là bộ ảnh chữ số viết tay từ 0 đến 9.

- Mỗi ảnh rộng 28 pixel, cao 28 pixel.
- Mỗi pixel có độ sáng từ 0 đến 255.
- Mỗi ảnh có một nhãn `0..9`.

Để đưa vào mô hình tuyến tính, ta trải phẳng ảnh:

```text
ảnh 28×28 → vector 784 số
```

Nếu có `n` ảnh:

- `X` có shape `(n,784)`.
- `y` có shape `(n,)`.
- `C=10`.
- `W` có shape `(784,10)`.

Chia pixel cho `255.0` để đưa về `0..1`.

---

## 17. Đọc file MNIST IDX nén

File MNIST trong repo là nhị phân `.gz`, không đọc bằng `pd.read_csv`.

```python
import gzip
import struct
import numpy as np

def load_idx_images(path):
    with gzip.open(path, "rb") as file:
        magic, n, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError("File ảnh IDX không hợp lệ")
        data = np.frombuffer(file.read(), dtype=np.uint8)

    if len(data) != n * rows * cols:
        raise ValueError("Số pixel không khớp header")

    return data.reshape(n, rows * cols).astype(np.float32) / 255.0

def load_idx_labels(path):
    with gzip.open(path, "rb") as file:
        magic, n = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError("File nhãn IDX không hợp lệ")
        labels = np.frombuffer(file.read(), dtype=np.uint8)

    if len(labels) != n:
        raise ValueError("Số nhãn không khớp header")

    return labels
```

`magic` xác nhận đúng loại file. Ký hiệu `>` yêu cầu đọc số nguyên big-endian theo định dạng IDX.

---

## 18. MNIST baseline bằng Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

X_train = load_idx_images("week4/train-images-idx3-ubyte.gz")
y_train = load_idx_labels("week4/train-labels-idx1-ubyte.gz")
X_test = load_idx_images("week4/t10k-images-idx3-ubyte.gz")
y_test = load_idx_labels("week4/t10k-labels-idx1-ubyte.gz")

model = LogisticRegression(max_iter=200, solver="lbfgs")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

Nếu máy yếu, thử tập con trước để xác minh pipeline, nhưng nên lấy đủ lớp thay vì chỉ cắt đầu dữ liệu một cách mù quáng.

---

## 19. Nhìn ảnh đoán sai để hiểu mô hình

```python
import matplotlib.pyplot as plt
import numpy as np

wrong_indexes = np.flatnonzero(y_pred != y_test)

for index in wrong_indexes[:9]:
    plt.imshow(X_test[index].reshape(28, 28), cmap="gray")
    plt.title(f"Thật={y_test[index]}, đoán={y_pred[index]}")
    plt.axis("off")
    plt.show()
```

Hỏi:

- Ảnh có viết mờ hoặc lệch không?
- Cặp chữ số nào hay nhầm?
- Con người có thấy ảnh đó khó không?
- Lỗi do dữ liệu hay do biên tuyến tính quá đơn giản?

Phân tích lỗi cho biết nên cải thiện dữ liệu, feature hay model.

---

## 20. Giới hạn của Softmax tuyến tính trên ảnh

Softmax Regression sau khi flatten không biết pixel nào ở gần nhau. Nó coi 784 vị trí là 784 feature riêng.

Biên giữa hai lớp `a`, `b` là:

$$
(w_a-w_b)^Tx+(b_a-b_b)=0
$$

Đó vẫn là một siêu phẳng tuyến tính. Nó không tự học nét cong hay tính bất biến khi ảnh dịch vài pixel.

CNN thường tốt hơn vì khai thác cấu trúc không gian, nhưng Softmax vẫn là baseline quan trọng: nhanh, dễ debug và giúp xác nhận pipeline dữ liệu đúng.

---

## 21. K-NN và Softmax khác nhau ra sao?

| | K-NN | Softmax Regression |
|---|---|---|
| Fit | gần như chỉ lưu train | tối ưu `W`, `b` |
| Predict | so với nhiều mẫu train | nhân ma trận với tham số |
| Bộ nhớ | giữ gần toàn bộ train | giữ `d×C+C` tham số |
| Biên | linh hoạt, cục bộ | tuyến tính |
| Scale | rất quan trọng | hữu ích cho tối ưu |
| Dữ liệu lớn | predict có thể chậm | predict nhanh |

Không có mô hình luôn thắng. So sánh trên cùng split và metric.

---

## 22. Bài trong repo nên học theo thứ tự

1. Ba cụm Gauss 2D — nhìn được decision regions.
2. Iris — 4 feature, 3 lớp.
3. Glass — nhiều lớp và mất cân bằng.
4. 20 Newsgroups — văn bản sparse nhiều chiều.
5. MNIST — ảnh 784 chiều, 10 lớp.

Notebook: `week4/NguyenHaiAnh_24001968_TH4_knnsoftmax.ipynb`.

Đừng bắt đầu bằng MNIST nếu chưa giải thích được softmax của ba logits `[2,1,0]`.

---

## 23. Lộ trình học 5 buổi

### Buổi 1

Phân biệt binary, multiclass, multilabel; tự làm K-NN ba lớp.

### Buổi 2

Tính softmax `[2,1,0]`; kiểm tra tổng xác suất bằng 1; hiểu stable trick.

### Buổi 3

Học one-hot, cross-entropy và shape map; tự ghi shape cho Iris.

### Buổi 4

Chạy Iris sklearn rồi scratch; so confusion matrix.

### Buổi 5

Đọc MNIST, hiển thị ảnh, chạy baseline và phân tích ít nhất 9 ảnh sai.

---

## 24. Lỗi người mới thường mắc

- Nhầm multiclass với multilabel.
- Xem logits là xác suất.
- Tính softmax không trừ max gây overflow.
- Trừ một max chung cho cả batch thay vì từng hàng.
- Dùng sai `axis`, tổng xác suất theo mẫu không bằng 1.
- Nhãn không liên tiếp nhưng dùng trực tiếp làm index one-hot.
- Trộn dữ liệu dạng `(n,d)` và `(d,n)`.
- Fit scaler bằng cả train/test.
- Chỉ nhìn Accuracy khi các lớp lệch.
- Chuyển ma trận văn bản sparse lớn thành dense gây hết RAM.
- Đọc IDX sai endian hoặc không kiểm tra header.

---

## 25. Câu hỏi tự kiểm tra

1. Multiclass khác binary thế nào?
2. Multiclass khác multilabel thế nào?
3. K-NN nhiều lớp bỏ phiếu ra sao?
4. Logit có phải xác suất không?
5. Softmax đảm bảo hai điều gì?
6. Vì sao trừ max không đổi kết quả?
7. One-hot của lớp 2 trong bốn lớp là gì?
8. Cross-entropy nhìn vào xác suất nào?
9. `P-Y` có trực giác gì?
10. Với Iris, `W` có shape nào?
11. `argmax` chạy `axis=1` nghĩa là gì?
12. MNIST có bao nhiêu feature sau flatten?
13. Vì sao chia pixel cho 255?
14. Confusion matrix nhiều lớp đọc thế nào?
15. Softmax tuyến tính hạn chế gì trên ảnh?

---

## 26. Tóm tắt Week 4 trong một trang

```text
BÀI TOÁN
    Mỗi mẫu thuộc đúng một trong C lớp.

K-NN NHIỀU LỚP
    Tìm k láng giềng.
    Đếm phiếu từng lớp.
    Chọn lớp nhiều phiếu nhất.

SOFTMAX REGRESSION
    X @ W + b → C logits mỗi mẫu
    softmax     → C xác suất tổng bằng 1
    argmax      → lớp dự đoán

HUẤN LUYỆN
    One-hot biểu diễn lớp đúng.
    Cross-entropy phạt xác suất thấp ở lớp đúng.
    Gradient dùng P-Y.

MNIST
    ảnh 28×28 → vector 784
    pixel / 255 → 0..1
    10 lớp → W shape (784,10)

ĐÁNH GIÁ
    Confusion matrix + metric từng lớp
    + macro F1 + xem trực tiếp mẫu sai.
```

---

## 27. Từ điển thuật ngữ

| Thuật ngữ | Nghĩa ngắn gọn |
|---|---|
| Multiclass | một trong nhiều lớp |
| Multilabel | một mẫu có nhiều nhãn cùng lúc |
| Logit | score thô trước softmax |
| Softmax | đổi nhiều score thành xác suất tổng 1 |
| One-hot | vector có một số 1 tại lớp đúng |
| Categorical cross-entropy | loss cho nhiều lớp |
| Argmax | vị trí giá trị lớn nhất |
| Overflow | số lớn vượt khả năng biểu diễn |
| Stable softmax | softmax sau khi trừ max từng mẫu |
| Flatten | trải ma trận ảnh thành vector |
| Pixel | điểm ảnh |
| Sparse matrix | ma trận phần lớn là số 0 |
| Macro average | trung bình coi các lớp ngang nhau |
| Weighted average | trung bình theo số mẫu mỗi lớp |

---

## 28. Checklist hoàn thành Week 4

- [ ] Tôi phân biệt binary, multiclass và multilabel.
- [ ] Tôi tự tính được K-NN ba lớp.
- [ ] Tôi giải thích được logit và softmax.
- [ ] Tôi tính được softmax `[2,1,0]`.
- [ ] Tôi hiểu stable softmax và đúng `axis`.
- [ ] Tôi tạo được one-hot.
- [ ] Tôi biết shape `X`, `W`, `b`, `P`, `Y`.
- [ ] Tôi chạy được Iris multiclass.
- [ ] Tôi đọc được file MNIST và hiển thị ảnh.
- [ ] Tôi phân tích confusion matrix và ảnh đoán sai.

