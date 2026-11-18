# Week 3 — K-NN Classification và Logistic Regression cho người bắt đầu từ số 0

> Mục tiêu là hiểu cách máy trả lời một câu hỏi **có/không**, vì sao mô hình tạo ra xác suất, và cách biến xác suất thành quyết định phù hợp với hậu quả thực tế.

---

## 0. Cách học tài liệu này

1. Đọc mục 1–7 để hiểu bài toán bằng lời.
2. Tự tính ví dụ K-NN và sigmoid ở mục 5, 9.
3. Chạy chương trình mục 15.
4. Chỉ đọc gradient descent ở mục 13 sau khi hiểu xác suất và loss.

Sau Week 3, bạn cần:

- Hiểu phân loại nhị phân và nhãn `0/1`.
- Phân biệt xác suất, threshold và nhãn dự đoán.
- Tự làm K-NN classification.
- Hiểu vì sao Logistic Regression dùng sigmoid.
- Hiểu cross-entropy dùng để phạt điều gì.
- Đọc Precision, Recall, F1, ROC-AUC theo bài toán.

---

## 1. Bài toán phân loại nhị phân là gì?

Phân loại nhị phân có đúng hai lựa chọn:

| Bài toán | Lớp 0 | Lớp 1 |
|---|---|---|
| Sinh viên | trượt | đỗ |
| Bệnh tim | không bệnh | có bệnh |
| Ngân hàng | không gửi tiết kiệm | có gửi |
| Email | bình thường | spam |

`0` và `1` chỉ là cách mã hóa. Số 1 không luôn có nghĩa “tốt”; nó thường là lớp ta quan tâm.

Một mẫu sinh viên:

```text
feature X: số giờ ôn thi = 3.75
label y:   đỗ = 1
```

Mô hình nhận feature và dự đoán nhãn.

---

## 2. Tại sao cần xác suất trước nhãn?

Thay vì chỉ nói “đỗ”, mô hình có thể nói:

```text
xác suất đỗ = 0.82
```

Sau đó dùng một ngưỡng — **threshold**:

```text
nếu xác suất >= 0.5 → dự đoán 1
nếu xác suất <  0.5 → dự đoán 0
```

Ba khái niệm khác nhau:

- **Score:** giá trị thô của mô hình.
- **Probability:** score đã biến thành số từ 0 đến 1.
- **Prediction:** quyết định cuối sau threshold.

Threshold 0.5 là mặc định, không phải luật tự nhiên. Trong y tế, ta có thể hạ threshold để giảm bỏ sót bệnh nhân.

---

## 3. Hai cách phân loại trong Week 3

### K-NN Classification

Hỏi các mẫu gần nhất rồi lấy phiếu đa số.

### Logistic Regression

Tính một score từ các feature, đổi score thành xác suất bằng sigmoid, rồi so với threshold.

```text
K-NN:     quyết định cục bộ từ hàng xóm.
Logistic: học một ranh giới chung cho toàn bộ dữ liệu.
```

---

## 4. K-NN Classification hoạt động thế nào?

Các bước:

1. Tính khoảng cách từ mẫu mới đến mọi mẫu train.
2. Lấy `k` mẫu gần nhất.
3. Đếm nhãn 0 và 1.
4. Chọn nhãn có nhiều phiếu hơn.

Khác K-NN Regression ở bước cuối:

- Regression lấy trung bình **giá trị số**.
- Classification lấy lớp **chiếm đa số**.

---

## 5. Ví dụ K-NN tính bằng tay

| Giờ ôn | Kết quả |
|---:|---:|
| 1.0 | 0 |
| 2.0 | 0 |
| 3.0 | 1 |
| 4.0 | 1 |
| 6.0 | 1 |

Dự đoán người học `2.5` giờ với `k=3`:

| Mẫu | Khoảng cách |
|---:|---:|
| 1.0 | 1.5 |
| 2.0 | 0.5 |
| 3.0 | 0.5 |
| 4.0 | 1.5 |
| 6.0 | 3.5 |

Ba láng giềng gần nhất có thể là `2.0`, `3.0` và một trong `1.0/4.0` tùy quy tắc hòa khoảng cách. Nếu lấy `4.0`, nhãn là `[0,1,1]`, đa số là `1`.

Tỷ lệ lớp 1 trong láng giềng là `2/3≈0.667`. Có thể coi đây là ước lượng xác suất cục bộ, nhưng không nhất thiết là xác suất đã hiệu chỉnh tốt.

```python
import numpy as np

def knn_binary_predict(X_train, y_train, X_test, k=5):
    predictions = []
    neighbor_rates = []

    for x in X_test:
        distances = np.linalg.norm(X_train - x, axis=1)
        nearest = np.argsort(distances)[:k]
        rate = y_train[nearest].mean()
        neighbor_rates.append(rate)
        predictions.append(int(rate >= 0.5))

    return np.array(predictions), np.array(neighbor_rates)
```

Với hai lớp, chọn `k` lẻ giúp giảm hòa phiếu. Dữ liệu nhiều feature vẫn cần scaling như Week 1.

---

## 6. Hạn chế của K-NN Classification

- Dự đoán chậm khi train lớn vì phải đo khoảng cách tới nhiều mẫu.
- Feature khác thang đo làm khoảng cách sai lệch.
- Nhiều chiều khiến khái niệm “gần” kém rõ — curse of dimensionality.
- Vùng dữ liệu thưa làm dự đoán không ổn định.
- `k` nhỏ dễ overfit; `k` lớn dễ làm mờ nhóm nhỏ.

K-NN vẫn là baseline rất trực quan để so sánh.

---

## 7. Vì sao Linear Regression không phù hợp cho nhãn 0/1?

Nếu dùng đường thẳng để dự đoán đỗ/trượt, kết quả có thể là `-0.4` hoặc `1.7`, không thể hiểu là xác suất.

Ta cần một hàm nhận mọi số thực và nén vào khoảng `0..1`. Đó là sigmoid.

---

## 8. Logistic Regression thực sự làm gì?

Mô hình tính score tuyến tính:

$$
z=w_0+w_1x_1+\cdots+w_dx_d
$$

Sau đó đưa qua sigmoid:

$$
p=\sigma(z)=\frac{1}{1+e^{-z}}
$$

`p` được diễn giải là xác suất lớp 1.

```text
feature → score z → sigmoid → xác suất p → threshold → nhãn 0/1
```

Tên có chữ “Regression” vì mô hình hóa một đại lượng liên tục là log-odds, nhưng mục tiêu sử dụng là **classification**.

---

## 9. Hiểu sigmoid bằng số

| `z` | `sigmoid(z)` | Ý nghĩa |
|---:|---:|---|
| -4 | khoảng 0.018 | rất nghiêng lớp 0 |
| -1 | khoảng 0.269 | nghiêng lớp 0 |
| 0 | 0.5 | ở giữa |
| 1 | khoảng 0.731 | nghiêng lớp 1 |
| 4 | khoảng 0.982 | rất nghiêng lớp 1 |

```python
import numpy as np

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))
```

`np.clip` ngăn `exp` tràn số với giá trị quá lớn.

### Ranh giới quyết định

Với threshold 0.5:

```text
p >= 0.5 tương đương z >= 0
```

Biên quyết định là nơi:

$$
w_0+w^Tx=0
$$

Trong hai chiều, đó là một đường thẳng; nhiều chiều là một siêu phẳng.

---

## 10. Odds và log-odds — phần hiểu sâu

Nếu xác suất đỗ `p=0.8`, odds là:

$$
odds=\frac{p}{1-p}=\frac{0.8}{0.2}=4
$$

Nghĩa là khả năng đỗ so với trượt theo tỷ lệ 4:1.

Logistic Regression giả sử log-odds là tuyến tính:

$$
\log\frac{p}{1-p}=w_0+w^Tx
$$

Giải phương trình này theo `p` sẽ thu được sigmoid. Sigmoid không phải mẹo ngẫu nhiên; nó xuất hiện tự nhiên từ giả định log-odds tuyến tính.

Nếu `w_j=0.7`, feature `x_j` tăng một đơn vị làm odds nhân với `e^0.7≈2.01`, khi các feature khác giữ nguyên. Không nên nói xác suất tăng cố định 0.7.

---

## 11. Mô hình học bằng cách nào?

Ban đầu trọng số có thể là 0, nên mọi mẫu có xác suất 0.5. Mô hình lặp:

1. Tính xác suất hiện tại.
2. Đo dự đoán sai đến đâu bằng loss.
3. Tính hướng điều chỉnh trọng số.
4. Cập nhật một bước nhỏ.
5. Lặp lại.

Quá trình đó là **gradient descent**.

`learning_rate` quyết định độ dài bước:

- Quá lớn: loss dao động hoặc tăng.
- Quá nhỏ: học rất chậm.

---

## 12. Cross-entropy phạt điều gì?

Với một mẫu:

$$
L=-[y\log p+(1-y)\log(1-p)]
$$

Nếu nhãn thật `y=1`:

| Dự đoán `p` | Loss `-log(p)` |
|---:|---:|
| 0.9 | 0.105 |
| 0.5 | 0.693 |
| 0.1 | 2.303 |
| 0.01 | 4.605 |

Dự đoán sai mà quá tự tin bị phạt mạnh. Đây là hành vi ta muốn.

Cross-entropy xuất phát từ Maximum Likelihood: chọn trọng số làm dữ liệu nhãn đã quan sát có khả năng xảy ra cao nhất.

---

## 13. Một bước gradient descent bằng số

Một mẫu có `x=2`, thêm bias thành `[1,2]`, nhãn `y=1`, trọng số đầu `[0,0]`:

1. `z=0`.
2. `p=sigmoid(0)=0.5`.
3. Sai số xác suất `p-y=-0.5`.
4. Gradient `(p-y)x=[-0.5,-1]`.
5. Với learning rate `0.1`, trọng số mới là `[0.05,0.1]`.
6. Score mới `0.05+0.1×2=0.25`, xác suất mới khoảng `0.562`.

Vì nhãn thật là 1, xác suất đã tăng đúng hướng.

Gradient cả tập:

$$
\nabla_wL=\frac{1}{n}X^T(p-y)
$$

Cập nhật:

$$
w\leftarrow w-learning\_rate\times\nabla_wL
$$

---

## 14. Logistic Regression từ đầu

```python
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def fit_logistic_regression(X, y, learning_rate=0.1, epochs=2000):
    X_bar = np.c_[np.ones(len(X)), X]
    weights = np.zeros(X_bar.shape[1])

    for _ in range(epochs):
        probabilities = sigmoid(X_bar @ weights)
        gradient = X_bar.T @ (probabilities - y) / len(y)
        weights -= learning_rate * gradient

    return weights

def predict_proba(X, weights):
    X_bar = np.c_[np.ones(len(X)), X]
    return sigmoid(X_bar @ weights)

def predict(X, weights, threshold=0.5):
    return (predict_proba(X, weights) >= threshold).astype(int)
```

Shape nếu có `n` mẫu, `d` feature:

| Biến | Shape |
|---|---|
| `X` | `(n, d)` |
| `X_bar` | `(n, d+1)` |
| `weights` | `(d+1,)` |
| `probabilities` | `(n,)` |
| `gradient` | `(d+1,)` |

---

## 15. Workflow sklearn đúng chuẩn

Ví dụ Admission: biến `Chance of Admit` thành nhãn 1 nếu từ 0.75 trở lên.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("Week3/Admission_Predict.csv")

features = [
    "GRE Score", "TOEFL Score", "University Rating",
    "SOP", "LOR ", "CGPA", "Research",
]
X = df[features]
y = (df["Chance of Admit"] >= 0.75).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000),
)
model.fit(X_train, y_train)

y_probability = model.predict_proba(X_test)[:, 1]
y_prediction = (y_probability >= 0.5).astype(int)

print(confusion_matrix(y_test, y_prediction))
print(classification_report(y_test, y_prediction))
```

`predict_proba(... )[:,1]` lấy xác suất lớp 1. Pipeline bảo đảm scaler chỉ fit trên train.

---

## 16. Confusion matrix từ số 0

| | Dự đoán 1 | Dự đoán 0 |
|---|---:|---:|
| Thực tế 1 | TP | FN |
| Thực tế 0 | FP | TN |

Ví dụ xét nghiệm bệnh:

- TP: có bệnh, máy báo bệnh.
- FN: có bệnh, máy bỏ sót.
- FP: không bệnh, máy báo nhầm.
- TN: không bệnh, máy báo không bệnh.

### Accuracy

Trong mọi mẫu, tỷ lệ đoán đúng:

$$
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
$$

### Precision

Trong các ca máy báo dương, bao nhiêu ca thật sự dương?

$$
Precision=\frac{TP}{TP+FP}
$$

### Recall

Trong các ca dương thật, máy tìm được bao nhiêu?

$$
Recall=\frac{TP}{TP+FN}
$$

### F1

Cân bằng Precision và Recall bằng trung bình điều hòa.

Accuracy có thể đánh lừa nếu 95% mẫu là lớp 0: luôn đoán 0 đạt 95% nhưng Recall lớp 1 bằng 0.

---

## 17. Threshold là quyết định nghiệp vụ

Hạ threshold:

- Nhiều mẫu được dự đoán 1 hơn.
- Recall thường tăng.
- False positive thường tăng; Precision có thể giảm.

Tăng threshold làm xu hướng ngược lại.

Ví dụ y tế, nếu bỏ sót bệnh nhân đắt gấp 10 lần cảnh báo nhầm:

$$
Cost(t)=10\times FN(t)+1\times FP(t)
$$

Chọn threshold trên validation, khóa lại rồi mới đánh giá test. Không được nhìn test để tinh chỉnh threshold.

---

## 18. ROC-AUC là gì?

ROC xét nhiều threshold thay vì chỉ 0.5. AUC đo khả năng model xếp một mẫu dương cao hơn một mẫu âm ngẫu nhiên.

- Gần 1: xếp hạng tốt.
- Khoảng 0.5: gần đoán ngẫu nhiên.

```python
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_test, y_probability)
```

Phải truyền probability/score, không truyền nhãn 0/1. Với lớp dương rất hiếm, Precision–Recall curve thường cho cái nhìn hữu ích hơn.

---

## 19. Regularization dễ hiểu

Mô hình có thể tạo trọng số rất lớn để bám sát train. L2 regularization phạt trọng số lớn:

$$
Loss_{total}=Loss_{BCE}+\lambda\sum_jw_j^2
$$

Trong sklearn, `C` là nghịch đảo độ mạnh regularization:

- `C` nhỏ: phạt mạnh hơn.
- `C` lớn: phạt yếu hơn.

Feature nên scale để penalty giữa các hệ số công bằng hơn. Chọn `C` bằng validation/cross-validation.

---

## 20. Bài trong repo nên học theo thứ tự

1. Ví dụ sinh viên ôn thi 1D trong `Week3/test_all_week3.py`.
2. Admission Prediction — target tạo bằng ngưỡng 0.75.
3. Banking — categorical encoding và mất cân bằng.
4. Framingham — missing value, Recall bệnh tim.

Với mỗi bài, trả lời:

- Lớp 1 nghĩa là gì?
- Hậu quả FP và FN là gì?
- Feature nào cần scale/mã hóa?
- Metric nào quan trọng nhất?
- Threshold 0.5 có phù hợp không?

---

## 21. Lộ trình học 5 buổi

### Buổi 1

Hiểu nhãn 0/1, probability, prediction và threshold.

### Buổi 2

Tự tính K-NN sinh viên với vài giá trị `k`; quan sát hòa phiếu.

### Buổi 3

Tính sigmoid bằng máy tính cho `z=-2,0,2`; hiểu biên `z=0`.

### Buổi 4

Chạy Admission sklearn; in probability và thử threshold 0.3, 0.5, 0.7.

### Buổi 5

Đọc scratch code; vẽ loss theo epoch và so dự đoán với sklearn.

---

## 22. Lỗi người mới thường mắc

- Nghĩ Logistic Regression dự đoán số liên tục vì tên có “Regression”.
- Nhầm probability với prediction.
- Cho rằng threshold luôn phải là 0.5.
- Không scale khiến tối ưu chậm và K-NN sai khoảng cách.
- Fit scaler trước khi chia dữ liệu.
- Chỉ nhìn Accuracy trên dữ liệu mất cân bằng.
- Dùng nhãn dự đoán để tính ROC-AUC.
- Chọn threshold hoặc `C` bằng test set.
- Nhầm dấu gradient làm loss tăng.
- Trộn quy ước mẫu theo hàng và mẫu theo cột.

---

## 23. Câu hỏi tự kiểm tra

1. Phân loại nhị phân là gì?
2. Lớp 1 có luôn là kết quả tốt không?
3. Probability khác prediction thế nào?
4. K-NN classification bỏ phiếu ra sao?
5. Vì sao cần sigmoid?
6. `sigmoid(0)` bằng bao nhiêu?
7. Threshold 0.5 tạo biên ở score nào?
8. Logistic Regression học đại lượng gì tuyến tính?
9. Cross-entropy phạt trường hợp nào mạnh nhất?
10. Gradient descent làm gì?
11. Precision trả lời câu hỏi nào?
12. Recall trả lời câu hỏi nào?
13. Vì sao Accuracy có thể đánh lừa?
14. Hạ threshold thường ảnh hưởng Recall ra sao?
15. ROC-AUC cần probability hay nhãn?

---

## 24. Tóm tắt Week 3 trong một trang

```text
BÀI TOÁN
    Chọn một trong hai lớp 0/1.

K-NN CLASSIFICATION
    Tìm k láng giềng.
    Lấy phiếu đa số.

LOGISTIC REGRESSION
    feature → score tuyến tính z
    → sigmoid → probability
    → threshold → nhãn

HUẤN LUYỆN
    Cross-entropy phạt dự đoán sai tự tin.
    Gradient descent cập nhật trọng số để giảm loss.

ĐÁNH GIÁ
    Accuracy: đúng tổng thể.
    Precision: báo dương đáng tin đến đâu.
    Recall: tìm được bao nhiêu dương thật.
    F1: cân bằng Precision/Recall.
    ROC-AUC: khả năng xếp hạng qua nhiều threshold.
```

---

## 25. Từ điển thuật ngữ

| Thuật ngữ | Nghĩa ngắn gọn |
|---|---|
| Binary classification | phân loại hai lớp |
| Probability | xác suất từ 0 đến 1 |
| Threshold | ngưỡng đổi xác suất thành nhãn |
| Score/logit | giá trị thô trước sigmoid |
| Sigmoid | hàm nén score về 0..1 |
| Decision boundary | nơi mô hình đổi lớp dự đoán |
| Loss | con số đo dự đoán tệ đến đâu |
| Gradient | hướng loss tăng nhanh nhất |
| Learning rate | kích thước bước cập nhật |
| Epoch | một lượt học qua dữ liệu |
| False positive | báo dương nhầm |
| False negative | bỏ sót mẫu dương |
| Regularization | hạn chế mô hình bám quá sát train |

---

## 26. Checklist hoàn thành Week 3

- [ ] Tôi xác định được lớp 0 và lớp 1.
- [ ] Tôi phân biệt được probability, threshold và prediction.
- [ ] Tôi tự tính được một dự đoán K-NN classification.
- [ ] Tôi giải thích được sigmoid bằng lời.
- [ ] Tôi hiểu score `z=0` tương ứng xác suất 0.5.
- [ ] Tôi hiểu cross-entropy phạt điều gì.
- [ ] Tôi đọc được TP, TN, FP, FN.
- [ ] Tôi chọn Precision/Recall theo hậu quả thực tế.
- [ ] Tôi chạy được Admission pipeline.
- [ ] Tôi hoàn thành ít nhất một bài thật trong `Week3/`.

