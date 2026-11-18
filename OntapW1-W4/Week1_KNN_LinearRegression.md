# Week 1 — K-NN và Linear Regression cho người bắt đầu từ số 0

> Mục tiêu không phải là học thuộc công thức. Mục tiêu là hiểu máy dự đoán **một con số** như thế nào, vì sao dự đoán có thể sai, và cách kiểm tra mô hình có thật sự hữu ích hay không.

---

## 0. Cách học tài liệu này

Học theo ba lượt:

1. **Hiểu bài toán:** đọc mục 1–7, chưa cần code.
2. **Tính bằng tay:** làm lại ví dụ K-NN và đường thẳng ở mục 5, 8.
3. **Thực hành:** chạy mục 12 rồi làm bài trong `Week1/work/`.

Sau Week 1, bạn cần:

- Phân biệt hồi quy với phân loại.
- Hiểu `X`, `y`, train và test.
- Tự dự đoán một mẫu bằng K-NN Regression.
- Hiểu Linear Regression đang tìm đường thẳng nào.
- Đọc được MAE, MSE, RMSE và R².
- Biết khi nào K-NN hoặc Linear Regression có thể thất bại.

---

## 1. Machine Learning dùng để làm gì?

Ta có dữ liệu quá khứ gồm đầu vào và đáp án. Máy tìm quy luật từ dữ liệu đó để dự đoán trường hợp mới.

Ví dụ `SAT_GPA.csv`:

| SAT | GPA |
|---:|---:|
| 1714 | 2.40 |
| 1664 | 2.52 |
| ... | ... |

Ta muốn dùng điểm SAT để dự đoán GPA.

- Một sinh viên là một **mẫu**.
- SAT là **feature/đặc trưng** đầu vào.
- GPA là **target/giá trị cần dự đoán**.
- Thuật toán học từ các cặp SAT–GPA cũ là **model/mô hình**.

### Hồi quy và phân loại

| Câu hỏi | Loại bài toán |
|---|---|
| Giá căn nhà là bao nhiêu? | Hồi quy |
| GPA dự kiến là bao nhiêu? | Hồi quy |
| Email có phải spam không? | Phân loại |
| Hoa thuộc loài nào? | Phân loại |

Nếu đáp án là một số liên tục như `3.42`, `78.5`, `2 tỷ`, đó thường là **regression/hồi quy**.

---

## 2. Dữ liệu `X` và `y`

Trong code Machine Learning:

- `X`: bảng các feature dùng để dự đoán.
- `y`: đáp án thật.

```python
import pandas as pd

df = pd.read_csv("Week1/guide/SAT_GPA.csv")
X = df[["SAT"]]
y = df["GPA"]

print(X.shape)  # (số mẫu, số feature)
print(y.shape)  # (số mẫu,)
```

Vì sao `df[["SAT"]]` có hai cặp ngoặc vuông?

- `df["SAT"]` tạo một Series một chiều.
- `df[["SAT"]]` tạo bảng hai chiều mà sklearn mong đợi.

### Train và test

Ta không cho mô hình học toàn bộ dữ liệu rồi tự chấm trên chính dữ liệu đó. Cần tách:

- **Train:** mô hình được nhìn cả đầu vào và đáp án để học.
- **Test:** mô hình chỉ nhìn đầu vào, dự đoán rồi mới so với đáp án.

Ví dụ đời thường: train là bộ bài đã luyện; test là đề thi chưa thấy.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

`test_size=0.2` giữ 20% để kiểm tra. `random_state=42` giúp lần chạy sau chia giống lần trước; số 42 không có ý nghĩa toán học đặc biệt.

---

## 3. Hai cách dự đoán của Week 1

### Cách 1 — hỏi những người gần giống: K-NN

Muốn dự đoán GPA của sinh viên SAT 1700, tìm vài sinh viên có SAT gần 1700 nhất rồi lấy GPA trung bình.

### Cách 2 — tìm xu hướng chung: Linear Regression

Vẽ SAT trên trục ngang, GPA trên trục dọc, rồi tìm một đường thẳng đi gần các điểm nhất. Dùng đường đó để dự đoán.

```text
K-NN:              “Những trường hợp gần bạn có kết quả thế nào?”
Linear Regression: “Xu hướng chung giữa đầu vào và đầu ra là gì?”
```

Hai cách nhìn khác nhau nên có điểm mạnh và yếu khác nhau.

---

## 4. K-NN Regression hoạt động thế nào?

K-NN là viết tắt của **K Nearest Neighbors** — `K` láng giềng gần nhất.

Khi dự đoán một mẫu mới:

1. Đo khoảng cách từ mẫu mới đến mọi mẫu train.
2. Sắp xếp từ gần đến xa.
3. Lấy `k` mẫu gần nhất.
4. Lấy trung bình target của chúng.

K-NN không tìm một phương trình cố định. Nó giữ dữ liệu train và tra cứu lúc dự đoán, nên được gọi là **lazy learning**.

---

## 5. K-NN tính bằng tay

Dữ liệu:

| Số giờ học `x` | Điểm `y` |
|---:|---:|
| 1 | 4 |
| 2 | 5 |
| 4 | 7 |
| 7 | 9 |

Cần dự đoán điểm của người học `x_new=3`, chọn `k=2`.

### Bước 1 — tính khoảng cách

Một chiều nên khoảng cách là trị tuyệt đối:

| Mẫu train | Khoảng cách tới 3 |
|---:|---:|
| 1 | `|1-3|=2` |
| 2 | `|2-3|=1` |
| 4 | `|4-3|=1` |
| 7 | `|7-3|=4` |

### Bước 2 — lấy hai hàng gần nhất

Hai láng giềng là `x=2` và `x=4`, có điểm `5` và `7`.

### Bước 3 — lấy trung bình

$$
\hat y=(5+7)/2=6
$$

Dấu mũ trên `y_hat` nghĩa là giá trị dự đoán, không phải đáp án thật.

### Code đúng chính xác các bước trên

```python
import numpy as np

x_train = np.array([1, 2, 4, 7])
y_train = np.array([4, 5, 7, 9])
x_new = 3
k = 2

distances = np.abs(x_train - x_new)
nearest_indexes = np.argsort(distances)[:k]
prediction = np.mean(y_train[nearest_indexes])

print(distances)        # [2, 1, 1, 4]
print(nearest_indexes)  # chỉ số của x=2 và x=4
print(prediction)       # 6.0
```

`np.argsort` trả về **chỉ số sắp xếp**, không trả khoảng cách đã sắp xếp.

---

## 6. Khoảng cách nhiều chiều và tại sao cần scale

Nếu dự đoán giá nhà bằng diện tích và số phòng:

```text
Nhà A: diện tích 100 m², 3 phòng
Nhà B: diện tích 120 m², 4 phòng
```

Khoảng cách Euclid:

$$
d(x,z)=\sqrt{(x_1-z_1)^2+(x_2-z_2)^2+\cdots}
$$

Trong NumPy, mỗi hàng là một mẫu:

```python
distances = np.linalg.norm(X_train - x_new, axis=1)
```

`axis=1` nghĩa là tính qua các cột feature để thu được một khoảng cách cho mỗi hàng.

### Vấn đề đơn vị

Giả sử hai feature:

- Diện tích: từ 30 đến 300.
- Số phòng: từ 1 đến 5.

Chênh lệch diện tích sẽ lấn át số phòng chỉ vì đơn vị lớn hơn. K-NN sẽ hiểu sai “gần”.

StandardScaler đổi mỗi feature về thang đo có trung bình gần 0 và độ lệch chuẩn gần 1:

$$
z=\frac{x-mean}{standard\ deviation}
$$

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Chỉ `fit` scaler trên train. Nếu dùng cả test để tính mean/standard deviation, thông tin test đã rò rỉ vào quá trình học — gọi là **data leakage**.

---

## 7. Chọn `k` và giới hạn của K-NN

### `k` nhỏ

`k=1` sao chép target của đúng một láng giềng. Mô hình rất nhạy với nhiễu: **variance cao**, dễ overfit.

### `k` lớn

Nhiều láng giềng làm dự đoán mượt hơn nhưng có thể lấy cả các mẫu không còn giống: **bias cao**, dễ underfit.

Không chọn `k` bằng test set. Dùng validation hoặc cross-validation.

### K-NN khó ngoại suy

Nếu GPA lớn nhất trong các láng giềng train là 3.5, trung bình của chúng không thể vượt xa 3.5. K-NN mạnh ở nội suy trong vùng có dữ liệu, yếu ngoài miền đã thấy.

Đây là điểm quan trọng trong bài SAT–GPA của repo: khi validation chứa nhóm GPA cao hơn train, đường thẳng có thể ngoại suy còn K-NN bị giới hạn bởi các target đã lưu.

---

## 8. Linear Regression từ trực giác

Ta muốn tìm đường thẳng:

$$
\hat y=w_0+w_1x
$$

- `w0`: intercept — giá trị dự đoán khi `x=0`.
- `w1`: slope — `x` tăng một đơn vị thì dự đoán thay đổi bao nhiêu.

Ví dụ:

$$
GPA=1.2+0.0012\times SAT
$$

Nếu SAT tăng 100 điểm, GPA dự đoán tăng `0.0012 × 100 = 0.12`.

Đây là diễn giải của mô hình, không chứng minh SAT **gây ra** GPA.

### Ví dụ đường thẳng hoàn hảo

| `x` | `y` |
|---:|---:|
| 1 | 3 |
| 2 | 5 |
| 3 | 7 |

Các điểm nằm đúng trên:

$$
y=1+2x
$$

Vậy `w0=1`, `w1=2`. Với `x=4`, dự đoán `9`.

Dữ liệu thật không nằm hoàn hảo trên một đường. Ta cần định nghĩa “đường tốt nhất”.

---

## 9. Sai số và đường thẳng tốt nhất

Với mẫu `i`:

$$
e_i=y_i-\hat y_i
$$

`e_i` gọi là residual/phần dư:

- Dương: mô hình dự đoán thấp hơn thực tế.
- Âm: mô hình dự đoán cao hơn thực tế.

Nếu cộng residual thô, số dương và âm triệt tiêu. Linear Regression bình phương chúng:

$$
SSE=\sum_i(y_i-\hat y_i)^2
$$

Mô hình chọn `w0`, `w1` làm SSE nhỏ nhất. Đây là **Ordinary Least Squares — OLS**.

Bình phương làm lỗi lớn bị phạt mạnh hơn. Vì vậy ngoại lệ có thể kéo đường hồi quy đáng kể.

---

## 10. Công thức một biến — hiểu sau trực giác

Với một feature:

$$
w_1=\frac{\sum_i(x_i-\bar x)(y_i-\bar y)}
{\sum_i(x_i-\bar x)^2}
$$

$$
w_0=\bar y-w_1\bar x
$$

Trực giác:

- Tử số đo `x` và `y` cùng thay đổi thế nào.
- Mẫu số đo `x` tự thay đổi nhiều đến đâu.
- Đường OLS luôn đi qua điểm trung bình `(x_bar, y_bar)`.

```python
x_mean = x_train.mean()
y_mean = y_train.mean()

w1 = ((x_train - x_mean) * (y_train - y_mean)).sum()
w1 /= ((x_train - x_mean) ** 2).sum()
w0 = y_mean - w1 * x_mean
```

### Nhiều feature

$$
\hat y=w_0+w_1x_1+\cdots+w_dx_d
$$

Trong code số học nên dùng `np.linalg.lstsq` hoặc sklearn thay vì tự nghịch đảo `X.T @ X`, vì ma trận có thể suy biến hoặc kém ổn định.

---

## 11. Đánh giá hồi quy bằng gì?

Giả sử đáp án thật `[3, 5, 8]`, dự đoán `[2, 6, 8]`. Sai số tuyệt đối là `[1, 1, 0]`.

### MAE — sai trung bình bao nhiêu đơn vị?

$$
MAE=\frac{|1|+|{-1}|+|0|}{3}=0.667
$$

Dễ hiểu nhất vì cùng đơn vị với target.

### MSE — phạt lỗi lớn mạnh hơn

$$
MSE=\frac{1^2+(-1)^2+0^2}{3}=0.667
$$

### RMSE — căn MSE để về đơn vị gốc

$$
RMSE=\sqrt{MSE}
$$

### R² — tốt hơn dự đoán trung bình bao nhiêu?

$$
R^2=1-\frac{\sum(y_i-\hat y_i)^2}
{\sum(y_i-\bar y)^2}
$$

- `R²=1`: hoàn hảo.
- `R²=0`: ngang với luôn đoán trung bình.
- `R²<0`: tệ hơn baseline trung bình trên tập đánh giá.

Không có một ngưỡng “R² tốt” chung cho mọi lĩnh vực.

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)
```

---

## 12. Bài SAT–GPA hoàn chỉnh

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("Week1/guide/SAT_GPA.csv")
X = df[["SAT"]]
y = df["GPA"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=5),
    ),
    "Linear Regression": LinearRegression(),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print("\n", name)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", mse ** 0.5)
    print("R2:", r2_score(y_test, y_pred))
```

### Đọc code theo quy trình

1. Đọc bảng.
2. Tách feature và target.
3. Chia train/test.
4. Tạo hai model.
5. `fit` chỉ bằng train.
6. `predict` test.
7. So với đáp án test bằng cùng bộ metric.

Pipeline bảo đảm scaler chỉ học train rồi áp dụng cùng phép biến đổi cho test.

---

## 13. Tự viết hai mô hình

### K-NN nhiều chiều

```python
import numpy as np

def knn_regression_predict(X_train, y_train, X_test, k=5):
    if not 1 <= k <= len(X_train):
        raise ValueError("k phải từ 1 đến số mẫu train")

    predictions = []
    for x in X_test:
        distances = np.linalg.norm(X_train - x, axis=1)
        nearest = np.argsort(distances)[:k]
        predictions.append(np.mean(y_train[nearest]))

    return np.asarray(predictions)
```

### Linear Regression bằng least squares

```python
def fit_linear_regression(X, y):
    X_bar = np.c_[np.ones(len(X)), X]
    weights, *_ = np.linalg.lstsq(X_bar, y, rcond=None)
    return weights

def predict_linear_regression(X, weights):
    X_bar = np.c_[np.ones(len(X)), X]
    return X_bar @ weights
```

Nếu `X` có shape `(n, d)`, `X_bar` có shape `(n, d+1)`: cột đầu toàn số 1 dành cho intercept.

---

## 14. Correlation không phải causation

Pearson correlation `r` đo liên hệ **tuyến tính** và nằm trong `[-1,1]`:

- Gần `1`: cùng tăng theo xu hướng tuyến tính.
- Gần `-1`: một tăng, một giảm.
- Gần `0`: không có liên hệ tuyến tính rõ; vẫn có thể có quan hệ cong.

```python
print(df[["SAT", "GPA"]].corr())
```

SAT tương quan với GPA không chứng minh SAT gây ra GPA. Năng lực học tập, điều kiện gia đình hoặc nhiều yếu tố khác có thể ảnh hưởng cả hai.

---

## 15. Cách làm thí nghiệm đúng

Trước khi tuyên bố model nào tốt hơn, hỏi:

1. Hai model có dùng cùng train/test split không?
2. Preprocessing có chỉ fit trên train không?
3. Có baseline như luôn đoán trung bình train không?
4. Kết quả có ổn định qua nhiều split/cross-validation không?
5. Test có đại diện tình huống triển khai không?
6. Residual có tạo hình cong, hình phễu hoặc nhóm lỗi rõ ràng không?

```python
import matplotlib.pyplot as plt

residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(0, color="black")
plt.xlabel("Giá trị dự đoán")
plt.ylabel("Residual")
plt.show()
```

Nếu residual còn hình dạng rõ, mô hình chưa nắm hết cấu trúc dữ liệu.

---

## 16. Bài trong repo nên học theo thứ tự

1. `Week1/work/bai_tap_B3.py` — SAT–GPA, so Linear với K-NN.
2. `Week1/work/bai_tap_B1.py` — mức tiêu thụ nhiên liệu.
3. `Week1/work/bai_tap_B4.py` — dữ liệu y tế.
4. `Week1/work/bai_tap_B5.py` — giá bất động sản nhiều feature.

Mỗi bài phải trả lời được:

- Một hàng dữ liệu là gì?
- Feature và target là cột nào?
- Chia train/test ra sao?
- Có cần scale không?
- Metric nói gì bằng đơn vị thực tế?
- Mô hình sai nhiều ở đâu?

---

## 17. Lộ trình học 5 buổi

### Buổi 1

Đọc mục 1–5, tự tính ví dụ K-NN trên giấy.

### Buổi 2

Học scale, thay đơn vị diện tích từ m² sang cm² và giải thích vì sao K-NN thay đổi nếu không scale.

### Buổi 3

Vẽ ba điểm, tự tìm đường thẳng đơn giản và tính residual/MAE.

### Buổi 4

Chạy pipeline SAT–GPA, in từng shape, so K-NN với Linear Regression.

### Buổi 5

Làm `bai_tap_B3.py` không nhìn lời giải; viết kết luận bằng ngôn ngữ đời thường.

---

## 18. Lỗi người mới thường mắc

- Không biết bài toán cần số liên tục hay nhãn lớp.
- Truyền `X` một chiều vào sklearn.
- Quên scale trước K-NN nhiều feature.
- Fit scaler bằng cả train và test.
- Nghĩ `np.argsort` trả giá trị thay vì chỉ số.
- Chọn `k` bằng test set.
- Đánh giá trên dữ liệu đã train.
- Dùng duy nhất R² rồi kết luận.
- Nghĩ correlation hoặc hệ số hồi quy chứng minh nhân quả.
- Dùng K-NN để ngoại suy xa khỏi miền train.

---

## 19. Câu hỏi tự kiểm tra

1. Hồi quy khác phân loại ở đâu?
2. `X` và `y` chứa gì?
3. Vì sao phải có test set?
4. K-NN dự đoán một mẫu qua bốn bước nào?
5. `k` quá nhỏ và quá lớn gây gì?
6. Vì sao K-NN cần scale?
7. K-NN có học phương trình không?
8. `w0`, `w1` nghĩa là gì?
9. Residual là gì?
10. Vì sao OLS bình phương sai số?
11. MAE và RMSE khác nhau thế nào?
12. R² âm có thể xảy ra không?
13. Mô hình nào có thể ngoại suy tốt hơn trong SAT–GPA?
14. Data leakage là gì?
15. Tương quan cao có chứng minh nhân quả không?

---

## 20. Tóm tắt Week 1 trong một trang

```text
BÀI TOÁN HỒI QUY
    Dự đoán một số liên tục.

K-NN REGRESSION
    Tìm k mẫu gần nhất.
    Lấy trung bình target.
    Cần scale.
    Mạnh ở nội suy, yếu ở ngoại suy.

LINEAR REGRESSION
    Tìm đường/mặt phẳng làm tổng sai số bình phương nhỏ nhất.
    y_hat = w0 + w1*x1 + ...
    Có thể ngoại suy nhưng chỉ hợp khi xu hướng gần tuyến tính.

ĐÁNH GIÁ
    MAE  → sai trung bình bao nhiêu đơn vị.
    RMSE → phạt lỗi lớn mạnh hơn.
    R²   → tốt hơn baseline trung bình đến đâu.

QUY TRÌNH
    đọc dữ liệu → tách X/y → chia train/test
    → fit preprocessing trên train → fit model
    → predict test → metric + phân tích residual
```

---

## 21. Từ điển thuật ngữ

| Thuật ngữ | Nghĩa ngắn gọn |
|---|---|
| Regression | dự đoán số liên tục |
| Feature | thông tin đầu vào |
| Target | số cần dự đoán |
| Train set | dữ liệu dùng để học |
| Test set | dữ liệu chưa thấy để kiểm tra |
| Distance | độ khác nhau giữa hai mẫu |
| Neighbor | mẫu train gần mẫu mới |
| Scaling | đưa feature về thang đo tương đương |
| Intercept | giá trị dự đoán khi feature bằng 0 |
| Coefficient | mức dự đoán đổi khi feature tăng một đơn vị |
| Residual | thực tế trừ dự đoán |
| Overfit | học quá sát train, kém trên dữ liệu mới |
| Underfit | mô hình quá đơn giản, chưa học đủ |
| Data leakage | thông tin ngoài train lọt vào quá trình học |

---

## 22. Checklist hoàn thành Week 1

- [ ] Tôi phân biệt được hồi quy và phân loại.
- [ ] Tôi xác định được mẫu, feature, target, `X`, `y`.
- [ ] Tôi tự tính được một dự đoán K-NN.
- [ ] Tôi giải thích được tác động của `k` và scaling.
- [ ] Tôi hiểu Linear Regression tìm đường nào.
- [ ] Tôi diễn giải được slope, intercept và residual.
- [ ] Tôi đọc được MAE, RMSE và R² bằng lời.
- [ ] Tôi chạy được pipeline SAT–GPA.
- [ ] Tôi so sánh hai mô hình trên cùng test set.
- [ ] Tôi hoàn thành `Week1/work/bai_tap_B3.py`.

