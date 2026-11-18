# BÁO CÁO THỰC HÀNH MACHINE LEARNING: BÀI THỰC HÀNH SỐ 2
## Chủ đề: Thuật toán K-Nearest Neighbors (K-NN) và Hồi quy tuyến tính (Linear Regression)

**Họ và tên sinh viên**: Nguyễn Hải Anh  
**Mã số sinh viên (MSSV)**: 24001968  

---

## MỤC LỤC
1. [Giới thiệu & Cơ sở lý thuyết](#1-giới-thiệu--cơ-sở-lý-thuyết)
2. [Cấu trúc thư mục & Dữ liệu bài tập](#2-cấu-trúc-thư-mục--dữ-liệu-bài-tập)
3. [Lưu ý quan trọng về dữ liệu và thuật toán trong tài liệu](#3-lưu-ý-quan-trọng-về-dữ-liệu-và-thuật-toán-trong-tài-liệu)
4. [Kết quả thực hiện Bài tập 1 (Bài tập B.1 - Nhiên liệu)](#4-kết-quả-thực-hiện-bài-tập-1-bài-tập-b1---nhiên-liệu)
5. [Kết quả thực hiện Bài tập 2 (Bài tập tự thực hành 1 / B.3 - SAT & GPA)](#5-kết-quả-thực-hiện-bài-tập-2-bài-tập-tự-thực-hành-1--b3---sat--gpa)
6. [Kết quả thực hiện Bài tập 3 (Bài tập 2 / B.4 - Độ dày NTM)](#6-kết-quả-thực-hiện-bài-tập-3-bài-tập-2--b4---độ-dày-ntm)
7. [Kết quả thực hiện Bài tập 4 (Bài tập 3 / B.5 - Giá bất động sản)](#7-kết-quả-thực-hiện-bài-tập-4-bài-tập-3--b5---giá-bất-động-sản)
8. [Tổng kết kết quả thực nghiệm & Trả lời câu hỏi vấn đáp](#8-tổng-kết-kết-quả-thực-nghiệm--trả-lời-câu-hỏi-vấn-đáp)

---

## 1. Giới thiệu & Cơ sở lý thuyết

Bài thực hành số 2 tập trung vào hai thuật toán cơ bản và nền tảng nhất trong Machine Learning cho bài toán dự báo liên tục (Hồi quy - Regression):

### 1.1. Thuật toán K-Nearest Neighbors (K-NN) cho Hồi quy
- **Ý tưởng cốt lõi**: Điểm dữ liệu chưa biết $x$ sẽ có giá trị đầu ra gần giống với các điểm huấn luyện gần nó nhất trong không gian đặc trưng.
- **Các bước thực hiện**:
  1. Với mẫu kiểm thử $x$, tính khoảng cách từ $x$ đến tất cả các mẫu $x_i$ trong tập huấn luyện:
     - Biến đơn ($d = 1$): $d(x, x_i) = |x - x_i|$
     - Đa biến ($d > 1$): Khoảng cách Euclid (chuẩn 2)
       $$d(\mathbf{x}, \mathbf{x}_i) = \|\mathbf{x} - \mathbf{x}_i\|_2 = \sqrt{\sum_{j=1}^d (x_j - x_{i,j})^2}$$
  2. Sắp xếp khoảng cách tăng dần, chọn ra $K$ mẫu có khoảng cách nhỏ nhất.
  3. Giá trị dự đoán là trung bình cộng đầu ra của $K$ láng giềng:
     $$\hat{y} = \frac{1}{K}\sum_{i \in \text{Neighbors}} y_i$$
- **Ưu điểm**: Đơn giản, phi tham số (non-parametric), nắm bắt tốt mối quan hệ phi tuyến tính cục bộ.
- **Nhược điểm**: Chi phí tính toán cao khi dữ liệu lớn, nhạy cảm với thang đo đặc trưng, **không có khả năng ngoại suy** (không thể dự đoán vượt ngoài dải giá trị $[y_{\min}, y_{\max}]$ của tập huấn luyện).

### 1.2. Thuật toán Hồi quy tuyến tính (Linear Regression)
- **Dạng mô hình**: Giả định quan hệ giữa các biến đầu vào $\mathbf{x} = [x_1, x_2, \dots, x_d]^T$ và đầu ra $y$ là tuyến tính:
  $$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_d x_d = \bar{\mathbf{x}}^T \mathbf{w}$$
  trong đó $\bar{\mathbf{x}} = [1, x_1, \dots, x_d]^T$ (đã bổ sung hệ số chặn bias $x_0 = 1$).
- **Phương pháp tìm tham số $\mathbf{w}$**:
  - Hồi quy đơn biến ($d=1$):
    $$w_1 = \frac{S_{XY}}{S_{XX}} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}, \quad w_0 = \bar{y} - w_1 \bar{x}$$
  - Hồi quy đa biến qua Ma trận giả nghịch đảo (Pseudo-inverse):
    $$\mathbf{w} = (\bar{\mathbf{X}}^T \bar{\mathbf{X}})^{-1} \bar{\mathbf{X}}^T \mathbf{y} = \bar{\mathbf{X}}^\dagger \mathbf{y}$$
  - Phân rã QR (Householder reflection): Giúp tránh hiện tượng ma trận kỳ dị hoặc số điều kiện lớn khi nghịch đảo trực tiếp $\bar{\mathbf{X}}^T\bar{\mathbf{X}}$.

### 1.3. Các độ đo đánh giá mô hình hồi quy
1. **MSE (Mean Squared Error - Sai số bình phương trung bình)**:
   $$\text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$
   *(Càng nhỏ càng tốt, phạt nặng các sai số lớn).*
2. **MAE (Mean Absolute Error - Sai số tuyệt đối trung bình)**:
   $$\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$$
   *(Dễ diễn giải, có cùng đơn vị đo với biến mục tiêu).*
3. **$R^2$ (R-squared - Hệ số xác định)**:
   $$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$
   *(Đo lường tỷ lệ biến thiên của $y$ được giải thích bởi mô hình. Thang đo tối đa là 1.0; nếu mô hình tệ hơn việc đoán giá trị trung bình $\bar{y}$ thì $R^2$ có thể mang giá trị âm).*
4. **RSS (Residual Sum of Squares - Tổng bình phương phần dư)**:
   $$\text{RSS} = \sum_{i=1}^n (y_i - \hat{y}_i)^2 = n \times \text{MSE}$$
5. **Kỳ vọng và Phương sai của sai số ($e_i = y_i - \hat{y}_i$)**:
   - Kỳ vọng sai số: $E[e] = \frac{1}{n}\sum e_i$ (đo lường độ lệch / bias).
   - Phương sai sai số: $\text{Var}(e) = \frac{1}{n}\sum (e_i - E[e])^2$.

---

## 2. Cấu trúc thư mục & Dữ liệu bài tập

```
MachineLearning/
└── Week1/
    ├── guide/                                   # Thư mục tài liệu gốc của môn học
    │   ├── thuchanh_bai2_K-NN - Linear_Regress.pdf
    │   ├── SAT_GPA.csv                          # Dữ liệu điểm SAT và GPA (84 sinh viên)
    │   ├── fuel.txt                             # Dữ liệu tiêu thụ nhiên liệu (51 bang)
    │   ├── vidu4_lin_reg.txt                    # Dữ liệu bề dày NTM (100 bệnh nhân)
    │   └── real_estate.csv                      # Dữ liệu giao dịch BĐS (414 căn)
    └── work/                                    # Thư mục bài làm thực hành
        ├── BAO_CAO_THUC_HANH.md                 # Báo cáo thực hành chi tiết
        ├── bai_tap_B1.py                        # Code Bài tập B.1 (Nhiên liệu)
        ├── bai_tap_B3.py                        # Code Bài tập tự thực hành 1 (SAT_GPA)
        ├── bai_tap_B4.py                        # Code Bài tập 2 (Độ dày NTM)
        ├── bai_tap_B5.py                        # Code Bài tập 3 (Bất động sản)
        ├── solve_all.py                         # Script tổng hợp chạy tất cả bài tập
        ├── hinh_B3_sat_gpa_regression.png       # Đồ thị hồi quy SAT vs GPA
        └── hinh_B5_real_estate_comparison.png   # Đồ thị so sánh giá nhà BĐS
```

---

## 3. Lưu ý quan trọng về dữ liệu và thuật toán trong tài liệu

1. **Lỗi `axis` khi tính khoảng cách Euclid đa chiều (Trang 5 PDF)**:
   - Trong code mẫu của PDF:
     ```python
     def distance(array, value):
         array = np.array(array)
         return np.linalg.norm(array - value, ord=2, axis=0) # CHÚ Ý!
     ```
   - Trong PDF ghi chú: *(Cần tự xem xét axis = 0 hay bằng 1!)*.
   - **Giải thích**: `array` có kích thước $(N, d)$ (gồm $N$ hàng, mỗi hàng là vector $d$ chiều), `value` có kích thước $(d,)$. Hiệu `array - value` có kích thước $(N, d)$. Để tính chuẩn vector cho từng hàng (khoảng cách tới từng mẫu), ta **bắt buộc phải truyền `axis=1`**. Nếu để `axis=0`, Numpy sẽ tính chuẩn theo cột và trả về vector $d$ phần tử, gây lỗi logic hoàn toàn.

2. **Dữ liệu `SAT_GPA.csv` bị sắp xếp theo GPA**:
   - File `SAT_GPA.csv` đã được xếp tăng dần theo cột GPA (từ 2.40 đến 3.81).
   - Khi chia tuần tự: 64 mẫu đầu làm Train (GPA $2.40 \to 3.50$), 20 mẫu sau làm Validation (GPA $3.51 \to 3.81$).
   - Điều này làm cho tập Validation toàn các giá trị GPA cao và có phương sai nội bộ rất nhỏ ($\text{SS}_{\text{tot}}$ nhỏ), khiến hệ số $R^2$ trên tập Validation bị âm. Đây là hiện tượng phân phối dữ liệu bị lệch (Distribution Shift), giải thích rất rõ vì sao K-NN hoàn toàn thất bại trong khi Hồi quy tuyến tính vẫn dự đoán tốt hơn nhờ tính năng ngoại suy.

---

## 4. Kết quả thực hiện Bài tập 1 (Bài tập B.1 - Nhiên liệu)

### 4.1. Đề bài
> Sử dụng 40 mẫu trong tập dữ liệu của ví dụ 2 (tệp `fuel.txt`, coi là tập training) để huấn luyện các đoạn chương trình đã có, sau đó dự đoán đầu ra với các mẫu dữ liệu còn lại (coi là tập validation). Hãy dùng các độ đo sau đây để đánh giá độ chính xác của mô hình trên các tập training và validation: MSE; MAE; R-square.

### 4.2. Quy đổi các biến từ `fuel.txt`
Dữ liệu gồm các trường: `Drivers`, `FuelC`, `Income`, `Miles`, `Pop`, `Tax`.
- Đầu ra mục tiêu:
  $$\text{Fuel} = 1000 \times \frac{\text{FuelC}}{\text{Pop}}$$
- Các thuộc tính đầu vào:
  - $\text{Tax}$: Thuế xăng (cents/gallon)
  - $\text{Dlic} = 1000 \times \frac{\text{Drivers}}{\text{Pop}}$: Số bằng lái trên 1000 dân
  - $\text{Income}$: Thu nhập bình quân đầu người
  - $\log_2(\text{Miles})$: Logarit cơ số 2 của chiều dài đường cao tốc

### 4.3. Mã nguồn thực hiện (`bai_tap_B1.py`)

```python
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Đọc và tiền xử lý dữ liệu
data_path = r"d:\MachineLearning\Week1\guide\fuel.txt"
with open(data_path, "r", encoding="utf-8") as f:
    lines = f.readlines()
lines.pop(0) # Bỏ tiêu đề

x_data, y_data = [], []
for line in lines:
    s = line.replace('\n', '').replace('"', '').split(',')
    drivers, fuelc, income = float(s[1]), float(s[2]), float(s[3])
    miles, pop, tax = float(s[4]), float(s[6]), float(s[7])
    
    fuel = 1000 * fuelc / pop
    dlic = 1000 * drivers / pop
    logMiles = math.log2(miles)
    
    y_data.append(fuel)
    x_data.append([tax, dlic, income, logMiles])

X = np.array(x_data)
y = np.array(y_data)

# 2. Chia tập: 40 mẫu đầu (Training), 11 mẫu sau (Validation)
X_train, y_train = X[:40], y[:40]
X_val, y_val = X[40:], y[40:]

# 3. Huấn luyện mô hình
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Dự đoán và đánh giá
pred_train = model.predict(X_train)
pred_val = model.predict(X_val)

def evaluate(y_true, y_pred, name):
    print(f"--- Đánh giá trên tập {name} ---")
    print(f"MSE : {mean_squared_error(y_true, y_pred):.4f}")
    print(f"MAE : {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"R^2 : {r2_score(y_true, y_pred):.4f}\n")

evaluate(y_train, pred_train, "TRAINING (40 mẫu)")
evaluate(y_val, pred_val, "VALIDATION (11 mẫu)")
```

### 4.4. Kết quả & Đáp số chính xác

#### Phương trình hồi quy tìm được:
$$\text{Fuel} = 135.1420 - 1.9030 \cdot \text{Tax} + 0.4186 \cdot \text{Dlic} - 0.0062 \cdot \text{Income} + 19.5939 \cdot \log_2(\text{Miles})$$

#### Bảng đánh giá độ chính xác:
| Độ đo | Tập Training (40 mẫu) | Tập Validation (11 mẫu) |
| :--- | :---: | :---: |
| **MSE** | **3344.7858** | **6253.3626** |
| **MAE** | **44.2097** | **60.6155** |
| **$R^2$** | **0.5516** | **-0.0242** |

#### Nhận xét:
- Trên tập huấn luyện, mô hình đạt $R^2 \approx 0.5516$, giải thích được khoảng 55.2% sự biến thiên của mức tiêu thụ nhiên liệu.
- Trên tập validation, sai số MSE và MAE tăng lên (MSE từ 3344 lên 6253), hệ số $R^2$ giảm nhẹ xuống âm (-0.0242) do kích thước tập validation nhỏ (11 mẫu) và có 1 bang có mức tiêu thụ đột biến cao (bang số 11 thực tế đạt 842.79 trong khi dự đoán chỉ là 633.96).

---

## 5. Kết quả thực hiện Bài tập 2 (Bài tập tự thực hành 1 / B.3 - SAT & GPA)

### 5.1. Đề bài
> Sử dụng lại tệp dữ liệu cũng như đoạn lệnh đọc dữ liệu từ Ví dụ A.1 (`SAT_GPA.csv`). Thực hiện các yêu cầu:
> (i) Sử dụng hệ số tương quan Pearson correlation để đánh giá mức độ phụ thuộc của đầu ra (GPA) với đầu vào (SAT).
> (ii) Chia dữ liệu thành phần training với 64 mẫu đầu và validation với các mẫu còn lại.
> (iii) Lập công thức hồi quy tuyến tính ứng với dữ liệu training. Hiển thị đồ thị của đường hồi quy với các điểm dữ liệu.
> (iv) Chạy thử mô hình với dữ liệu validation và tính MSE; MAE; R-square.
> (v) So sánh kết quả này với kết quả sử dụng phương pháp K-NN trong Ví dụ A.1.

### 5.2. Công thức toán học
1. **Hệ số tương quan Pearson ($r$)**:
   $$r_{XY} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$$
2. **Hệ số hồi quy đơn biến**:
   $$w_1 = \frac{S_{XY}}{S_{XX}} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}, \quad w_0 = \bar{y} - w_1 \bar{x}$$

### 5.3. Mã nguồn thực hiện (`bai_tap_B3.py`)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv(r"d:\MachineLearning\Week1\guide\SAT_GPA.csv")
x = data['SAT'].values
y = data['GPA'].values

# (i) Pearson correlation
pearson_r = np.corrcoef(x, y)[0, 1]
print(f"Pearson r: {pearson_r:.5f}")

# (ii) Split Train/Val: 64/20
X_train, y_train = x[:64], y[:64]
X_val, y_val = x[64:], y[64:]

# (iii) Linear Regression
lr = LinearRegression()
lr.fit(X_train.reshape(-1, 1), y_train)
w0, w1 = lr.intercept_, lr.coef_[0]
print(f"Phương trình hồi quy: GPA = {w0:.6f} + {w1:.6f} * SAT")

# Vẽ đồ thị
plt.figure(figsize=(9, 5))
plt.scatter(X_train, y_train, color='blue', label='Train (64 mẫu)')
plt.scatter(X_val, y_val, color='green', marker='^', label='Val (20 mẫu)')
x_line = np.linspace(min(x) - 20, max(x) + 20, 100)
plt.plot(x_line, w0 + w1 * x_line, color='orange', lw=2.5, label='Đường hồi quy')
plt.xlabel('SAT'); plt.ylabel('GPA'); plt.legend(); plt.grid(True)
plt.savefig(r"d:\MachineLearning\Week1\work\hinh_B3_sat_gpa_regression.png", dpi=300)
plt.close()

# (iv) Đánh giá Linear Regression trên Validation
y_pred_lr = lr.predict(X_val.reshape(-1, 1))
mse_lr = mean_squared_error(y_val, y_pred_lr)
mae_lr = mean_absolute_error(y_val, y_pred_lr)
r2_lr = r2_score(y_val, y_pred_lr)

# (v) K-NN (k = 8)
def knn_1d(tr_x, tr_y, te_x, k=8):
    return np.array([np.mean(tr_y[np.argsort(np.abs(tr_x - val))[:k]]) for val in te_x])

y_pred_knn = knn_1d(X_train, y_train, X_val, k=8)
mse_knn = mean_squared_error(y_val, y_pred_knn)
mae_knn = mean_absolute_error(y_val, y_pred_knn)
r2_knn = r2_score(y_val, y_pred_knn)
```

### 5.4. Kết quả & Đáp số chính xác

1. **Hệ số tương quan Pearson**:
   $$r \approx 0.63718$$
   - **Ý nghĩa**: Giá trị $0.637$ thể hiện mối tương quan thuận tuyến tính ở mức độ trung bình-khá giữa điểm thi đầu vào SAT và điểm tốt nghiệp đại học GPA.
2. **Phương trình đường hồi quy**:
   $$\text{GPA} = 0.838210 + 0.001318 \times \text{SAT}$$
   - *Ý nghĩa*: Cứ tăng $100$ điểm SAT thì điểm GPA trung bình kỳ vọng tăng thêm khoảng $0.1318$ điểm.
3. **Bảng so sánh Linear Regression và K-NN ($k=8$)**:

| Tiêu chí | Hồi quy tuyến tính (Linear Regression) | K-Nearest Neighbors ($k=8$) | Nhận xét |
| :--- | :---: | :---: | :--- |
| **MSE** | **0.072638** | **0.081378** | Linear Regression có MSE thấp hơn |
| **MAE** | **0.248804** | **0.271188** | Linear Regression có MAE thấp hơn |
| **$R^2$** | **-9.667975** | **-10.951519** | Linear Regression tốt hơn K-NN |

4. **Biểu đồ minh họa**: Đã được vẽ và lưu tại [hinh_B3_sat_gpa_regression.png](file:///d:/MachineLearning/Week1/work/hinh_B3_sat_gpa_regression.png).

---

## 6. Kết quả thực hiện Bài tập 3 (Bài tập 2 / B.4 - Độ dày NTM)

### 6.1. Đề bài
> Dữ liệu trong tệp `vidu4_lin_reg.txt` (100 bệnh nhân). Các trường dữ liệu gồm: `ID`, `TUOI`, `BMI`, `HA`, `GLUCOSE`, `CHOLESTEROL`, `BEDAYNTM`.
> Mô hình cần xây dựng:
> $$\text{Bề dày NTM} = \beta_0 + \beta_1(\text{tuổi}) + \beta_2(\text{cholesterol}) + \beta_3(\text{glucose}) + \beta_4(\text{huyết áp TT}) + \beta_5(\text{BMI})$$
> a) Xác định các hệ số với 100 dữ liệu trên.
> b) Chia dữ liệu thành: 80 dòng đầu dùng cho training; 20 dòng sau dùng cho testing. Tính lại các hệ số với bộ dữ liệu này, sau đó chạy thử trên bộ dữ liệu test và tính các đại lượng kỳ vọng, phương sai của sai số.

### 6.2. Mã nguồn thực hiện (`bai_tap_B4.py`)

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Đọc file dữ liệu dạng bảng văn bản
df = pd.read_csv(r"d:\MachineLearning\Week1\guide\vidu4_lin_reg.txt", sep=r'\s+')

# Sắp xếp đúng thứ tự biến theo đề bài
features = ['TUOI', 'CHOLESTEROL', 'GLUCOSE', 'HA', 'BMI']
X = df[features].values
y = df['BEDAYNTM'].values

# a) Huấn luyện trên toàn bộ 100 mẫu
model_all = LinearRegression()
model_all.fit(X, y)

print("--- CÂU A: HỆ SỐ TRÊN TOÀN BỘ 100 MẪU ---")
print(f"beta_0 (Intercept) : {model_all.intercept_:.6f}")
for name, coef in zip(features, model_all.coef_):
    print(f"{name:<15}: {coef:.6f}")

# b) Huấn luyện trên 80 mẫu đầu, test trên 20 mẫu sau
X_train, y_train = X[:80], y[:80]
X_test, y_test = X[80:], y[80:]

model_80 = LinearRegression()
model_80.fit(X_train, y_train)

# Dự đoán trên test
y_pred_test = model_80.predict(X_test)
errors = y_test - y_pred_test # Sai số: e = y - y_hat

mean_err = np.mean(errors) # Kỳ vọng sai số
var_err = np.var(errors)   # Phương sai sai số

print("\n--- CÂU B: SAI SỐ TRÊN TẬP TEST (20 MẪU) ---")
print(f"Kỳ vọng sai số E[e]        : {mean_err:.6f}")
print(f"Phương sai sai số Var(e)    : {var_err:.6f}")
```

### 6.3. Kết quả & Đáp số chính xác

#### a) Các hệ số hồi quy với 100 mẫu dữ liệu:
| Tham số | Ý nghĩa | Giá trị ước lượng |
| :--- | :--- | :---: |
| **$\beta_0$** | Hệ số chặn (Intercept) | **-0.108029** |
| **$\beta_1$** | Hệ số Tuổi (`TUOI`) | **0.009947** |
| **$\beta_2$** | Hệ số Cholesterol (`CHOLESTEROL`) | **0.038470** |
| **$\beta_3$** | Hệ số Đường huyết (`GLUCOSE`) | **0.007670** |
| **$\beta_4$** | Hệ số Huyết áp tâm thu (`HA`) | **0.002181** |
| **$\beta_5$** | Hệ số Khối cơ thể (`BMI`) | **-0.000207** |

$$\text{Bề dày NTM} = -0.1080 + 0.009947 \cdot \text{TUOI} + 0.038470 \cdot \text{CHOLESTEROL} + 0.007670 \cdot \text{GLUCOSE} + 0.002181 \cdot \text{HA} - 0.000207 \cdot \text{BMI}$$

#### b) Kết quả huấn luyện 80 mẫu và kiểm thử 20 mẫu:
- **Hệ số hồi quy mới ($\beta_0 \to \beta_5$)**:
  - $\beta_0 = 0.043064$
  - $\beta_1 (\text{Tuổi}) = 0.008989$
  - $\beta_2 (\text{Cholesterol}) = 0.007085$
  - $\beta_3 (\text{Glucose}) = 0.008086$
  - $\beta_4 (\text{Huyết áp TT}) = 0.002602$
  - $\beta_5 (\text{BMI}) = -0.000477$
- **Đại lượng sai số trên 20 mẫu test**:
  - **Kỳ vọng của sai số (Mean Error) $E[e]$**: **$0.079989$** $\approx 0.08$ (Độ lệch dương nhỏ, cho thấy trung bình mô hình ước lượng thấp hơn giá trị thực khoảng 0.08 mm).
  - **Phương sai của sai số (Variance of Error) $\text{Var}(e)$**: **$0.221003$**.
  - **Độ lệch chuẩn sai số $\text{Std}(e)$**: **$0.470109$**.

---

## 7. Kết quả thực hiện Bài tập 4 (Bài tập 3 / B.5 - Giá bất động sản)

### 7.1. Đề bài
> Trong tệp dữ liệu `real_estate.csv` đính kèm chứa 414 mẫu dữ liệu giao dịch bất động sản. Mỗi bản ghi có 8 cột:
> • Cột x1: Số thứ tự (bỏ qua trường này)
> • Cột x2: Ngày giao dịch mua bán (lấy phần nguyên là năm)
> • Cột x3: Tuổi của căn nhà (theo năm)
> • Cột x4: Khoảng cách tới ga MRT gần nhất
> • Cột x5: Số cửa hàng tiện ích gần đó
> • Cột x6: Kinh độ; Cột x7: Vĩ độ
> • Cột Y: Giá của căn nhà
> Chia dữ liệu thành phần training với 350 mẫu đầu, phần validation với các mẫu còn lại (64 mẫu). Xây dựng mô hình hồi quy tuyến tính mô tả sự phụ thuộc của Y vào các cột từ X2 đến X6. Chạy dự đoán cho validation và đưa ra tổng bình phương sai số của dự đoán (RSS). So sánh với phương pháp K-NN, lấy $K = 18$.

### 7.2. Mã nguồn thực hiện (`bai_tap_B5.py`)

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Đọc file CSV
df = pd.read_csv(r"d:\MachineLearning\Week1\guide\real_estate.csv")

# Tiền xử lý: Lấy năm giao dịch dạng nguyên
df['Transaction_Year'] = df['X1 transaction date'].apply(lambda x: int(x))

# Tập đặc trưng từ X2 đến X6 theo tên cột trong file CSV:
features = [
    'X2 house age', 
    'X3 distance to the nearest MRT station', 
    'X4 number of convenience stores', 
    'X5 latitude', 
    'X6 longitude'
]

X = df[features].values
y = df['Y house price of unit area'].values

# Chia tập: 350 mẫu đầu làm Training, 64 mẫu sau làm Validation
X_train, y_train = X[:350], y[:350]
X_val, y_val = X[350:], y[350:]

# 1. Hồi quy tuyến tính
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_val)
rss_lr = np.sum((y_val - y_pred_lr) ** 2)

# 2. K-NN hồi quy đa chiều (K = 18, khoảng cách Euclid chuẩn 2)
def knn_regression(train_X, train_y, val_X, k=18):
    preds = []
    for x in val_X:
        dists = np.linalg.norm(train_X - x, ord=2, axis=1) # axis=1
        idx = np.argsort(dists)[:k]
        preds.append(np.mean(train_y[idx]))
    return np.array(preds)

y_pred_knn = knn_regression(X_train, y_train, X_val, k=18)
rss_knn = np.sum((y_val - y_pred_knn) ** 2)

print(f"Tổng bình phương sai số Hồi quy tuyến tính (RSS_LR) : {rss_lr:.4f}")
print(f"Tổng bình phương sai số K-NN K=18        (RSS_KNN): {rss_knn:.4f}")
```

### 7.3. Kết quả & Đáp số chính xác

#### Bảng so sánh kết quả trên 64 mẫu Validation:

| Phương pháp | Tổng bình phương sai số (RSS) | Sai số MSE | Sai số MAE | Hệ số $R^2$ |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | **4228.1469** | **66.0648** | **6.1255** | **0.5860** |
| **K-NN ($K = 18$)** | **2809.2935** | **43.8952** | **4.7818** | **0.7250** |

*(Ghi chú: Nếu sử dụng đầy đủ cả 6 biến bao gồm năm giao dịch `Transaction_Year`, RSS của Linear Regression là $4083.6448$ và K-NN là $2809.3115$, K-NN vẫn tốt hơn rất nhiều).*

#### So sánh chuyên sâu & Lý giải bản chất:
1. **Hiệu năng**: Mô hình K-NN ($K = 18$) cho kết quả vượt trội rõ rệt:
   - Tổng bình phương sai số giảm từ **$4228.15$ xuống $2809.29$** (giảm hơn $33.5\%$).
   - Hệ số $R^2$ tăng từ $0.586$ lên $0.725$.
2. **Lý giải bản chất bài toán**:
   - Dữ liệu giá nhà đất phụ thuộc cực kỳ lớn vào yếu tố không gian địa lý (kinh độ, vĩ độ, khoảng cách ga tàu điện MRT, số cửa hàng tiện ích).
   - Thị trường bất động sản có đặc tính **phân cụm cục bộ và phi tuyến tính cao** (các căn nhà cùng một khu phố, gần cùng một ga tàu sẽ có mặt bằng giá tương đồng nhau, không tuân theo một mặt phẳng dốc đều trên toàn thành phố).
   - Thuật toán **K-NN** gom 18 căn nhà có đặc trưng gần nhất trong không gian Euclid để lấy trung bình nên mô tả rất tốt các vùng giá cục bộ này.
   - Ngược lại, **Linear Regression** buộc phải khớp một siêu phẳng tuyến tính toàn cục trên toàn bộ bản đồ, do đó không nắm bắt được các "túi giá" hoặc khu vực tăng giá đột biến.
3. **Biểu đồ so sánh**: Đã được lưu tại [hinh_B5_real_estate_comparison.png](file:///d:/MachineLearning/Week1/work/hinh_B5_real_estate_comparison.png).

---

## 8. Tổng kết kết quả thực nghiệm & Trả lời câu hỏi vấn đáp

### 8.1. Hướng dẫn thực thi mã nguồn kiểm thử
```powershell
# Chạy Bài tập 1 (Fuel Linear Regression)
python d:\MachineLearning\Week1\work\bai_tap_B1.py

# Chạy Bài tập 2 (SAT - GPA Hồi quy & K-NN, sinh biểu đồ B3)
python d:\MachineLearning\Week1\work\bai_tap_B3.py

# Chạy Bài tập 3 (Độ dày NTM, tính kỳ vọng và phương sai sai số)
python d:\MachineLearning\Week1\work\bai_tap_B4.py

# Chạy Bài tập 4 (Dự đoán giá BĐS, so sánh RSS giữa LR và K-NN, sinh biểu đồ B5)
python d:\MachineLearning\Week1\work\bai_tap_B5.py

# Chạy kiểm tra toàn bộ 4 bài cùng lúc
python d:\MachineLearning\Week1\work\solve_all.py
```

### 8.2. Phân tích & Trả lời các câu hỏi thảo luận bảo vệ bài

1. **Câu hỏi**: *Tại sao trong bài SAT_GPA (Bài tập B.3), Linear Regression lại tốt hơn K-NN, nhưng trong bài Real Estate (Bài tập B.5) K-NN lại tốt hơn Linear Regression?*
   - **Trả lời**:
     - Trong bài **SAT_GPA**: Dữ liệu chỉ có 1 chiều và mang tính quy luật tuyến tính đồng biến rõ rệt. Đồng thời do dữ liệu bị chia lệch (tập validation toàn điểm cao ngoài dải tập train), Linear Regression có ưu thế **ngoại suy (extrapolation)** dọc theo độ dốc, trong khi K-NN chỉ có thể **nội suy (interpolation)** trong dải giá trị đã học.
     - Trong bài **Real Estate**: Dữ liệu đa chiều và có tính chất vị trí không gian địa lý phi tuyến mạnh. K-NN vượt trội nhờ khả năng mô hình hóa các cụm giá địa phương (local neighborhood).

2. **Câu hỏi**: *Trong hàm tính khoảng cách Euclid đa chiều bằng `numpy.linalg.norm(array - value)`, tại sao phải là `axis=1` mà không phải `axis=0`?*
   - **Trả lời**: `array` có shape $(N, d)$ và `value` có shape $(d,)$. Phép trừ `array - value` cho ma trận hiệu shape $(N, d)$. Ta cần tính khoảng cách giữa điểm `value` tới từng điểm trong $N$ điểm huấn luyện, tức là tính chuẩn Euclid trên từng dòng (vector $d$ chiều), do đó phải gom theo chiều ngang `axis=1` để trả về vector khoảng cách shape $(N,)$.

3. **Câu hỏi**: *Hệ số xác định $R^2$ có bao giờ bị âm không? Vì sao?*
   - **Trả lời**: Có. $R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$. Khi mô hình đưa ra các dự đoán có tổng sai số bình phương lớn hơn việc đơn giản chỉ lấy giá trị trung bình $\bar{y}_{\text{val}}$ của tập kiểm thử để dự đoán, $\text{SS}_{\text{res}} > \text{SS}_{\text{tot}}$ dẫn đến $R^2 < 0$. Hiện tượng này thường xảy ra khi kiểm thử trên tập dữ liệu bị lệch phân phối so với tập huấn luyện.
