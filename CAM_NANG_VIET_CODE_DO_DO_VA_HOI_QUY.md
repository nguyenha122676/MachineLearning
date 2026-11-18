# CẨM NANG THỰC CHIẾN: VIẾT CODE ĐỘ ĐO & HỒI QUY BẰNG THƯ VIỆN (SCIKIT-LEARN)
## TÀI LIỆU ÔN TẬP VÀ MẪU CODE THỰC CHIẾN — HỌC MÁY (MACHINE LEARNING)

> **Dành cho**: Sinh viên ôn tập thi thực hành, thi trắc nghiệm, vấn đáp Machine Learning  
> **Trọng tâm**: Sử dụng thành thạo thư viện **`scikit-learn`**, **`numpy`**, **`pandas`**, **`matplotlib`**  
> **Nguyên tắc**: Code chuẩn mực, tối giản, dễ học thuộc lòng, hiểu sâu bản chất từng hàm, tránh 100% bẫy lỗi thường gặp.

---

## MỤC LỤC
1. [Bản đồ tư duy: Phân biệt bài toán & Nguyên tắc chọn độ đo](#1-bản-đồ-tư-duy-phân-biệt-bài-toán--nguyên-tắc-chọn-độ-đo)
2. [Quy trình chuẩn 5 bước (Pipeline) làm việc với Scikit-learn](#2-quy-trình-chuẩn-5-bước-pipeline-làm-việc-với-scikit-learn)
3. [Chuyên đề 1: Viết code các mô hình Hồi quy (Regression Models)](#3-chuyên-đề-1-viết-code-các-mô-hình-hồi-quy-regression-models)
   - 3.1. Hồi quy tuyến tính (`LinearRegression`)
   - 3.2. Hồi quy đa thức (`PolynomialFeatures` + `LinearRegression`)
   - 3.3. Hồi quy có điều chuẩn Ridge & Lasso (`Ridge`, `Lasso`)
   - 3.4. K-NN cho Hồi quy (`KNeighborsRegressor`)
   - 3.5. Hồi quy Logistic (`LogisticRegression` - Dành cho phân loại)
4. [Chuyên đề 2: Viết code các độ đo Hồi quy (Regression Metrics)](#4-chuyên-đề-2-viết-code-các-độ-đo-hồi-quy-regression-metrics)
   - 4.1. MSE (Mean Squared Error)
   - 4.2. RMSE (Root Mean Squared Error)
   - 4.3. MAE (Mean Absolute Error)
   - 4.4. $R^2$ Score (Hệ số xác định)
   - 4.5. Phân tích phần dư sai số (Residuals) & Biểu đồ Residual Plot
5. [Chuyên đề 3: Viết code các độ đo Phân loại (Classification Metrics)](#5-chuyên-đề-3-viết-code-các-độ-đo-phân-loại-classification-metrics)
   - 5.1. Ma trận nhầm lẫn (`confusion_matrix`, `ConfusionMatrixDisplay`)
   - 5.2. `accuracy_score` & Cạm bẫy mất cân bằng dữ liệu
   - 5.3. `precision_score`, `recall_score`, `f1_score`
   - 5.4. Đa lớp (Multiclass): Tham số `average='macro'|'micro'|'weighted'`
   - 5.5. Báo cáo phân loại tổng hợp (`classification_report`)
   - 5.6. Đường cong ROC & chỉ số AUC (`roc_curve`, `roc_auc_score`)
6. [Chuyên đề 4: Tiền xử lý & Tránh rò rỉ dữ liệu (Data Preprocessing & Leakage)](#6-chuyên-đề-4-tiền-xử-lý--tránh-rò-rỉ-dữ-liệu-data-preprocessing--leakage)
   - 6.1. Phân chia tập dữ liệu (`train_test_split`, tham số `stratify`)
   - 6.2. Chuẩn hóa đặc trưng (`StandardScaler`, `MinMaxScaler`)
   - 6.3. Đóng gói chuyên nghiệp với `Pipeline`
7. [Chuyên đề 5: 8 Bẫy lỗi kinh điển khi dùng thư viện & Cách xử lý trong 5 giây](#7-chuyên-đề-5-8-bẫy-lỗi-kinh-điển-khi-dùng-thư-viện--cách-xử-lý-trong-5-giây)
8. [Chuyên đề 6: Bộ Template "Ăn liền" đi thi (Copy-Paste là chạy)](#8-chuyên-đề-6-bộ-template-ăn-liền-đi-thi-copy-paste-là-chạy)
   - Template A: Toàn bộ quy trình Hồi quy chuẩn (Linear / Ridge / KNN)
   - Template B: Toàn bộ quy trình Phân loại chuẩn (Logistic Regression)
   - Template C: Script kiểm thử tự động với dữ liệu mô phỏng

---

# 1. BẢN ĐỒ TƯ DUY: PHÂN BIỆT BÀI TOÁN & NGUYÊN TẮC CHỌN ĐỘ ĐO

Trước khi viết bất kỳ dòng code nào, bạn phải xác định ngay: **Biến mục tiêu ($y$) thuộc loại gì?**

```
                            BÀI TOÁN MACHINE LEARNING CÓ GIÁM SÁT
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
       HỒI QUY (REGRESSION)                               PHÂN LOẠI (CLASSIFICATION)
┌───────────────────────────────────────┐   ┌───────────────────────────────────────┐
│ • Biến mục tiêu y: SỐ THỰC LIÊN TỤC   │   │ • Biến mục tiêu y: NHÃN RỜI RẠC       │
│   (Ví dụ: Giá nhà, Điểm thi GPA,      │   │   (Ví dụ: 0 hoặc 1, Đỗ/Trượt,         │
│    Độ dày màng NTM, Tiêu thụ xăng)    │   │    Chữ số 0-9, Mắc bệnh tim hay không)│
├───────────────────────────────────────┤   ├───────────────────────────────────────┤
│ • Mô hình sử dụng:                    │   │ • Mô hình sử dụng:                    │
│   - LinearRegression                  │   │   - LogisticRegression                │
│   - Ridge, Lasso                      │   │   - KNeighborsClassifier              │
│   - KNeighborsRegressor               │   │   - GaussianNB, MultinomialNB         │
├───────────────────────────────────────┤   ├───────────────────────────────────────┤
│ • Các độ đo BẮT BUỘC dùng:            │   │ • Các độ đo BẮT BUỘC dùng:            │
│   - MSE (Mean Squared Error)          │   │   - Confusion Matrix (TP, FP, FN, TN) │
│   - RMSE (Root Mean Squared Error)    │   │   - Accuracy                          │
│   - MAE (Mean Absolute Error)         │   │   - Precision, Recall, F1-Score       │
│   - R-squared (R² Score)              │   │   - ROC-AUC Score                     │
└───────────────────────────────────────┘   └───────────────────────────────────────┘
```

> [!WARNING]
> **TỬ HUYỆT ĐI THI**:
> - Tuyệt đối **KHÔNG BAO GIỜ** dùng `accuracy_score` hay `confusion_matrix` cho bài toán Hồi quy (sẽ bị lỗi crash ngay vì $y$ là số thực liên tục)!
> - Tuyệt đối **KHÔNG BAO GIỜ** dùng `mean_squared_error` làm tiêu chí đánh giá phân loại chính cho `LogisticRegression` trong bài thi trắc nghiệm hoặc báo cáo!

---

# 2. QUY TRÌNH CHUẨN 5 BƯỚC (PIPELINE) LÀM VIỆC VỚI SCIKIT-LEARN

Mọi bài toán trong `scikit-learn` đều tuân theo đúng một khuôn mẫu duy nhất:

```python
# BƯỚC 1: Import thư viện cần thiết
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression  # hoặc LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score  # hoặc accuracy_score, f1_score

# BƯỚC 2: Chia dữ liệu Train / Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# BƯỚC 3: Chuẩn hóa đặc trưng (nếu cần, ví dụ KNN hoặc Ridge/Lasso)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # CHỈ dùng transform trên Test!

# BƯỚC 4: Khởi tạo mô hình & Huấn luyện (Fit)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# BƯỚC 5: Dự đoán (Predict) & Đánh giá (Evaluate)
y_pred = model.predict(X_test_scaled)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 :", r2_score(y_test, y_pred))
```

---

# 3. CHUYÊN ĐỀ 1: VIẾT CODE CÁC MÔ HÌNH HỒI QUY (REGRESSION MODELS)

## 3.1. Hồi quy tuyến tính (`LinearRegression`)

Mô hình giả định mối quan hệ tuyến tính: $\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_d x_d$.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Khởi tạo mô hình
# fit_intercept=True: Tự động học hệ số tự do w_0 (bias). Mặc định là True (không cần chỉnh).
model_lr = LinearRegression(fit_intercept=True)

# 2. Huấn luyện mô hình
model_lr.fit(X_train, y_train)

# 3. Bóc tách hệ số hồi quy (Rất hay bị hỏi khi thi lý thuyết / vấn đáp)
w0 = model_lr.intercept_     # Hệ số chặn (Bias / Intercept) -> một số thực scalar
w_vec = model_lr.coef_       # Vector trọng số [w1, w2, ..., wd] -> mảng numpy 1D

print(f"Hệ số tự do w0: {w0:.4f}")
print(f"Vector trọng số w: {w_vec}")

# 4. Dự đoán trên tập kiểm thử
y_pred = model_lr.predict(X_test)

# 5. Đánh giá nhanh bằng hàm có sẵn của model
# Chú ý: Với bài toán Hồi quy, hàm .score() MẶC ĐỊNH trả về R^2 score!
train_r2 = model_lr.score(X_train, y_train)
test_r2 = model_lr.score(X_test, y_test)
print(f"Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
```

---

## 3.2. Hồi quy đa thức (`PolynomialFeatures` + `LinearRegression`)

Khi mối quan hệ giữa $X$ và $y$ là đường cong cong (phi tuyến), ta sinh thêm các lũy thừa bậc cao ($x^2, x^3, x_1 x_2$) rồi mới đưa qua `LinearRegression`.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Cách 1: Thủ công từng bước
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)
y_pred_poly = model_poly.predict(X_test_poly)

# Cách 2: Gói gọn bằng make_pipeline (KHUYÊN DÙNG ĐI THI)
pipeline_poly = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression()
)
pipeline_poly.fit(X_train, y_train)
y_pred_poly = pipeline_poly.predict(X_test)
```

---

## 3.3. Hồi quy có điều chuẩn Ridge & Lasso (`Ridge`, `Lasso`)

Dùng khi mô hình bị Overfitting (học vẹt) hoặc dữ liệu có hiện tượng đa cộng tuyến (các cột biến tương quan mạnh với nhau).

```python
from sklearn.linear_model import Ridge, Lasso

# Ridge Regression: Thêm phạt L2 (tổng bình phương w). Ép w nhỏ lại nhưng KHÔNG bằng 0.
# alpha: Hệ số phạt (alpha càng to thì ép càng mạnh)
model_ridge = Ridge(alpha=1.0)
model_ridge.fit(X_train, y_train)
y_pred_ridge = model_ridge.predict(X_test)

# Lasso Regression: Thêm phạt L1 (tổng trị tuyệt đối w). Có khả năng triệt tiêu w = 0 (chọn lọc đặc trưng).
model_lasso = Lasso(alpha=0.1)
model_lasso.fit(X_train, y_train)
y_pred_lasso = model_lasso.predict(X_test)

print("Số lượng đặc trưng bị Lasso triệt tiêu về 0:", np.sum(model_lasso.coef_ == 0))
```

---

## 3.4. K-NN cho Hồi quy (`KNeighborsRegressor`)

Tìm $K$ điểm láng giềng gần nhất theo khoảng cách Euclid và lấy trung bình cộng giá trị của chúng: $\hat{y} = \frac{1}{K} \sum_{i \in \mathcal{N}_K} y_i$.

> [!IMPORTANT]
> **BẮT BUỘC PHẢI SCALE DỮ LIỆU VỚI K-NN!**  
> Do K-NN dùng khoảng cách Euclid, nếu một biến có thang đo hàng ngàn (giá tiền) và một biến có thang đo từ 1-5 (số phòng ngủ), biến giá tiền sẽ nuốt chửng biến còn lại!

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# 1. Scale dữ liệu
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# 2. Khởi tạo mô hình
# n_neighbors: số láng giềng K
# weights='uniform': trung bình cộng đều
# weights='distance': trung bình có trọng số theo nghịch đảo khoảng cách
knn_reg = KNeighborsRegressor(n_neighbors=5, weights='uniform')

# 3. Huấn luyện và dự đoán
knn_reg.fit(X_train_sc, y_train)
y_pred_knn = knn_reg.predict(X_test_sc)
```

---

## 3.5. Hồi quy Logistic (`LogisticRegression` - Dành cho Phân loại)

**Nghịch lý đi thi**: Tên có chữ "Hồi quy" (Regression) nhưng mục đích thực tế lại là **Phân loại** (Classification).  
Mô hình ước lượng xác suất $P(y=1 \mid x) = \sigma(\mathbf{w}^T \mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^T \mathbf{x}}}$.

```python
from sklearn.linear_model import LogisticRegression

# 1. Khởi tạo mô hình
# max_iter: Số vòng lặp tối đa của bộ tối ưu (khuyên để 1000 để tránh cảnh báo ConvergenceWarning)
log_reg = LogisticRegression(max_iter=1000, random_state=42)

# 2. Huấn luyện
log_reg.fit(X_train, y_train)

# 3. Hai phương thức dự đoán cực kỳ quan trọng:
# a) .predict(X): Trả về nhãn rời rạc 0 hoặc 1 (dùng ngưỡng mặc định 0.5)
y_pred_label = log_reg.predict(X_test)

# b) .predict_proba(X): Trả về ma trận xác suất kích thước (N, 2)
# Cột 0: Xác suất thuộc lớp 0 -> P(y = 0)
# Cột 1: Xác suất thuộc lớp 1 -> P(y = 1) (CỰC KỲ DÙNG NHIỀU ĐỂ TÍNH ROC-AUC)
y_pred_proba = log_reg.predict_proba(X_test)
y_prob_positive = y_pred_proba[:, 1]  # Lấy riêng xác suất lớp 1
```

---

# 4. CHUYÊN ĐỀ 2: VIẾT CODE CÁC ĐỘ ĐO HỒI QUY (REGRESSION METRICS)

Toàn bộ độ đo hồi quy nằm trong module: `sklearn.metrics`.

```python
from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,      # Có từ Scikit-learn phiên bản >= 1.4
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)
```

## Bảng so sánh 4 độ đo Hồi quy cốt lõi

| Độ đo | Công thức | Cú pháp Scikit-Learn | Ý nghĩa & Ứng dụng | Giá trị tối ưu |
| :--- | :--- | :--- | :--- | :--- |
| **MSE** | $\frac{1}{N}\sum (y - \hat{y})^2$ | `mean_squared_error(y_test, y_pred)` | Phạt cực nặng sai số lớn do có bình phương | Càng gần $0$ càng tốt |
| **RMSE** | $\sqrt{\text{MSE}}$ | `root_mean_squared_error(y_test, y_pred)` *(hoặc `np.sqrt(mean_squared_error(y_test, y_pred))`)* | Đưa sai số về cùng đơn vị đo thực tế của $y$ | Càng gần $0$ càng tốt |
| **MAE** | $\frac{1}{N}\sum \|y - \hat{y}\|$ | `mean_absolute_error(y_test, y_pred)` | Trực quan, ít bị ảnh hưởng bởi điểm ngoại lai (Outliers) | Càng gần $0$ càng tốt |
| **$R^2$ Score** | $1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}$ | `r2_score(y_test, y_pred)` | Đo tỷ lệ phương sai của $y$ được mô hình giải thích | Càng gần $1.0$ càng tốt |

---

## 4.1. Chi tiết cách viết code đánh giá trọn gói Hồi quy

```python
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Giả sử đã có y_test (thực tế) và y_pred (dự đoán từ model)

# 1. Tính MSE
mse = mean_squared_error(y_test, y_pred)

# 2. Tính RMSE (Viết code tương thích mọi phiên bản Scikit-learn)
try:
    from sklearn.metrics import root_mean_squared_error
    rmse = root_mean_squared_error(y_test, y_pred)
except ImportError:
    # Nếu máy trường cài thư viện bản cũ, dùng np.sqrt(mse)
    rmse = np.sqrt(mse)

# 3. Tính MAE
mae = mean_absolute_error(y_test, y_pred)

# 4. Tính R^2
r2 = r2_score(y_test, y_pred)

# In kết quả định dạng chuyên nghiệp
print("="*40)
print("     KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH HỒI QUY")
print("="*40)
print(f"1. MSE  (Mean Squared Error)     : {mse:.4f}")
print(f"2. RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"3. MAE  (Mean Absolute Error)    : {mae:.4f}")
print(f"4. R²   (Coefficient of Deter.)  : {r2:.4f}")
print("="*40)
```

> [!NOTE]
> **HIỂU SÂU VỀ CHỈ SỐ $R^2$**:
> - $R^2 = 1.0$: Dự đoán chuẩn xác $100\%$.
> - $R^2 = 0.8$: Mô hình giải thích được $80\%$ sự biến thiên của dữ liệu.
> - $R^2 = 0$: Mô hình chỉ ngang ngửa việc đoán giá trị trung bình $\bar{y}$ cho tất cả các mẫu.
> - $R^2 < 0$: Mô hình cực kỳ tệ hại, tệ hơn cả việc lấy trung bình cộng để đoán!

---

## 4.2. Phân tích phần dư sai số (Residuals Analysis)

Phần dư là chênh lệch: $e_i = y_i - \hat{y}_i$.  
Mô hình tuyến tính tốt khi phần dư có kỳ vọng xấp xỉ 0 và phân tán ngẫu nhiên (không có quy luật hình phễu hay hình cánh quạt).

```python
import matplotlib.pyplot as plt

# 1. Tính vector phần dư
residuals = y_test - y_pred

# 2. Tính kỳ vọng và phương sai của sai số
mean_residual = np.mean(residuals)
var_residual = np.var(residuals)
print(f"Kỳ vọng sai số E[e]: {mean_residual:.6f} (Lý tưởng ~ 0)")
print(f"Phương sai sai số Var(e): {var_residual:.4f}")

# 3. Vẽ biểu đồ Residual Plot
plt.figure(figsize=(7, 4))
plt.scatter(y_pred, residuals, alpha=0.6, color='blue', edgecolors='k')
plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
plt.title("Biểu đồ phần dư (Residual Plot)")
plt.xlabel("Giá trị dự đoán (Predicted)")
plt.ylabel("Phần dư (Residuals = Actual - Pred)")
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
```

---

# 5. CHUYÊN ĐỀ 3: VIẾT CODE CÁC ĐỘ ĐO PHÂN LOẠI (CLASSIFICATION METRICS)

Dành riêng cho bài toán phân loại nhị phân (Logistic Regression) hoặc đa lớp (Softmax, Naive Bayes, KNN Classifier).

```python
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_curve,
    roc_auc_score
)
```

---

## 5.1. Ma trận nhầm lẫn (`confusion_matrix`)

```
                           BẢN ĐỒ MA TRẬN NHẦM LẪN (Nhị phân 0 và 1)
                                         DỰ ĐOÁN (Predicted)
                                       Lớp 0            Lớp 1
                       THỰC TẾ ┌───────────────────┬───────────────────┐
                      Lớp 0    │   TN (True Neg)   │   FP (False Pos)  │
                               ├───────────────────┼───────────────────┤
                      Lớp 1    │   FN (False Neg)  │   TP (True Pos)   │
                               └───────────────────┴───────────────────┘
```

### Code lấy 4 chỉ số TN, FP, FN, TP:
```python
# 1. Tạo ma trận nhầm lẫn dạng mảng numpy 2x2
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# 2. Bóc tách trực tiếp ra 4 biến bằng .ravel()
tn, fp, fn, tp = cm.ravel()
print(f"TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")

# 3. Trực quan hóa đồ họa đẹp mắt chỉ với 3 dòng code
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Lớp 0', 'Lớp 1'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Ma trận nhầm lẫn")
plt.show()
```

---

## 5.2. Accuracy, Precision, Recall và F1-Score

```python
# 1. Accuracy (Độ chính xác tổng thể): (TP + TN) / Total
acc = accuracy_score(y_test, y_pred)

# 2. Precision (Độ chuẩn xác): TP / (TP + FP)
# Ý nghĩa: Máy báo Dương tính thì bao nhiêu % là chuẩn thật? (Cần cao cho bài toán lọc Spam)
prec = precision_score(y_test, y_pred)

# 3. Recall (Độ nhạy): TP / (TP + FN)
# Ý nghĩa: Ngoài đời có bao nhiêu ca bệnh thì máy bắt được bấy nhiêu? (Cần cao cho bài toán Y tế, Ung thư)
rec = recall_score(y_test, y_pred)

# 4. F1-Score: Trung bình điều hòa giữa Precision và Recall
f1 = f1_score(y_test, y_pred)

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-Score : {f1:.4f}")
```

---

## 5.3. Xử lý bài toán Đa lớp (Multiclass): Tham số `average`

Khi biến $y$ có từ 3 lớp trở lên (ví dụ nhận dạng chữ số MNIST từ 0 đến 9), nếu gọi `f1_score(y_test, y_pred)` trực tiếp sẽ bị lỗi:  
`ValueError: Target is multiclass but average='binary'`

**Cách xử lý**: Luôn truyền tham số `average`:
- `average='macro'`: Tính F1 riêng cho từng lớp rồi lấy trung bình cộng giản đơn (coi mọi lớp quan trọng như nhau).
- `average='micro'`: Tính tổng TP, FP, FN toàn cầu rồi mới tính F1.
- `average='weighted'`: Tính F1 từng lớp rồi nhân với trọng số tỷ lệ mẫu của lớp đó (khuyên dùng khi các lớp không đều nhau).

```python
# Code chuẩn khi phân loại đa lớp (Multiclass)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')
```

---

## 5.4. Báo cáo phân loại chi tiết (`classification_report`)

Chỉ với 1 dòng lệnh, thư viện tính sẵn toàn bộ Precision, Recall, F1 và Support cho từng lớp:

```python
report = classification_report(y_test, y_pred, digits=4)
print("BÁO CÁO PHÂN LOẠI CHI TIẾT:")
print(report)
```

---

## 5.5. Đường cong ROC và diện tích dưới đường cong AUC (`roc_curve`, `roc_auc_score`)

> [!CAUTION]
> **CẠM BẪY SỐNG CÒN KHI TÍNH ROC-AUC**:  
> Hàm `roc_auc_score` và `roc_curve` **BẮT BUỘC PHẢI NHẬN XÁC SUẤT** `y_prob` (lấy từ `.predict_proba(X)[:, 1]`), **TUYỆT ĐỐI KHÔNG TRUYỀN NHÃN 0/1** `y_pred`!  
> Nếu truyền nhãn 0/1, đường cong ROC sẽ bị biến dạng thành đường gấp khúc cụt lủn và chỉ số AUC bị sai lệch!

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# 1. Lấy xác suất dự đoán của lớp 1
y_prob_positive = log_reg.predict_proba(X_test)[:, 1]

# 2. Tính chỉ số AUC
auc = roc_auc_score(y_test, y_prob_positive)
print(f"Chỉ số AUC: {auc:.4f}")

# 3. Tính tọa độ FPR và TPR để vẽ đường cong ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob_positive)

# 4. Vẽ đường cong ROC
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Đường ROC (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Đoán bừa (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tỷ lệ dương tính giả (FPR)')
plt.ylabel('Tỷ lệ dương tính thật (TPR / Recall)')
plt.title('Đặc tuyến hoạt động của bộ thu (ROC Curve)')
plt.legend(loc="lower right")
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()
```

---

# 6. CHUYÊN ĐỀ 4: TIỀN XỬ LÝ & TRÁNH RÒ RỈ DỮ LIỆU (PREPROCESSING & LEAKAGE)

## 6.1. Phân chia Train/Test (`train_test_split`)

```python
from sklearn.model_selection import train_test_split

# Cho bài toán Hồi quy (Linear Regression):
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Cho bài toán Phân loại (Logistic Regression):
# BẮT BUỘC THÊM stratify=y để đảm bảo tỷ lệ lớp 0 và lớp 1 ở tập train và test bằng nhau!
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

---

## 6.2. Chuẩn hóa đặc trưng: Quy tắc vàng chống rò rỉ dữ liệu (Data Leakage)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# ĐÚNG:
X_train_scaled = scaler.fit_transform(X_train)  # Học mu, sigma từ Train và biến đổi Train
X_test_scaled = scaler.transform(X_test)        # CHỈ dùng mu, sigma cũ để biến đổi Test!

# TUYỆT ĐỐI KHÔNG ĐƯỢC VIẾT THẾ NÀY (BỊ TRỪ ĐIỂM NGAY VÌ DATA LEAKAGE):
# X_test_scaled = scaler.fit_transform(X_test)  <-- SAI NGHIÊM TRỌNG!
# scaler.fit(X) trước khi train_test_split      <-- SAI NGHIÊM TRỌNG!
```

---

## 6.3. Đóng gói chuyên nghiệp với `Pipeline`

Khi dùng `Pipeline`, Scikit-learn sẽ tự động quản lý việc scale dữ liệu, không bao giờ lo rò rỉ dữ liệu hay quên scale tập test:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=1.0))
])

# Khi gọi fit, Pipeline tự fit_transform trên scaler rồi chuyển dữ liệu vào Ridge
pipe.fit(X_train, y_train)

# Khi gọi predict, Pipeline tự transform trên X_test rồi mới đưa vào Ridge dự đoán
y_pred = pipe.predict(X_test)
```

---

# 7. CHUYÊN ĐỀ 5: 8 BẪY LỖI KINH ĐIỂN KHI DÙNG THƯ VIỆN & CÁCH SỬA TRONG 5 GIÂY

### Bẫy 1: Lỗi `ValueError: Expected 2D array, got 1D array instead`
- **Nguyên nhân**: Khi tập đặc trưng $X$ chỉ có duy nhất 1 cột (Hồi quy đơn biến), Pandas Series hoặc NumPy array chỉ có shape 1 chiều `(N,)`. Trong khi Scikit-learn yêu cầu $X$ bắt buộc phải là ma trận 2 chiều `(N, 1)`.
- **Cách sửa trong 5 giây**:
  ```python
  # Sửa cách 1: Dùng 2 dấu ngoặc vuông khi lọc DataFrame
  X = df[['SAT']]  # Thay vì df['SAT']
  
  # Sửa cách 2: Reshape lại mảng NumPy
  X = X.reshape(-1, 1)
  ```

### Bẫy 2: Nhầm thứ tự đối số trong các hàm độ đo
- **Nguyên nhân**: Viết ngược `mean_squared_error(y_pred, y_test)`.
- **Hậu quả**: Với MSE hay MAE thì không đổi vì tính chất đối xứng, nhưng với các độ đo khác (như ma trận nhầm lẫn hay $R^2$), viết ngược sẽ cho kết quả **SAI HOÀN TOÀN**!
- **Quy tắc vàng nhớ đời**: Luôn là **`metric(y_true, y_pred)`** (Thực tế đi trước, Dự đoán theo sau).

### Bẫy 3: Đưa nhãn 0/1 vào hàm `roc_auc_score`
- **Nguyên nhân**: Gọi `roc_auc_score(y_test, y_pred)`.
- **Cách sửa**: Phải lấy xác suất lớp 1: `y_prob = model.predict_proba(X_test)[:, 1]`, sau đó gọi `roc_auc_score(y_test, y_prob)`.

### Bẫy 4: Quên `random_state` khi chia tập dữ liệu hoặc khởi tạo mô hình
- **Hậu quả**: Mỗi lần bấm chạy Run Notebook lại ra một con số MSE/Accuracy khác nhau, giảng viên chấm bài thấy không khớp kết quả mẫu.
- **Cách sửa**: Luôn cố định `random_state=42` ở `train_test_split` và trong `LogisticRegression(random_state=42)`.

### Bẫy 5: Dữ liệu chứa giá trị khuyết thiếu (`NaN`) khiến hàm `.fit()` báo lỗi
- **Báo lỗi**: `ValueError: Input contains NaN, infinity or a value too large for dtype('float64')`.
- **Cách sửa nhanh trước khi split**:
  ```python
  # Xóa các dòng có NaN:
  df = df.dropna()
  
  # Hoặc điền giá trị trung bình vào cột số:
  df = df.fillna(df.mean(numeric_only=True))
  ```

### Bẫy 6: Lỗi `ConvergenceWarning` khi chạy `LogisticRegression`
- **Nguyên nhân**: Thuật toán tối ưu chưa kịp hội tụ đã hết số vòng lặp tối đa mặc định (100 vòng).
- **Cách sửa**: Tăng `max_iter=1000` hoặc scale dữ liệu bằng `StandardScaler()`.

### Bẫy 7: Gọi nhầm phương thức `.predict()` cho bài toán K-NN
- **Bẫy**: Dùng `KNeighborsClassifier` cho bài toán dự đoán giá nhà $\to$ Mô hình coi mỗi mức giá nhà là 1 lớp riêng biệt, ra kết quả vô nghĩa!
- **Cách sửa**: Hồi quy thì dùng `KNeighborsRegressor`, phân loại mới dùng `KNeighborsClassifier`.

### Bẫy 8: Đánh giá mô hình chỉ dựa vào tập Huấn luyện (Train)
- **Hậu quả**: Mô hình bị Overfitting (học vẹt) nhưng sinh viên tưởng mô hình tốt vì thấy Train MSE = 0.
- **Quy tắc**: Luôn luôn báo cáo kết quả trên tập **Test** độc lập!

---

# 8. CHUYÊN ĐỀ 6: BỘ TEMPLATE "ĂN LIỀN" ĐI THI (COPY-PASTE LÀ CHẠY)

## Template A: Quy trình Hồi quy Tuyến tính chuẩn (Linear Regression)

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Đọc dữ liệu
# df = pd.read_csv("ten_file.csv")
# X = df[['feature1', 'feature2']]
# y = df['target']

# Giả lập dữ liệu mẫu:
np.random.seed(42)
X = np.random.rand(100, 3)
y = 2.5 * X[:, 0] + 1.2 * X[:, 1] - 3.0 * X[:, 2] + 4.0 + np.random.randn(100) * 0.1

# 2. Chia Train / Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Fit mô hình
reg = LinearRegression()
reg.fit(X_train, y_train)

# 4. Dự đoán
y_pred = reg.predict(X_test)

# 5. Đánh giá toàn bộ độ đo
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Hệ số tự do w0 : {reg.intercept_:.4f}")
print(f"Vector trọng số: {reg.coef_}")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R²  : {r2:.4f}")
```

---

## Template B: Quy trình Hồi quy Logistic chuẩn (Phân loại nhị phân)

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# 1. Giả lập dữ liệu phân loại nhị phân mẫu
np.random.seed(42)
X = np.random.randn(200, 4)
y = (X[:, 0] + 2 * X[:, 1] - X[:, 2] > 0).astype(int)

# 2. Chia Train / Test (Có stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Chuẩn hóa đặc trưng
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# 4. Huấn luyện Logistic Regression
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_sc, y_train)

# 5. Dự đoán nhãn và xác suất
y_pred = clf.predict(X_test_sc)
y_prob = clf.predict_proba(X_test_sc)[:, 1]  # Xác suất lớp 1

# 6. Đánh giá toàn bộ độ đo
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("Ma trận nhầm lẫn:\n", cm)
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")
print("\nBáo cáo phân loại:\n", classification_report(y_test, y_pred, digits=4))
```

---

## Template C: Script kiểm thử tự động một-chạm (`test_all_metrics.py`)

Bạn có thể lưu đoạn mã sau thành file `.py` và chạy bằng lệnh `python test_all_metrics.py` để tự động kiểm tra xem toàn bộ các thư viện độ đo và hồi quy trên máy đã sẵn sàng hoạt động chính xác 100% hay chưa:

```python
"""
Script kiểm tra toàn bộ thư viện Hồi quy & Độ đo Machine Learning
Chạy: python test_all_metrics.py
"""
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

def test_regression():
    print("--- 1. KIỂM THỬ HỒI QUY TUYẾN TÍNH & ĐỘ ĐO HỒI QUY ---")
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2.1, 3.9, 6.1, 8.0, 9.9])
    
    model = LinearRegression().fit(X, y)
    y_p = model.predict(X)
    
    print(f"Nghiệm: y = {model.coef_[0]:.2f} * x + {model.intercept_:.2f}")
    print(f"MSE : {mean_squared_error(y, y_p):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y, y_p)):.4f}")
    print(f"MAE : {mean_absolute_error(y, y_p):.4f}")
    print(f"R²  : {r2_score(y, y_p):.4f}")
    assert r2_score(y, y_p) > 0.99, "Lỗi: R² phải xấp xỉ 1.0"
    print("=> HỒI QUY: PASS [OK]\n")

def test_classification():
    print("--- 2. KIỂM THỬ HỒI QUY LOGISTIC & ĐỘ ĐO PHÂN LOẠI ---")
    X = np.array([[-2], [-1], [-0.5], [0.5], [1], [2]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    clf = LogisticRegression().fit(X, y)
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1]
    
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
    print(f"Accuracy : {accuracy_score(y, y_pred):.4f}")
    print(f"Precision: {precision_score(y, y_pred):.4f}")
    print(f"Recall   : {recall_score(y, y_pred):.4f}")
    print(f"F1-Score : {f1_score(y, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y, y_prob):.4f}")
    assert accuracy_score(y, y_pred) == 1.0, "Lỗi: Accuracy phải bằng 1.0"
    print("=> PHÂN LOẠI: PASS [OK]\n")

if __name__ == "__main__":
    test_regression()
    test_classification()
    print("CHÚC MỪNG! TẤT CẢ CÁC ĐỘ ĐO VÀ MÔ HÌNH HOẠT ĐỘNG HOÀN HẢO!")
```
