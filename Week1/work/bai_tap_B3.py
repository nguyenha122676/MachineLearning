# Sinh viên thực hiện: Nguyễn Hải Anh - MSSV: 24001968
"""
BÀI TẬP TỰ THỰC HÀNH 1 - VÍ DỤ B.3 (Trang 9 PDF)
Đề bài:
Sử dụng lại tệp dữ liệu SAT_GPA.csv. Xây dựng mô hình Hồi quy tuyến tính:
(i) Sử dụng hệ số tương quan Pearson để đánh giá mức độ phụ thuộc của GPA vào SAT.
(ii) Chia dữ liệu thành phần training (64 mẫu đầu) và validation (các mẫu còn lại - 20 mẫu).
(iii) Lập công thức hồi quy tuyến tính ứng với dữ liệu training. Vẽ đồ thị đường hồi quy và điểm dữ liệu.
(iv) Chạy thử mô hình với dữ liệu validation và tính MSE, MAE, R-square.
(v) So sánh kết quả này với K-NN (k = 8) trong Ví dụ A.1.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os

# Tự động tìm đường dẫn file dữ liệu linh hoạt trên mọi máy tính
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def find_data_file(filename):
    for p in [
        os.path.join(BASE_DIR, filename),
        os.path.join(BASE_DIR, "guide", filename),
        os.path.join(BASE_DIR, "..", "guide", filename),
        os.path.join(BASE_DIR, "Week1", "guide", filename),
        os.path.join(r"d:\MachineLearning\Week1\guide", filename)
    ]:
        if os.path.exists(p):
            return os.path.abspath(p)
    return filename

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Đọc dữ liệu
data_path = find_data_file("SAT_GPA.csv")
data = pd.read_csv(data_path)

x = data['SAT'].values
y = data['GPA'].values

print("=== BÀI TẬP TỰ THỰC HÀNH 1 (SAT_GPA) ===")
print(f"Tổng số mẫu: {len(data)}")

# (i) Tính hệ số tương quan Pearson
# Công thức: r = sum((x - x_bar)*(y - y_bar)) / (sqrt(sum(x - x_bar)^2) * sqrt(sum(y - y_bar)^2))
pearson_r = np.corrcoef(x, y)[0, 1]
print(f"\n(i) Hệ số tương quan Pearson giữa SAT và GPA:")
print(f"    - Toàn bộ dữ liệu: r = {pearson_r:.5f}")
print("    => Nhận xét: r ≈ 0.637 mang giá trị dương ở mức khá cao, cho thấy điểm SAT và GPA có tương quan đồng biến (học sinh có SAT cao thường có GPA cao).")

# (ii) Chia dữ liệu: 64 mẫu đầu (train), 20 mẫu còn lại (validation)
X_train = x[:64]
y_train = y[:64]
X_val = x[64:]
y_val = y[64:]

print(f"\n(ii) Kích thước tập dữ liệu sau khi chia:")
print(f"    - Tập Training: {len(X_train)} mẫu (GPA từ {y_train.min():.2f} đến {y_train.max():.2f})")
print(f"    - Tập Validation: {len(X_val)} mẫu (GPA từ {y_val.min():.2f} đến {y_val.max():.2f})")

# (iii) Xây dựng mô hình Hồi quy tuyến tính trên tập Training
# Ta có thể tính trực tiếp theo công thức giải tích:
# theta1 = SXY / SXX
# theta0 = y_bar - theta1 * x_bar
x_bar = np.mean(X_train)
y_bar = np.mean(y_train)
sxx = np.sum((X_train - x_bar) ** 2)
sxy = np.sum((X_train - x_bar) * (y_train - y_bar))
theta_1_formula = sxy / sxx
theta_0_formula = y_bar - theta_1_formula * x_bar

# Sử dụng scikit-learn để kiểm chứng
lr = LinearRegression()
lr.fit(X_train.reshape(-1, 1), y_train)

w0 = lr.intercept_
w1 = lr.coef_[0]

print(f"\n(iii) Phương trình hồi quy tuyến tính (tính từ 64 mẫu train):")
print(f"    Công thức: GPA = theta_0 + theta_1 * SAT")
print(f"    - Tính theo công thức trực tiếp: theta_0 = {theta_0_formula:.6f}, theta_1 = {theta_1_formula:.6f}")
print(f"    - Tính bằng Scikit-Learn:       w0 = {w0:.6f}, w1 = {w1:.6f}")
print(f"    => Phương trình: GPA = {w0:.6f} + {w1:.6f} * SAT")

# Vẽ đồ thị đường hồi quy và các điểm dữ liệu
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Train Data (64 mẫu)')
plt.scatter(X_val, y_val, color='green', marker='^', label='Validation Data (20 mẫu)')

# Đường hồi quy
x_line = np.linspace(min(x) - 20, max(x) + 20, 100)
y_line = w0 + w1 * x_line
plt.plot(x_line, y_line, color='orange', linewidth=2.5, label='Regression Line (y = w0 + w1*x)')

plt.xlabel('Điểm SAT', fontsize=14)
plt.ylabel('Điểm GPA', fontsize=14)
plt.title('Hồi quy tuyến tính dự đoán GPA theo SAT (Bài tập B.3)', fontsize=15)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plot_path = os.path.join(BASE_DIR, "hinh_B3_sat_gpa_regression.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"    Đã vẽ và lưu đồ thị vào: {plot_path}")

# (iv) Đánh giá mô hình trên tập validation
y_val_pred_lr = lr.predict(X_val.reshape(-1, 1))
mse_lr = mean_squared_error(y_val, y_val_pred_lr)
mae_lr = mean_absolute_error(y_val, y_val_pred_lr)
r2_lr = r2_score(y_val, y_val_pred_lr)

print(f"\n(iv) Kết quả đánh giá mô hình Hồi quy tuyến tính trên tập Validation:")
print(f"    - MSE (Mean Squared Error) : {mse_lr:.6f}")
print(f"    - MAE (Mean Absolute Error): {mae_lr:.6f}")
print(f"    - R-squared (R2 Score)     : {r2_lr:.6f}")

# (v) Chạy thuật toán K-NN (k = 8) như trong Ví dụ A.1 để so sánh
k = 8
def knn_predict_1d(train_x, train_y, test_x, k):
    preds = []
    for val in test_x:
        dists = np.abs(train_x - val)
        nearest_idx = np.argsort(dists)[:k]
        preds.append(np.mean(train_y[nearest_idx]))
    return np.array(preds)

y_val_pred_knn = knn_predict_1d(X_train, y_train, X_val, k)
mse_knn = mean_squared_error(y_val, y_val_pred_knn)
mae_knn = mean_absolute_error(y_val, y_val_pred_knn)
r2_knn = r2_score(y_val, y_val_pred_knn)

print(f"\n(v) Bảng so sánh Hồi quy tuyến tính và K-NN (k=8) trên tập Validation:")
print(f"{'Phương pháp':<25}{'MSE':<15}{'MAE':<15}{'R2':<15}")
print("-" * 70)
print(f"{'Linear Regression':<25}{mse_lr:<15.6f}{mae_lr:<15.6f}{r2_lr:<15.6f}")
print(f"{'K-NN (K = 8)':<25}{mse_knn:<15.6f}{mae_knn:<15.6f}{r2_knn:<15.6f}")

print("\nNhận xét so sánh:")
print("1. Cả MSE và MAE của Linear Regression (0.0726 và 0.2488) đều thấp hơn so với K-NN (0.0814 và 0.2712).")
print("   => Mô hình Hồi quy tuyến tính cho kết quả dự đoán chính xác hơn K-NN trên tập dữ liệu này.")
print("2. Giải thích: Dữ liệu SAT_GPA trong file CSV đã bị sắp xếp tăng dần theo GPA.")
print("   - Tập Training (64 mẫu đầu) chứa toàn bộ các điểm GPA từ thấp đến trung bình (2.40 - 3.50).")
print("   - Tập Validation (20 mẫu cuối) lại là nhóm sinh viên đạt GPA cao nhất (3.51 - 3.81).")
print("   - Vì K-NN chỉ nội suy/lấy trung bình từ các điểm có sẵn trong tập train (tối đa chỉ là 3.50), K-NN không thể dự đoán vượt ngưỡng 3.50.")
print("   - Ngược lại, Hồi quy tuyến tính có khả năng ngoại suy (extrapolation) theo độ dốc w1, do đó dự đoán tốt hơn các giá trị ngoài phạm vi tập huấn luyện.")
