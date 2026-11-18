# Sinh viên thực hiện: Nguyễn Hải Anh - MSSV: 24001968
"""
BÀI TẬP 2 - VÍ DỤ B.4 (Trang 10 PDF)
Đề bài:
Dữ liệu trong tệp vidu4_lin_reg.txt (100 bệnh nhân).
Các trường dữ liệu: ID, TUOI, BMI, HA, GLUCOSE, CHOLESTEROL, BEDAYNTM.
Mô hình hồi quy cần xây dựng:
  Bề dày NTM = beta0 + beta1*(tuổi) + beta2*(cholesterol) + beta3*(glucose) + beta4*(huyết áp TT) + beta5*(BMI)

Yêu cầu:
a) Xác định các hệ số với toàn bộ 100 dữ liệu trên.
b) Chia dữ liệu thành: 80 dòng đầu dùng cho training; 20 dòng sau dùng cho testing.
   Tính lại các hệ số với bộ dữ liệu này, sau đó chạy thử trên bộ dữ liệu test và tính các đại lượng:
   - Kỳ vọng của sai số (Mean error)
   - Phương sai của sai số (Variance of error).
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
from sklearn.linear_model import LinearRegression

# 1. Đọc dữ liệu
data_path = find_data_file("vidu4_lin_reg.txt")
# Tệp văn bản phân cách bằng khoảng trắng/tab
df = pd.read_csv(data_path, sep=r'\s+')

print("=== BÀI TẬP 2: HỒI QUY BỀ DÀY LỚP NỘI TRUNG MẠC (NTM) ===")
print("5 dòng đầu tiên của dữ liệu:")
print(df.head())

# Thứ tự các biến độc lập theo mô hình đề bài:
# beta0 (hệ số chặn)
# beta1: TUOI
# beta2: CHOLESTEROL
# beta3: GLUCOSE
# beta4: HA (Huyết áp tâm thu)
# beta5: BMI (Chỉ số khối cơ thể)
feature_names = ['TUOI', 'CHOLESTEROL', 'GLUCOSE', 'HA', 'BMI']
beta_labels = ['beta1 (Tuổi)', 'beta2 (Cholesterol)', 'beta3 (Glucose)', 'beta4 (Huyết áp TT)', 'beta5 (BMI)']

X_all = df[feature_names].values
y_all = df['BEDAYNTM'].values

# =========================================================================
# a) Xác định các hệ số với toàn bộ 100 dữ liệu
# =========================================================================
print("\n" + "="*60)
print("a) XÁC ĐỊNH HỆ SỐ HỒI QUY VỚI TOÀN BỘ 100 DỮ LIỆU")
print("="*60)

# Cách 1: Dùng Scikit-Learn
model_all = LinearRegression()
model_all.fit(X_all, y_all)

print(f"Hệ số chặn beta0 (Intercept): {model_all.intercept_:.6f}")
for label, coef in zip(beta_labels, model_all.coef_):
    print(f"{label:<25}: {coef:.6f}")

print("\nPhương trình hồi quy cho toàn bộ 100 bệnh nhân:")
print(f"Bề dày NTM = {model_all.intercept_:.4f} + " + 
      " + ".join([f"{coef:.6f} * {name}" for name, coef in zip(feature_names, model_all.coef_)]))

# Cách 2: Tự giải bằng phương trình chuẩn (Normal Equation) / Ma trận giả nghịch đảo của Numpy:
# w = (X_bar.T @ X_bar)^(-1) @ (X_bar.T @ y)
X_bar_all = np.column_stack([np.ones(len(X_all)), X_all])
w_numpy_all = np.linalg.pinv(X_bar_all.T @ X_bar_all) @ (X_bar_all.T @ y_all)
print("\nKiểm chứng bằng phương pháp Ma trận giả nghịch đảo (Numpy):")
print(f"w0 (beta0) = {w_numpy_all[0]:.6f}")
for label, w_val in zip(beta_labels, w_numpy_all[1:]):
    print(f"{label:<25}: {w_val:.6f}")


# =========================================================================
# b) Chia 80 dòng đầu làm Training, 20 dòng sau làm Testing
# =========================================================================
print("\n" + "="*60)
print("b) HUẤN LUYỆN TRÊN 80 MẪU ĐẦU VÀ KIỂM THỬ TRÊN 20 MẪU SAU")
print("="*60)

X_train, y_train = X_all[:80], y_all[:80]
X_test, y_test = X_all[80:], y_all[80:]

model_split = LinearRegression()
model_split.fit(X_train, y_train)

print(f"Hệ số hồi quy mới huấn luyện từ 80 mẫu:")
print(f"Hệ số chặn beta0 (Intercept): {model_split.intercept_:.6f}")
for label, coef in zip(beta_labels, model_split.coef_):
    print(f"{label:<25}: {coef:.6f}")

# Dự đoán trên tập kiểm thử 20 mẫu
y_test_pred = model_split.predict(X_test)

# Sai số: e_i = y_true - y_pred (hoặc ngược lại)
errors = y_test - y_test_pred

mean_error = np.mean(errors)
var_error = np.var(errors)  # Phương sai
std_error = np.std(errors)  # Độ lệch chuẩn

print("\n--- KẾT QUẢ ĐO LƯỜNG SAI SỐ TRÊN TẬP TESTING (20 MẪU) ---")
print(f"Kỳ vọng của sai số (Mean error)       E[e]   = {mean_error:.6f}")
print(f"Phương sai của sai số (Variance error) Var(e) = {var_error:.6f}")
print(f"Độ lệch chuẩn sai số (Std error)      Std(e) = {std_error:.6f}")

print("\nBảng chi tiết giá trị thực, dự đoán và sai số từng bệnh nhân tập test:")
print(f"{'STT':<6}{'Thực tế (y)':<16}{'Dự đoán (y_hat)':<18}{'Sai số (e = y - y_hat)':<20}")
print("-" * 60)
for i in range(len(y_test)):
    print(f"{i+1:<6}{y_test[i]:<16.4f}{y_test_pred[i]:<18.4f}{errors[i]:<20.4f}")
