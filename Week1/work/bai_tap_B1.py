# Sinh viên thực hiện: Nguyễn Hải Anh - MSSV: 24001968
"""
BÀI TẬP B.1 (Trang 9 PDF)
Đề bài:
- Sử dụng 40 mẫu trong tập dữ liệu của Ví dụ B.2 (fuel.txt) làm tập training để huấn luyện mô hình.
- Dự đoán đầu ra với các mẫu còn lại (11 mẫu - validation).
- Đánh giá độ chính xác của mô hình trên cả 2 tập (training và validation) bằng các độ đo: MSE, MAE, R-square.
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

import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Đọc và tiền xử lý dữ liệu từ file fuel.txt
data_path = find_data_file("fuel.txt")

with open(data_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Bỏ dòng tiêu đề
lines.pop(0)

x_data = []
y_data = []

for line in lines:
    splitted = line.replace('\n', '').replace('"', '').split(',')
    drivers = float(splitted[1])
    fuelc = float(splitted[2])
    income = float(splitted[3])
    miles = float(splitted[4])
    pop = float(splitted[6])
    tax = float(splitted[7])
    
    # Quy đổi các thuộc tính theo hướng dẫn:
    fuel = 1000 * fuelc / pop
    dlic = 1000 * drivers / pop
    logMiles = math.log2(miles)
    
    y_data.append(fuel)
    x_data.append([tax, dlic, income, logMiles])

X = np.array(x_data)
y = np.array(y_data)

# 2. Phân chia tập dữ liệu: 40 mẫu đầu là Training, 11 mẫu sau là Validation
X_train, y_train = X[:40], y[:40]
X_val, y_val = X[40:], y[40:]

print(f"Số lượng mẫu Training: {len(X_train)}")
print(f"Số lượng mẫu Validation: {len(X_val)}")

# 3. Huấn luyện mô hình Hồi quy tuyến tính bằng Scikit-Learn
model = LinearRegression()
model.fit(X_train, y_train)

# In các hệ số trọng số
feature_names = ['Tax', 'Dlic', 'Income', 'log(Miles)']
print("\n--- KẾT QUẢ HUẤN LUYỆN (HỆ SỐ HỒI QUY) ---")
print(f"Hệ số chặn (Intercept - w0): {model.intercept_:.6f}")
for name, coef in zip(feature_names, model.coef_):
    print(f"Hệ số {name:>12}: {coef:.6f}")

# 4. Dự đoán trên cả hai tập
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

# 5. Đánh giá mô hình bằng MSE, MAE, R-square
def print_evaluation(y_true, y_pred, dataset_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n--- ĐÁNH GIÁ TRÊN TẬP {dataset_name} ---")
    print(f"Mean Squared Error (MSE)    : {mse:.4f}")
    print(f"Mean Absolute Error (MAE)   : {mae:.4f}")
    print(f"R-squared (R2 Score)        : {r2:.4f}")
    return mse, mae, r2

train_metrics = print_evaluation(y_train, y_train_pred, "TRAINING (40 mẫu)")
val_metrics = print_evaluation(y_val, y_val_pred, "VALIDATION (11 mẫu)")

# 6. Hiển thị bảng so sánh thực tế vs dự đoán trên tập Validation
print("\n--- CHI TIẾT DỰ ĐOÁN TRÊN TẬP VALIDATION ---")
print(f"{'STT':<5}{'Giá trị thực (y_val)':<25}{'Dự đoán (y_pred)':<25}{'Sai số (e = y - y_pred)':<25}")
print("-" * 75)
for i in range(len(y_val)):
    diff = y_val[i] - y_val_pred[i]
    print(f"{i+1:<5}{y_val[i]:<25.4f}{y_val_pred[i]:<25.4f}{diff:<25.4f}")
