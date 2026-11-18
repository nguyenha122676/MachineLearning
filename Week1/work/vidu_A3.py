# Sinh viên thực hiện: Nguyễn Hải Anh - MSSV: 24001968
"""
VÍ DỤ A.3 (Trang 3-5 PDF)
Thuật toán K-Nearest Neighbors (K-NN) đa chiều dự báo mức tiêu thụ nhiên liệu (fuel.txt)
- 4 đặc trưng: Tax, Dlic, Income, logMiles
- 40 mẫu đầu làm Training, 11 mẫu sau làm Validation
- k = 6
- Sửa lỗi axis trong hàm distance: dùng axis=1
- Đánh giá bằng MSE, MAE, R-square
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("="*65)
print("VÍ DỤ A.3: K-NN ĐA CHIỀU TRÊN DỮ LIỆU TIÊU THỤ NHIÊN LIỆU (fuel.txt)")
print("="*65)

# 1. Đọc và tiền xử lý dữ liệu
data_path = find_data_file("fuel.txt")

with open(data_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

lines.pop(0) # Bỏ header

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
    
    # Quy đổi thuộc tính theo công thức trong PDF:
    fuel = 1000 * fuelc / pop
    dlic = 1000 * drivers / pop
    logMiles = math.log2(miles)
    
    y_data.append(fuel)
    x_data.append([tax, dlic, income, logMiles])

X = np.asarray(x_data)
y = np.asarray(y_data)

# 2. Chia tập: 40 mẫu đầu Training, 11 mẫu sau Validation
X_train = X[:40]
y_train = y[:40]
X_val = X[40:]
y_val = y[40:]

print(f"Tổng số mẫu: {len(X)}")
print(f" - Tập Training  : {len(X_train)} bang")
print(f" - Tập Validation: {len(X_val)} bang")

# 3. Hàm tính khoảng cách Euclid đa chiều
# LƯU Ý: Trong PDF viết axis=0 kèm chú thích (Cần tự xem xét axis = 0 hay bằng 1!)
# Bắt buộc phải dùng axis=1 để tính chuẩn trên từng hàng (từng mẫu)
def distance(array, value):
    array = np.array(array)
    return np.linalg.norm(array - value, ord=2, axis=1)

def find_nearest_index(array, value, k):
    array_D = distance(array, value)
    return np.argsort(array_D)[:k]

# 4. Dự đoán K-NN với k = 6
k = 6
y_pred = np.zeros(len(X_val))

print(f"\n3. Dự đoán K-NN (k = {k}) cho 11 bang trong tập Validation:")
print(f"{'STT':<6}{'Thực tế (Fuel)':<18}{'K-NN Dự đoán':<18}{'Sai số |y - y_hat|':<18}")
print("-" * 60)

for i in range(len(X_val)):
    nearest_idx = find_nearest_index(X_train, X_val[i], k)
    y_pred[i] = np.mean(y_train[nearest_idx])
    diff = abs(y_val[i] - y_pred[i])
    print(f"{i+1:<6}{y_val[i]:<18.4f}{y_pred[i]:<18.4f}{diff:<18.4f}")

# 5. Đánh giá bằng MSE, MAE, R-square
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"\n4. Đánh giá mô hình K-NN (k = {k}) trên tập Validation:")
print(f" - Sai số bình phương trung bình (MSE) : {mse:.4f}")
print(f" - Sai số tuyệt đối trung bình  (MAE) : {mae:.4f}")
print(f" - Hệ số xác định               (R2)  : {r2:.4f}")
