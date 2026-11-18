# Sinh viên thực hiện: Nguyễn Hải Anh - MSSV: 24001968
"""
VÍ DỤ A.1 (Trang 2-3 PDF)
Thuật toán K-Nearest Neighbors (K-NN) dự đoán GPA từ điểm SAT
- Dữ liệu: SAT_GPA.csv (84 sinh viên)
- 64 mẫu đầu làm Training, 20 mẫu sau làm Validation
- k = 8 = sqrt(64)
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("="*65)
print("VÍ DỤ A.1: THUẬT TOÁN K-NN TRÊN DỮ LIỆU SAT_GPA.csv")
print("="*65)

# 1. Đọc dữ liệu
data_path = find_data_file("SAT_GPA.csv")
data = pd.read_csv(data_path)

print("\n1. Mô tả sơ lược dữ liệu (describe):")
print(data.describe())

# Trích xuất dữ liệu
y = data['GPA'].values
x = data['SAT'].values

# 2. Phân chia tập dữ liệu
data_len = len(x)
X_train = np.array(x[:64])
Y_train = np.array(y[:64])
X_test = np.array(x[64:data_len])
Y_test = np.array(y[64:data_len])

print(f"\n2. Kích thước tập dữ liệu:")
print(f" - Tập Training  : {len(X_train)} mẫu (64 mẫu đầu)")
print(f" - Tập Validation: {len(X_test)} mẫu (20 mẫu sau)")

# 3. Định nghĩa các hàm thuật toán K-NN (theo đúng code mẫu trong PDF)
k = 8

def distance(array, value):
    array = np.array(array)
    return abs(array - value)

def find_nearest_index(array, value, k):
    array_D = distance(array, value)
    return np.argsort(array_D)[:k]

# 4. Dự đoán đầu ra cho tập Validation
Y_pred = np.zeros(len(X_test))

print("\n3. Chi tiết dự đoán trên tập Validation:")
print(f"{'STT':<6}{'Điểm SAT':<12}{'GPA Dự đoán':<15}{'GPA Thực tế':<15}{'Sai số |y - y_hat|':<15}")
print("-" * 65)

for i in range(len(X_test)):
    indexis = find_nearest_index(X_train, X_test[i], k)
    for id in indexis:
        Y_pred[i] = Y_pred[i] + Y_train[id]
    Y_pred[i] = Y_pred[i] / len(indexis)
    diff = abs(Y_test[i] - Y_pred[i])
    print(f"{i+1:<6}{X_test[i]:<12.1f}{Y_pred[i]:<15.4f}{Y_test[i]:<15.4f}{diff:<15.4f}")

# 5. Đánh giá kết quả bằng MSE, MAE, R-square (theo yêu cầu trang 3 PDF)
mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("\n4. Kết quả các độ đo đánh giá trên tập Validation:")
print(f" - Sai số bình phương trung bình (MSE) : {mse:.6f}")
print(f" - Sai số tuyệt đối trung bình  (MAE) : {mae:.6f}")
print(f" - Hệ số xác định               (R2)  : {r2:.6f}")

# 6. Vẽ đồ thị dữ liệu
plt.figure(figsize=(9, 5))
plt.scatter(X_train, Y_train, color='blue', label='Train (64 mẫu)')
plt.scatter(X_test, Y_test, color='red', marker='s', label='Validation thực tế (20 mẫu)')
plt.scatter(X_test, Y_pred, color='green', marker='^', label=f'K-NN Dự đoán (k={k})')
plt.xlabel('Điểm SAT', fontsize=12)
plt.ylabel('Điểm GPA', fontsize=12)
plt.title(f'Ví dụ A.1: K-NN trên SAT_GPA (k = {k})', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plot_path = os.path.join(BASE_DIR, "hinh_A1_knn_sat_gpa.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nĐã lưu đồ thị minh họa vào: {plot_path}")
