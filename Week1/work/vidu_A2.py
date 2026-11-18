# Sinh viên thực hiện: Nguyễn Hải Anh - MSSV: 24001968
"""
VÍ DỤ A.2 (Trang 3 PDF)
Thuật toán K-Nearest Neighbors (K-NN) ước lượng áp suất khí quyển dựa trên nhiệt độ sôi của nước
- Dữ liệu: 17 mẫu thực nghiệm
- 16 mẫu đầu làm Training, mẫu thứ 17 làm Validation
- Chọn k = 4
- Đánh giá kết quả bằng MSE, MAE
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
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("="*65)
print("VÍ DỤ A.2: K-NN ƯỚC LƯỢNG ÁP SUẤT THEO NHIỆT ĐỘ SÔI")
print("="*65)

# 1. Khởi tạo dữ liệu
# Temp (F degree)
X = np.array([194.5, 194.3, 197.9, 198.4, 199.4, 199.9, 200.9, 201.1, 
              201.4, 201.3, 203.6, 204.6, 209.5, 208.6, 210.7, 211.9, 212.2])

# Press (Atm)
y = np.array([20.79, 20.79, 22.40, 22.67, 23.15, 23.35, 23.89, 23.99, 
              24.02, 24.01, 25.14, 26.57, 28.49, 27.76, 29.04, 29.88, 30.06])

print(f"Tổng số mẫu dữ liệu: {len(X)}")

# 2. Phân chia dữ liệu: 16 mẫu đầu (Training), mẫu cuối (Validation)
X_train = X[:16]
y_train = y[:16]
X_val = X[16:]
y_val = y[16:]

print(f" - Tập Training  : {len(X_train)} mẫu")
print(f" - Tập Validation: {len(X_val)} mẫu (Nhiệt độ = {X_val[0]}°F, Áp suất thực = {y_val[0]} atm)")

# 3. K-NN với k = 4
k = 4

# Tính khoảng cách từ điểm validation tới 16 điểm train
distances = np.abs(X_train - X_val[0])
nearest_indices = np.argsort(distances)[:k]

# Lấy trung bình đầu ra của 4 điểm gần nhất
y_pred = np.mean(y_train[nearest_indices])

print(f"\n3. Kết quả chạy K-NN (k = {k}):")
print(f" - 4 điểm gần nhất có chỉ số (0-based): {nearest_indices}")
print(f" - Nhiệt độ của 4 điểm gần nhất      : {X_train[nearest_indices]}")
print(f" - Áp suất tương ứng của 4 điểm đó   : {y_train[nearest_indices]}")
print(f" - Áp suất thực tế (y_true)          : {y_val[0]:.4f} atm")
print(f" - Áp suất K-NN dự đoán (y_pred)     : {y_pred:.4f} atm")

# 4. Đánh giá sai số
mae = abs(y_val[0] - y_pred)
mse = (y_val[0] - y_pred) ** 2

print("\n4. Đánh giá kết quả:")
print(f" - Sai số tuyệt đối (MAE)           : {mae:.4f}")
print(f" - Sai số bình phương (MSE)         : {mse:.4f}")

# 5. Vẽ đồ thị
plt.figure(figsize=(9, 5))
plt.plot(X_train, y_train, 'bo', label='Tập Training (16 mẫu)')
plt.plot(X_val, y_val, 'ro', markersize=8, label=f'Validation thực tế ({X_val[0]}°F, {y_val[0]} atm)')
plt.plot(X_val, [y_pred], 'g^', markersize=9, label=f'K-NN dự đoán ({X_val[0]}°F, {y_pred:.2f} atm)')
plt.plot(X_train[nearest_indices], y_train[nearest_indices], 'kx', markersize=10, label=f'{k} láng giềng gần nhất')

plt.axis([193, 213, 19, 31])
plt.xlabel('Nhiệt độ - Temperature (°F)', fontsize=12)
plt.ylabel('Áp suất - Pressure (Atm)', fontsize=12)
plt.title(f'Ví dụ A.2: K-NN dự đoán Áp suất theo Nhiệt độ (k = {k})', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plot_path = os.path.join(BASE_DIR, "hinh_A2_knn_temp_press.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nĐã lưu đồ thị minh họa vào: {plot_path}")
