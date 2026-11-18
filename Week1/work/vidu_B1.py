# Sinh viên thực hiện: Nguyễn Hải Anh - MSSV: 24001968
"""
VÍ DỤ B.1 (Trang 5-6 PDF)
Hồi quy tuyến tính đơn biến: Mối liên hệ giữa Nhiệt độ sôi của nước và Áp suất khí quyển
- Dữ liệu: 17 mẫu thực nghiệm từ Ví dụ A.2
- a) Tính toán trực tiếp theo công thức giải tích
- b) Chương trình Python dùng Ma trận giả nghịch đảo (pinv) và vẽ đường hồi quy
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

print("="*65)
print("VÍ DỤ B.1: HỒI QUY TUYẾN TÍNH ĐƠN BIẾN (NHIỆT ĐỘ & ÁP SUẤT)")
print("="*65)

# Dữ liệu
X = np.array([[194.5, 194.3, 197.9, 198.4, 199.4, 199.9, 200.9, 201.1, 
               201.4, 201.3, 203.6, 204.6, 209.5, 208.6, 210.7, 211.9, 212.2]]).T
y = np.array([[20.79, 20.79, 22.40, 22.67, 23.15, 23.35, 23.89, 23.99, 
               24.02, 24.01, 25.14, 26.57, 28.49, 27.76, 29.04, 29.88, 30.06]]).T

# =========================================================================
# a) Tính toán trực tiếp theo công thức giải tích (Trang 5 PDF)
# =========================================================================
print("\n--- a) TÍNH TOÁN TRỰC TIẾP THEO CÔNG THỨC GIẢI TÍCH ---")
x_flat = X.flatten()
y_flat = y.flatten()

x_bar = np.mean(x_flat)
y_bar = np.mean(y_flat)

sxx = np.sum((x_flat - x_bar) ** 2)
sxy = np.sum((x_flat - x_bar) * (y_flat - y_bar))

theta_1 = sxy / sxx
theta_0 = y_bar - theta_1 * x_bar

print(f" - Trung bình cộng của X (x_bar): {x_bar:.7f}")
print(f" - Trung bình cộng của Y (y_bar): {y_bar:.7f}")
print(f" - Phương sai mẫu SXX           : {sxx:.7f}")
print(f" - Hiệp phương sai mẫu SXY       : {sxy:.7f}")
print(f" - Hệ số góc theta_1 (SXY / SXX): {theta_1:.9f}")
print(f" - Hệ số chặn theta_0           : {theta_0:.8f}")
print(f" => Phương trình: Áp suất = {theta_0:.4f} + {theta_1:.4f} * Nhiệt độ")

# =========================================================================
# b) Chương trình Python dùng Ma trận giả nghịch đảo (Trang 5-6 PDF)
# =========================================================================
print("\n--- b) CHƯƠNG TRÌNH PYTHON VỚI MA TRẬN GIẢ NGHỊCH ĐẢO (pinv) ---")

# Building Xbar (thêm cột 1 biểu diễn hệ số chặn w0)
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)

# Calculating weights of the fitting line
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)

print("Trọng số w tính được từ numpy.linalg.pinv:")
print(f" - w_0 (Hệ số chặn) = {w[0][0]:.8f}")
print(f" - w_1 (Hệ số góc)  = {w[1][0]:.9f}")

# Chuẩn bị đường vẽ (fitting line)
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(193, 213, 100)
y0 = w_0 + w_1 * x0

# Vẽ đồ thị
plt.figure(figsize=(9, 5))
plt.plot(X.T, y.T, 'ro', label='Điểm dữ liệu thực nghiệm')
plt.plot(x0, y0, 'b-', linewidth=2, label=f'Đường hồi quy: y = {w_0:.2f} + {w_1:.4f}*x')
plt.axis([193, 213, 19, 31])
plt.xlabel('Nhiệt độ sôi (°F)', fontsize=12)
plt.ylabel('Áp suất khí quyển (Atm)', fontsize=12)
plt.title('Ví dụ B.1: Đường hồi quy tuyến tính Nhiệt độ - Áp suất', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plot_path = os.path.join(BASE_DIR, "hinh_B1_lin_reg_temp_press.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nĐã vẽ và lưu đồ thị vào: {plot_path}")
