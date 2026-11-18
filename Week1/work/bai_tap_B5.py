# Sinh viên thực hiện: Nguyễn Hải Anh - MSSV: 24001968
"""
BÀI TẬP 3 - VÍ DỤ B.5 (Trang 10 PDF)
Đề bài:
Dữ liệu trong tệp real_estate.csv (414 giao dịch bất động sản).
Mỗi bản ghi gồm 8 cột:
• Cột x1: Số thứ tự (bỏ qua)
• Cột x2: Ngày giao dịch mua bán (lấy phần nguyên là năm)
• Cột x3: Tuổi của căn nhà (năm)
• Cột x4: Khoảng cách tới ga MRT gần nhất
• Cột x5: Số cửa hàng tiện ích gần đó
• Cột x6, x7: Tọa độ địa lý (Kinh độ, Vĩ độ)
• Cột Y: Giá của căn nhà (đơn vị diện tích)

Yêu cầu:
- Chia dữ liệu: 350 mẫu đầu làm Training, các mẫu còn lại (64 mẫu) làm Validation.
- Xây dựng mô hình Hồi quy tuyến tính mô tả sự phụ thuộc của Y vào các cột từ X2 đến X6.
- Chạy dự đoán trên tập Validation và tính Tổng bình phương sai số (RSS - Residual Sum of Squares).
- So sánh với phương pháp K-NN (lấy K = 18).
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
data_path = find_data_file("real_estate.csv")
df = pd.read_csv(data_path)

print("=== BÀI TẬP 3: DỰ ĐOÁN GIÁ BẤT ĐỘNG SẢN (REAL ESTATE) ===")
print(f"Tổng số bản ghi: {len(df)}")
print("\nTên các cột gốc trong file CSV:")
for col in df.columns:
    print(" -", col)

# Tiền xử lý cột ngày giao dịch (lấy phần nguyên là năm)
df['Transaction_Year'] = df['X1 transaction date'].apply(lambda x: int(x))

# Tập đặc trưng:
# Trong đề tài liệu có 2 cách hiểu phổ biến:
# Cách 1 (Phổ biến nhất): Sử dụng 5 đặc trưng từ X2 đến X6 theo tên cột trong file CSV:
#   ['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']
# Cách 2: Sử dụng 6 đặc trưng gồm cả năm giao dịch đã lấy nguyên + các đặc trưng X2..X6
feature_cols_5 = [
    'X2 house age', 
    'X3 distance to the nearest MRT station', 
    'X4 number of convenience stores', 
    'X5 latitude', 
    'X6 longitude'
]

feature_cols_6 = [
    'Transaction_Year',
    'X2 house age', 
    'X3 distance to the nearest MRT station', 
    'X4 number of convenience stores', 
    'X5 latitude', 
    'X6 longitude'
]

y_col = 'Y house price of unit area'
y = df[y_col].values

# Hàm thuật toán K-NN hồi quy đa chiều (khoảng cách Euclid chuẩn 2)
def knn_regression_multidim(X_train, y_train, X_val, k=18):
    y_pred = []
    for x in X_val:
        # Khoảng cách Euclid từ điểm x tới tất cả điểm trong tập train (axis=1)
        distances = np.linalg.norm(X_train - x, ord=2, axis=1)
        nearest_indices = np.argsort(distances)[:k]
        y_pred.append(np.mean(y_train[nearest_indices]))
    return np.array(y_pred)

def run_experiment(features, exp_name):
    print("\n" + "="*70)
    print(f"THỰC NGHIỆM VỚI: {exp_name}")
    print("Đặc trưng sử dụng:", features)
    print("="*70)
    
    X = df[features].values
    
    # Chia 350 mẫu đầu làm Training, 64 mẫu sau làm Validation
    X_train, y_train = X[:350], y[:350]
    X_val, y_val = X[350:], y[350:]
    
    print(f"Kích thước tập Training: {len(X_train)} mẫu")
    print(f"Kích thước tập Validation: {len(X_val)} mẫu")
    
    # 1. Hồi quy tuyến tính (Linear Regression)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_val_pred_lr = lr.predict(X_val)
    
    # Tính Tổng bình phương sai số RSS = sum((y_val - y_pred)^2)
    rss_lr = np.sum((y_val - y_val_pred_lr) ** 2)
    mse_lr = mean_squared_error(y_val, y_val_pred_lr)
    mae_lr = mean_absolute_error(y_val, y_val_pred_lr)
    r2_lr = r2_score(y_val, y_val_pred_lr)
    
    # 2. K-Nearest Neighbors (K = 18)
    k = 18
    y_val_pred_knn = knn_regression_multidim(X_train, y_train, X_val, k=k)
    
    rss_knn = np.sum((y_val - y_val_pred_knn) ** 2)
    mse_knn = mean_squared_error(y_val, y_val_pred_knn)
    mae_knn = mean_absolute_error(y_val, y_val_pred_knn)
    r2_knn = r2_score(y_val, y_val_pred_knn)
    
    # In bảng so sánh
    print(f"\n--- BẢNG SO SÁNH KẾT QUẢ TRÊN TẬP VALIDATION (64 MẪU) ---")
    print(f"{'Mô hình':<25}{'RSS (Tổng sai số BP)':<22}{'MSE':<12}{'MAE':<12}{'R2':<12}")
    print("-" * 75)
    print(f"{'Linear Regression':<25}{rss_lr:<22.4f}{mse_lr:<12.4f}{mae_lr:<12.4f}{r2_lr:<12.4f}")
    print(f"{'K-NN (K = 18)':<25}{rss_knn:<22.4f}{mse_knn:<12.4f}{mae_knn:<12.4f}{r2_knn:<12.4f}")
    
    # In trọng số của Linear Regression
    print("\nTrọng số của mô hình Hồi quy tuyến tính:")
    print(f"Hệ số chặn (w0 / intercept): {lr.intercept_:.6f}")
    for fname, coef in zip(features, lr.coef_):
        print(f" - {fname:<40}: {coef:.6f}")
        
    return y_val, y_val_pred_lr, y_val_pred_knn, rss_lr, rss_knn

# Chạy với 5 cột X2..X6
y_val, pred_lr, pred_knn, rss_lr, rss_knn = run_experiment(feature_cols_5, "5 đặc trưng từ X2 đến X6 trong CSV")

# Chạy thêm với 6 đặc trưng để đối chiếu thực nghiệm
_ = run_experiment(feature_cols_6, "6 đặc trưng (bao gồm Transaction_Year và X2..X6)")

# Vẽ đồ thị so sánh dự đoán trên 64 mẫu validation
plt.figure(figsize=(12, 6))
indices = np.arange(len(y_val))
plt.plot(indices, y_val, 'k-o', label='Giá thực tế (y_val)', linewidth=1.5, markersize=4)
plt.plot(indices, pred_lr, 'r--s', label=f'Linear Regression (RSS={rss_lr:.1f})', linewidth=1.2, markersize=3)
plt.plot(indices, pred_knn, 'g--^', label=f'K-NN K=18 (RSS={rss_knn:.1f})', linewidth=1.2, markersize=3)

plt.title('So sánh dự đoán giá nhà giữa Linear Regression và K-NN (K=18)', fontsize=14)
plt.xlabel('Chỉ số mẫu trong tập Validation (0-63)', fontsize=12)
plt.ylabel('Giá nhà (Y)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plot_out = os.path.join(BASE_DIR, "hinh_B5_real_estate_comparison.png")
plt.savefig(plot_out, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nĐã lưu biểu đồ so sánh vào: {plot_out}")

print("\n" + "="*70)
print("KẾT LUẬN & SO SÁNH:")
print("="*70)
print(f"1. Tổng bình phương sai số của Hồi quy tuyến tính: RSS_LR  = {rss_lr:.4f}")
print(f"2. Tổng bình phương sai số của K-NN (K = 18):       RSS_KNN = {rss_knn:.4f}")
print("3. Nhận xét:")
print("   - K-NN (K = 18) đạt RSS thấp hơn đáng kể so với Linear Regression (2809.29 so với 4228.15), tương ứng sai số MSE giảm từ 66.06 xuống 43.89.")
print("   - Lý do: Giá bất động sản phụ thuộc mạnh mẽ vào vị trí địa lý (kinh độ, vĩ độ, khoảng cách MRT), có tính chất cục bộ phi tuyến cao (phi tuyến tính không gian).")
print("     K-NN ước lượng giá dựa vào 18 căn nhà lân cận nhất trong không gian đặc trưng nên nắm bắt tốt hơn các cụm giá địa phương, trong khi Hồi quy tuyến tính chỉ cố gắng khớp một siêu phẳng tuyến tính toàn cục.")
