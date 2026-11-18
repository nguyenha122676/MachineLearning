# Sinh viên thực hiện: Nguyễn Hải Anh - MSSV: 24001968
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
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------------------------------------------------
# 1. Ví dụ A.1: K-NN trên SAT_GPA
# -------------------------------------------------------------
print("=== 1. VÍ DỤ A.1: K-NN SAT_GPA ===")
df_sat = pd.read_csv(find_data_file("SAT_GPA.csv"))
x_sat = df_sat['SAT'].values
y_sat = df_sat['GPA'].values

X_train_sat = x_sat[:64]
y_train_sat = y_sat[:64]
X_val_sat = x_sat[64:]
y_val_sat = y_sat[64:]

k_sat = 8
def knn_1d(X_train, y_train, X_val, k):
    y_pred = []
    for x in X_val:
        dists = np.abs(X_train - x)
        idx = np.argsort(dists)[:k]
        y_pred.append(np.mean(y_train[idx]))
    return np.array(y_pred)

y_pred_knn_sat = knn_1d(X_train_sat, y_train_sat, X_val_sat, k_sat)
mse_knn_sat = mean_squared_error(y_val_sat, y_pred_knn_sat)
mae_knn_sat = mean_absolute_error(y_val_sat, y_pred_knn_sat)
r2_knn_sat = r2_score(y_val_sat, y_pred_knn_sat)
print(f"K-NN (k={k_sat}) on SAT_GPA Validation:")
print(f"MSE: {mse_knn_sat:.5f}, MAE: {mae_knn_sat:.5f}, R2: {r2_knn_sat:.5f}")

# -------------------------------------------------------------
# 2. Bài tập B.1: Hồi quy tuyến tính trên fuel.txt
# -------------------------------------------------------------
print("\n=== 2. BÀI TẬP B.1: Linear Regression fuel.txt ===")
with open(find_data_file("fuel.txt")) as f:
    lines = f.readlines()
lines.pop(0)

x_data_fuel = []
y_data_fuel = []
for line in lines:
    splitted = line.replace('\n', '').replace('"', '').split(',')
    # State is index 0
    state = splitted[0]
    drivers = float(splitted[1])
    fuelc = float(splitted[2])
    income = float(splitted[3])
    miles = float(splitted[4])
    mpc = splitted[5]
    pop = float(splitted[6])
    tax = float(splitted[7])
    
    fuel = 1000 * fuelc / pop
    dlic = 1000 * drivers / pop
    logMiles = math.log2(miles)
    
    y_data_fuel.append(fuel)
    x_data_fuel.append([tax, dlic, income, logMiles])

x_fuel = np.array(x_data_fuel)
y_fuel = np.array(y_data_fuel)

X_train_f, y_train_f = x_fuel[:40], y_fuel[:40]
X_val_f, y_val_f = x_fuel[40:], y_fuel[40:]

lr_fuel = LinearRegression()
lr_fuel.fit(X_train_f, y_train_f)

pred_train_f = lr_fuel.predict(X_train_f)
pred_val_f = lr_fuel.predict(X_val_f)

print("Intercept:", lr_fuel.intercept_)
print("Coefficients [Tax, Dlic, Income, logMiles]:", lr_fuel.coef_)
print(f"Train - MSE: {mean_squared_error(y_train_f, pred_train_f):.4f}, MAE: {mean_absolute_error(y_train_f, pred_train_f):.4f}, R2: {r2_score(y_train_f, pred_train_f):.4f}")
print(f"Val   - MSE: {mean_squared_error(y_val_f, pred_val_f):.4f}, MAE: {mean_absolute_error(y_val_f, pred_val_f):.4f}, R2: {r2_score(y_val_f, pred_val_f):.4f}")

# -------------------------------------------------------------
# 3. Ví dụ B.3 (Bài tập tự thực hành 1): Linear Regression SAT_GPA
# -------------------------------------------------------------
print("\n=== 3. VÍ DỤ B.3 (BÀI TẬP TỰ THỰC HÀNH 1): SAT_GPA ===")
pearson_r = np.corrcoef(x_sat, y_sat)[0, 1]
print(f"(i) Hệ số tương quan Pearson (GPA vs SAT toàn bộ): {pearson_r:.5f}")
pearson_r_train = np.corrcoef(X_train_sat, y_train_sat)[0, 1]
print(f"    Pearson r trên tập Train (64 mẫu): {pearson_r_train:.5f}")

lr_sat = LinearRegression()
lr_sat.fit(X_train_sat.reshape(-1, 1), y_train_sat)
w0 = lr_sat.intercept_
w1 = lr_sat.coef_[0]
print(f"(iii) Phương trình hồi quy: GPA = {w0:.6f} + {w1:.6f} * SAT")

y_pred_val_lr_sat = lr_sat.predict(X_val_sat.reshape(-1, 1))
mse_lr_sat = mean_squared_error(y_val_sat, y_pred_val_lr_sat)
mae_lr_sat = mean_absolute_error(y_val_sat, y_pred_val_lr_sat)
r2_lr_sat = r2_score(y_val_sat, y_pred_val_lr_sat)
print(f"(iv) Đánh giá trên Validation (Linear Regression):")
print(f"     MSE: {mse_lr_sat:.5f}, MAE: {mae_lr_sat:.5f}, R2: {r2_lr_sat:.5f}")
print(f"(v) So sánh với K-NN (k=8):")
print(f"    Linear Regression - MSE: {mse_lr_sat:.5f}, MAE: {mae_lr_sat:.5f}, R2: {r2_lr_sat:.5f}")
print(f"    K-NN (k=8)        - MSE: {mse_knn_sat:.5f}, MAE: {mae_knn_sat:.5f}, R2: {r2_knn_sat:.5f}")

# -------------------------------------------------------------
# 4. Ví dụ B.4 (Bài tập 2): vidu4_lin_reg.txt
# -------------------------------------------------------------
print("\n=== 4. VÍ DỤ B.4 (BÀI TẬP 2): vidu4_lin_reg.txt ===")
# Header: ID TUOI BMI HA GLUCOSE CHOLESTEROL BEDAYNTM
df_ntm = pd.read_csv(find_data_file("vidu4_lin_reg.txt"), sep=r'\s+')
# Variables: Tuoi, Cholesterol, Glucose, HA, BMI
# Formula: BEDAYNTM = b0 + b1*TUOI + b2*CHOLESTEROL + b3*GLUCOSE + b4*HA + b5*BMI
features = ['TUOI', 'CHOLESTEROL', 'GLUCOSE', 'HA', 'BMI']
X_ntm = df_ntm[features].values
y_ntm = df_ntm['BEDAYNTM'].values

# a) 100 samples
lr_ntm_all = LinearRegression()
lr_ntm_all.fit(X_ntm, y_ntm)
print("a) Hệ số với 100 mẫu:")
print(f"   b0 (Intercept): {lr_ntm_all.intercept_:.6f}")
for name, coef in zip(features, lr_ntm_all.coef_):
    print(f"   {name}: {coef:.6f}")

# b) 80 train / 20 test
X_train_ntm, y_train_ntm = X_ntm[:80], y_ntm[:80]
X_test_ntm, y_test_ntm = X_ntm[80:], y_ntm[80:]

lr_ntm_split = LinearRegression()
lr_ntm_split.fit(X_train_ntm, y_train_ntm)
print("\nb) Hệ số với 80 mẫu train:")
print(f"   b0 (Intercept): {lr_ntm_split.intercept_:.6f}")
for name, coef in zip(features, lr_ntm_split.coef_):
    print(f"   {name}: {coef:.6f}")

y_pred_test_ntm = lr_ntm_split.predict(X_test_ntm)
errors_ntm = y_test_ntm - y_pred_test_ntm
print(f"   Kỳ vọng sai số (Mean error): {np.mean(errors_ntm):.6f}")
print(f"   Phương sai sai số (Variance of error): {np.var(errors_ntm):.6f}")
print(f"   Độ lệch chuẩn sai số (Std of error): {np.std(errors_ntm):.6f}")
print(f"   MSE test: {mean_squared_error(y_test_ntm, y_pred_test_ntm):.6f}")

# -------------------------------------------------------------
# 5. Ví dụ B.5 (Bài tập 3): Real_estate.csv
# -------------------------------------------------------------
print("\n=== 5. VÍ DỤ B.5 (BÀI TẬP 3): real_estate.csv ===")
df_re = pd.read_csv(find_data_file("real_estate.csv"))
# Columns: No, X1 transaction date, X2 house age, X3 distance to the nearest MRT station,
# X4 number of convenience stores, X5 latitude, X6 longitude, Y house price of unit area
# Let's test with:
# Case 1: Features = [X1_year, X2, X3, X4, X5, X6]
df_re['X1_year'] = df_re['X1 transaction date'].apply(lambda x: int(x))

feat_cols_all = ['X1_year', 'X2 house age', 'X3 distance to the nearest MRT station',
                 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']

feat_cols_x2_x6 = ['X2 house age', 'X3 distance to the nearest MRT station',
                   'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']

for name, cols in [("Case A (gồm X1_year và X2..X6)", feat_cols_all), 
                   ("Case B (chỉ gồm các cột tên X2..X6)", feat_cols_x2_x6)]:
    X_re = df_re[cols].values
    y_re = df_re['Y house price of unit area'].values
    
    X_train_re, y_train_re = X_re[:350], y_re[:350]
    X_val_re, y_val_re = X_re[350:], y_re[350:]
    
    # Linear Regression
    lr_re = LinearRegression()
    lr_re.fit(X_train_re, y_train_re)
    y_pred_lr_re = lr_re.predict(X_val_re)
    rss_lr = np.sum((y_val_re - y_pred_lr_re)**2)
    mse_lr = mean_squared_error(y_val_re, y_pred_lr_re)
    
    # K-NN (K=18)
    def knn_regression(X_train, y_train, X_val, k):
        y_pred = []
        for x in X_val:
            dists = np.linalg.norm(X_train - x, axis=1)
            idx = np.argsort(dists)[:k]
            y_pred.append(np.mean(y_train[idx]))
        return np.array(y_pred)
    
    y_pred_knn_re = knn_regression(X_train_re, y_train_re, X_val_re, 18)
    rss_knn = np.sum((y_val_re - y_pred_knn_re)**2)
    mse_knn = mean_squared_error(y_val_re, y_pred_knn_re)
    
    print(f"\n--- {name} ---")
    print(f"Linear Regression - RSS: {rss_lr:.4f}, MSE: {mse_lr:.4f}")
    print(f"K-NN (K=18)       - RSS: {rss_knn:.4f}, MSE: {mse_knn:.4f}")
