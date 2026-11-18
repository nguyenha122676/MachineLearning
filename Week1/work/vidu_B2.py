# Sinh viên thực hiện: Nguyễn Hải Anh - MSSV: 24001968
"""
VÍ DỤ B.2 (Trang 6-9 PDF)
Hồi quy tuyến tính đa biến trên dữ liệu tiêu thụ nhiên liệu (fuel.txt)
- a) Cách 1: Tự viết thuật toán khai triển QR Householder bằng Numpy
- b) Cách 2: Sử dụng thư viện Scikit-Learn
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
from sklearn import linear_model

print("="*65)
print("VÍ DỤ B.2: HỒI QUY ĐA BIẾN BẰNG QR HOUSEHOLDER & SCIKIT-LEARN")
print("="*65)

# 1. Đọc dữ liệu từ file fuel.txt (theo đúng code trang 6-7 PDF)
data_path = find_data_file("fuel.txt")

with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

x_data = []
y_data = []
lines.pop(0)

for line in lines:
    splitted = line.replace('\n', '').replace('"', '').split(',')
    # Chuyển các trường số thành float
    drivers = float(splitted[1])
    fuelc = float(splitted[2])
    income = float(splitted[3])
    miles = float(splitted[4])
    pop = float(splitted[6])
    tax = float(splitted[7])
    
    fuel = 1000 * fuelc / pop
    dlic = 1000 * drivers / pop
    logMiles = math.log2(miles)
    
    y_data.append([fuel])
    x_data.append([tax, dlic, income, logMiles])

x_data = np.asarray(x_data)
y_data = np.asarray(y_data)

print(f"Kích thước ma trận x_data: {x_data.shape}")
print(f"Kích thước ma trận y_data: {y_data.shape}")

# =========================================================================
# a) Trường hợp sử dụng Numpy - Thuật toán Householder khai triển QR
# =========================================================================
print("\n" + "="*65)
print("a) SỬ DỤNG NUMPY - KHAI TRIỂN QR BẰNG PHẢN XẠ HOUSEHOLDER")
print("="*65)

def qr_householder(A):
    """ Compute QR decomposition of A using Householder reflection """
    M = A.shape[0]
    N = A.shape[1]
    
    # set Q to the identity matrix
    Q = np.identity(M)
    # set R to zero matrix
    R = np.copy(A)
    
    for n in range(N):
        # vector to transform
        x = A[n:, n]
        k = x.shape[0]
        
        # compute ro = -sign(x0) * ||x||
        ro = -np.sign(x[0]) * np.linalg.norm(x)
        
        # compute the householder vector v
        e = np.zeros(k)
        e[0] = 1
        v = (1 / (x[0] - ro)) * (x - (ro * e))
        
        # apply v to each column of A to find R
        for i in range(N):
            R[n:, i] = R[n:, i] - (2 / (v @ v)) * ((np.outer(v, v)) @ R[n:, i])
            
        # apply v to each column of Q
        for i in range(M):
            Q[n:, i] = Q[n:, i] - (2 / (v @ v)) * ((np.outer(v, v)) @ Q[n:, i])
            
    return Q.transpose(), R

def linear_regression_qr(x_data, y_data):
    """ Tính hệ số hồi quy qua QR """
    # add column 1
    x_bars = np.concatenate((np.ones((x_data.shape[0], 1)), x_data), axis=1)
    Q, R = qr_householder(x_bars) # QR decomposition
    R_pinv = np.linalg.pinv(R)     # calculate inverse matrix of R
    A = np.dot(R_pinv, Q.T)       # apply formula
    return np.dot(A, y_data)

w_qr = linear_regression_qr(x_data, y_data)
w_list_qr = w_qr.T.tolist()[0]

line_labels = ['Intercept', 'Tax', 'Dlic', 'Income', 'LogMiles']
print("\nKết quả trọng số w từ thuật toán QR Householder:")
for name, val in zip(line_labels, w_list_qr):
    print(f"{name:>20}: {val:>15.6f}")

# =========================================================================
# b) Trường hợp sử dụng thư viện Scikit-Learn
# =========================================================================
print("\n" + "="*65)
print("b) SỬ DỤNG THƯ VIỆN SCIKIT-LEARN")
print("="*65)

# Bổ sung cột 1 vào x_data
Xbar = np.concatenate((np.ones((x_data.shape[0], 1)), x_data), axis=1)

regr = linear_model.LinearRegression(fit_intercept=False)
regr.fit(Xbar, y_data)
w_sklearn = regr.coef_[0]

print("Kết quả trọng số w từ Scikit-Learn:")
for name, val in zip(line_labels, w_sklearn):
    print(f"{name:>20}: {val:>15.6f}")

print("\n=> NHẬN XÉT: Kết quả của thuật toán tự cài đặt QR Householder và Scikit-Learn hoàn toàn trùng khớp nhau!")
