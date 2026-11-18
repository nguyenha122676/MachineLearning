"""
Script kiểm tra toàn bộ thư viện Hồi quy & Độ đo Machine Learning
Chạy: python test_all_metrics.py
"""
import sys
import io
import numpy as np

# Đảm bảo UTF-8 trên Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

def test_regression():
    print("--- 1. KIEM TRA HOI QUY TUYEN TINH & DO DO HOI QUY ---")
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2.1, 3.9, 6.1, 8.0, 9.9])
    
    model = LinearRegression().fit(X, y)
    y_p = model.predict(X)
    
    print(f"Nghiem: y = {model.coef_[0]:.2f} * x + {model.intercept_:.2f}")
    print(f"MSE : {mean_squared_error(y, y_p):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y, y_p)):.4f}")
    print(f"MAE : {mean_absolute_error(y, y_p):.4f}")
    print(f"R2  : {r2_score(y, y_p):.4f}")
    assert r2_score(y, y_p) > 0.99, "Lỗi: R2 phải xấp xỉ 1.0"
    print("=> HOI QUY: PASS [OK]\n")

def test_classification():
    print("--- 2. KIEM TRA HOI QUY LOGISTIC & DO DO PHAN LOAI ---")
    X = np.array([[-2], [-1], [-0.5], [0.5], [1], [2]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    clf = LogisticRegression().fit(X, y)
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1]
    
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
    print(f"Accuracy : {accuracy_score(y, y_pred):.4f}")
    print(f"Precision: {precision_score(y, y_pred):.4f}")
    print(f"Recall   : {recall_score(y, y_pred):.4f}")
    print(f"F1-Score : {f1_score(y, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y, y_prob):.4f}")
    assert accuracy_score(y, y_pred) == 1.0, "Lỗi: Accuracy phải bằng 1.0"
    print("=> PHAN LOAI: PASS [OK]\n")

if __name__ == "__main__":
    test_regression()
    test_classification()
    print("CHUC MUNG! TAT CA CAC DO DO VA MO HINH HOAT DONG HOAN HAO!")
