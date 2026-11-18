# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("--- PHẦN A.1: K-NN 1D (20 SV ÔN THI) ---")
X_a1 = np.array([0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
                 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50])
y_a1 = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

test_students = pd.DataFrame({
    'Student ID': [21, 22, 23, 24, 25],
    'Hours': [2.45, 1.85, 3.75, 3.21, 4.05]
})

def knn_predict_1d(X_train, y_train, x_val, k):
    # Tính khoảng cách
    dists = np.abs(X_train - x_val)
    indexes = np.argsort(dists)[:k]
    prob = np.mean(y_train[indexes])
    pred = 1 if prob >= 0.5 else 0
    return pred, prob

for k in [3, 4, 5]:
    preds = [knn_predict_1d(X_a1, y_a1, h, k)[0] for h in test_students['Hours']]
    probs = [knn_predict_1d(X_a1, y_a1, h, k)[1] for h in test_students['Hours']]
    test_students[f'Pred (K={k})'] = preds
    test_students[f'Prob (K={k})'] = probs

print(test_students)

print("\n--- PHẦN A.2: K-NN ĐỘ DÀY NTM (vidu4_lin_reg.txt) ---")
df_ntm = pd.read_csv('vidu4_lin_reg.txt', sep=r'\s+')
y_ntm = (df_ntm['BEDAYNTM'] >= 1.0).astype(int).values
X_ntm = df_ntm.drop(columns=['ID', 'BEDAYNTM']).values

X_train_ntm, y_train_ntm = X_ntm[:80], y_ntm[:80]
X_val_ntm, y_val_ntm = X_ntm[80:], y_ntm[80:]

def knn_predict_multidim(X_tr, y_tr, X_va, k=5):
    preds = []
    for x in X_va:
        dists = np.linalg.norm(X_tr - x, ord=2, axis=1)
        indexes = np.argsort(dists)[:k]
        prob = np.mean(y_tr[indexes])
        preds.append(1 if prob >= 0.5 else 0)
    return np.array(preds)

y_pred_ntm = knn_predict_multidim(X_train_ntm, y_train_ntm, X_val_ntm, k=5)
print("Accuracy K=5:", accuracy_score(y_val_ntm, y_pred_ntm))
print("Precision K=5:", precision_score(y_val_ntm, y_pred_ntm, zero_division=0))
print("Recall K=5:", recall_score(y_val_ntm, y_pred_ntm, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_val_ntm, y_pred_ntm))

print("\n--- PHẦN B.1: LOGISTIC REGRESSION SCRATCH (20 SV) ---")
def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-s))

def logistic_sigmoid_regression(X, y, w_init, eta=0.05, tol=1e-4, max_count=10000):
    w = [w_init]
    N = X.shape[1]
    d = X.shape[0]
    count = 0
    check_w_after = 20
    while count < max_count:
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = y[i]
            zi = sigmoid(np.dot(w[-1].T, xi))
            w_new = w[-1] + eta * (yi - zi) * xi
            count += 1
            if count % check_w_after == 0:
                if len(w) >= check_w_after and np.linalg.norm(w_new - w[-check_w_after]) < tol:
                    return w
                w.append(w_new)
    return w

X_b1 = np.array([X_a1]) # (1, 20)
X_b1_bar = np.concatenate((np.ones((1, X_b1.shape[1])), X_b1), axis=0) # (2, 20)
np.random.seed(2)
w_init = np.random.randn(2, 1)
w_history = logistic_sigmoid_regression(X_b1_bar, y_a1, w_init, eta=0.05)
w_best = w_history[-1]
print("Trọng số w:", w_best.ravel())

# Dự đoán cho sinh viên 21-25
X_test_b1 = np.array(test_students['Hours']).reshape(1, -1)
X_test_b1_bar = np.concatenate((np.ones((1, X_test_b1.shape[1])), X_test_b1), axis=0)
probs_b1 = sigmoid(np.dot(w_best.T, X_test_b1_bar)).ravel()
preds_b1 = (probs_b1 >= 0.5).astype(int)
test_students['Prob LogReg'] = np.round(probs_b1, 4)
test_students['Pred LogReg'] = preds_b1
print(test_students[['Student ID', 'Hours', 'Pred (K=3)', 'Pred (K=5)', 'Pred LogReg', 'Prob LogReg']])

print("\n--- PHẦN B.3: ADMISSION PREDICTION ---")
df_adm = pd.read_csv('Admission_Predict.csv')
target_col = 'Chance of Admit'
features_adm = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']

X_adm = df_adm[features_adm].values
y_adm_cont = df_adm[target_col].values
y_adm_bin = (y_adm_cont >= 0.75).astype(int)

# 350 train, 50 test
X_tr_adm, X_te_adm = X_adm[:350], X_adm[350:]
y_tr_adm, y_te_adm = y_adm_bin[:350], y_adm_bin[350:]
y_tr_cont, y_te_cont = y_adm_cont[:350], y_adm_cont[350:]

# a) Logistic Regression (custom & sklearn)
# Dùng StandardScaler để hội tụ ổn định
scaler = StandardScaler()
X_tr_adm_sc = scaler.fit_transform(X_tr_adm)
X_te_adm_sc = scaler.transform(X_te_adm)

# Custom Logistic Regression
X_tr_sc_bar = np.concatenate((np.ones((1, len(X_tr_adm_sc))), X_tr_adm_sc.T), axis=0)
X_te_sc_bar = np.concatenate((np.ones((1, len(X_te_adm_sc))), X_te_adm_sc.T), axis=0)
w_init_adm = np.zeros((X_tr_sc_bar.shape[0], 1))
t0 = time.perf_counter()
w_adm = logistic_sigmoid_regression(X_tr_sc_bar, y_tr_adm, w_init_adm, eta=0.01, max_count=20000)
t_logreg_custom = time.perf_counter() - t0
prob_adm_custom = sigmoid(np.dot(w_adm[-1].T, X_te_sc_bar)).ravel()
pred_adm_custom = (prob_adm_custom >= 0.5).astype(int)
print("Custom LogReg Acc:", accuracy_score(y_te_adm, pred_adm_custom))

# Sklearn LogReg
t0 = time.perf_counter()
clf_lr = LogisticRegression(penalty=None, max_iter=1000)
clf_lr.fit(X_tr_adm_sc, y_tr_adm)
t_logreg_sk = time.perf_counter() - t0
pred_lr_sk = clf_lr.predict(X_te_adm_sc)
print("Sklearn LogReg Acc:", accuracy_score(y_te_adm, pred_lr_sk))
print("Sklearn LogReg Prec:", precision_score(y_te_adm, pred_lr_sk))
print("Sklearn LogReg Recall:", recall_score(y_te_adm, pred_lr_sk))

# b) Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_tr_adm, y_tr_cont)
pred_lin = lin_reg.predict(X_te_adm)
mse = mean_squared_error(y_te_cont, pred_lin)
print("Linear Regression MSE:", mse)

# c) Naive Bayes (GaussianNB)
t0 = time.perf_counter()
gnb = GaussianNB()
gnb.fit(X_tr_adm, y_tr_adm)
t_gnb = time.perf_counter() - t0
pred_gnb = gnb.predict(X_te_adm)
print("GaussianNB Acc:", accuracy_score(y_te_adm, pred_gnb))
print("GaussianNB Prec:", precision_score(y_te_adm, pred_gnb))
print("GaussianNB Recall:", recall_score(y_te_adm, pred_gnb))
print(f"Time LogReg (sklearn): {t_logreg_sk:.6f}s vs GaussianNB: {t_gnb:.6f}s")

print("\n--- PHẦN B.4: BANKING PREDICTION ---")
df_bank = pd.read_csv('banking.csv')
dict_month = {'jan' : 1, 'feb' : 2, 'mar' : 3, 'apr' : 4, 'may' : 5, 'jun' : 6,
              'jul' : 7, 'aug' : 8, 'sep' : 9, 'oct' : 10, 'nov' : 11, 'dec' : 12}
df_bank['month'] = df_bank['month'].map(dict_month)
dict_day = {'sun' : 1, 'mon' : 2, 'tue' : 3, 'wed' : 4, 'thu' : 5, 'fri' : 6, 'sat' : 7}
df_bank['day_of_week'] = df_bank['day_of_week'].map(dict_day)

# Binary fields
df_bank['default'] = df_bank['default'].replace({'no': 0, 'yes': 1, 'unknown': 0})
df_bank['housing'] = df_bank['housing'].replace({'no': 0, 'yes': 1, 'unknown': 0})
df_bank['loan'] = df_bank['loan'].replace({'no': 0, 'yes': 1, 'unknown': 0})

# One-hot encoding
for col in ['marital', 'job', 'education', 'contact', 'poutcome']:
    dummies = pd.get_dummies(df_bank[col], prefix=col, drop_first=True)
    df_bank = pd.concat([df_bank, dummies], axis=1)

df_bank['pdays'] = df_bank['pdays'].apply(lambda row: 0 if (row == -1 or row == 999) else 1)
df_bank.drop(['job', 'education', 'marital', 'contact', 'poutcome'], axis=1, inplace=True)

X_bank = df_bank.drop(columns=['y']).values
y_bank = df_bank['y'].values

X_tr_bk, X_te_bk, y_tr_bk, y_te_bk = train_test_split(X_bank, y_bank, test_size=0.2, random_state=42)
scaler_bk = StandardScaler()
X_tr_bk_sc = scaler_bk.fit_transform(X_tr_bk)
X_te_bk_sc = scaler_bk.transform(X_te_bk)

# Logistic Regression
t0 = time.perf_counter()
lr_bk = LogisticRegression(max_iter=1000)
lr_bk.fit(X_tr_bk_sc, y_tr_bk)
t_lr_bk = time.perf_counter() - t0
pred_lr_bk = lr_bk.predict(X_te_bk_sc)

# Naive Bayes
t0 = time.perf_counter()
nb_bk = GaussianNB()
nb_bk.fit(X_tr_bk_sc, y_tr_bk)
t_nb_bk = time.perf_counter() - t0
pred_nb_bk = nb_bk.predict(X_te_bk_sc)

print("Banking LogReg:")
print(f"  Acc: {accuracy_score(y_te_bk, pred_lr_bk):.4f}, Prec: {precision_score(y_te_bk, pred_lr_bk):.4f}, Rec: {recall_score(y_te_bk, pred_lr_bk):.4f}, F1: {f1_score(y_te_bk, pred_lr_bk):.4f}, Time: {t_lr_bk:.4f}s")
print("Banking GaussianNB:")
print(f"  Acc: {accuracy_score(y_te_bk, pred_nb_bk):.4f}, Prec: {precision_score(y_te_bk, pred_nb_bk):.4f}, Rec: {recall_score(y_te_bk, pred_nb_bk):.4f}, F1: {f1_score(y_te_bk, pred_nb_bk):.4f}, Time: {t_nb_bk:.4f}s")

print("\n--- PHẦN B.6: FRAMINGHAM CHRONIC HEART DISEASE ---")
df_fram = pd.read_csv('framingham.csv')
print("Trước khi dropna:", df_fram.shape)
df_fram = df_fram.dropna(how='any', axis=0)
print("Sau khi dropna:", df_fram.shape)

X_fram = df_fram.drop(columns=['TenYearCHD']).values
y_fram = df_fram['TenYearCHD'].values

X_tr_fr, X_val_fr, y_tr_fr, y_val_fr = train_test_split(X_fram, y_fram, test_size=0.3, random_state=42)
scaler_fr = StandardScaler()
X_tr_fr_sc = scaler_fr.fit_transform(X_tr_fr)
X_val_fr_sc = scaler_fr.transform(X_val_fr)

lr_fr = LogisticRegression(max_iter=1000)
lr_fr.fit(X_tr_fr_sc, y_tr_fr)
pred_fr = lr_fr.predict(X_val_fr_sc)

print("Framingham LogReg:")
print(f"  Accuracy: {accuracy_score(y_val_fr, pred_fr):.4f}")
print(f"  Precision: {precision_score(y_val_fr, pred_fr, zero_division=0):.4f}")
print(f"  Recall: {recall_score(y_val_fr, pred_fr, zero_division=0):.4f}")
