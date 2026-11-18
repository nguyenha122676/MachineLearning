"""
Bài 3: Phân loại văn bản ẩm thực Bắc (B) - Nam (N) bằng Multinomial NB và Bernoulli NB
Họ và tên: Nguyễn Hải Anh - MSSV: 24001968
Môn học: Machine Learning - Bài thực hành số 2 (Naive Bayes)

Nội dung:
- Văn bản huấn luyện:
  D1: "Hà-Nội phở cháo-lòng Hà-Nội" (Lớp B)
  D2: "Hà-Nội bún-chả phở ô-mai" (Lớp B)
  D3: "Phở bánh-giò ô-mai" (Lớp B)
  D4: "Sài-gòn hủ-tiếu bánh-bò phở" (Lớp N)
- Văn bản kiểm thử cần dự đoán:
  D5: "Hà-Nội Hà-Nội bún-chả hủ-tiếu"
  D6: "phở hủ-tiếu bánh-bò"
- Triển khai chi tiết:
  1. Mô hình Multinomial Naive Bayes (Bag-of-Words / Tần suất từ có làm mịn Laplace alpha=1)
  2. Mô hình Bernoulli Naive Bayes (Vector nhị phân 0/1 - sự xuất hiện của từ)
- So sánh kết quả tính tay toán học lý thuyết và kết quả chạy thư viện Scikit-Learn
"""

import sys
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def run_bai3():
    print("=" * 70)
    print(" BÀI 3: PHÂN LOẠI VĂN BẢN BẮC (B) - NAM (N) (MULTINOMIAL & BERNOULLI NB)")
    print("=" * 70)

    # 1. Định nghĩa từ điển vựng (Vocabulary) gồm 9 từ:
    vocab = ['Hà-Nội', 'phở', 'cháo-lòng', 'bún-chả', 'ô-mai', 'bánh-giò', 'Sài-gòn', 'hủ-tiếu', 'bánh-bò']
    V_size = len(vocab)
    print(f"[*] Từ điển V (|V|={V_size}): {vocab}\n")

    # ==============================================================
    # PHẦN A: MULTINOMIAL NAIVE BAYES (ĐẾM TẦN SUẤT TỪ)
    # ==============================================================
    print("--- PHẦN A: MÔ HÌNH MULTINOMIAL NAIVE BAYES (TF / COUNT VECTOR) ---")
    # Biểu diễn vector tần suất (Count / Term Frequency)
    # Index: [Hà-Nội, phở, cháo-lòng, bún-chả, ô-mai, bánh-giò, Sài-gòn, hủ-tiếu, bánh-bò]
    X_train_mnb = np.array([
        [2, 1, 1, 0, 0, 0, 0, 0, 0],  # D1 (B)
        [1, 1, 0, 1, 1, 0, 0, 0, 0],  # D2 (B)
        [0, 1, 0, 0, 1, 1, 0, 0, 0],  # D3 (B)
        [0, 1, 0, 0, 0, 0, 1, 1, 1]   # D4 (N)
    ])
    y_train = np.array(['B', 'B', 'B', 'N'])

    X_test_mnb = np.array([
        [2, 0, 0, 1, 0, 0, 0, 1, 0],  # D5: "Hà-Nội Hà-Nội bún-chả hủ-tiếu"
        [0, 1, 0, 0, 0, 0, 0, 1, 1]   # D6: "phở hủ-tiếu bánh-bò"
    ])

    print("Bảng tần suất từ trong tập Training:")
    df_train_mnb = pd.DataFrame(X_train_mnb, columns=vocab, index=['D1 (B)', 'D2 (B)', 'D3 (B)', 'D4 (N)'])
    print(df_train_mnb)

    # Tính toán lý thuyết toán học có làm mịn Laplace (alpha = 1)
    # Tổng số từ xuất hiện trong class B và class N:
    N_B = np.sum(X_train_mnb[y_train == 'B']) # 11 từ
    N_N = np.sum(X_train_mnb[y_train == 'N']) # 4 từ
    p_B = 3.0 / 4.0
    p_N = 1.0 / 4.0

    print(f"\nTổng số từ trong Lớp B: N_B = {N_B}, Xác suất tiên nghiệm P(B) = {p_B}")
    print(f"Tổng số từ trong Lớp N: N_N = {N_N}, Xác suất tiên nghiệm P(N) = {p_N}")

    # Chạy mô hình Sklearn MultinomialNB (alpha = 1.0)
    clf_mnb = MultinomialNB(alpha=1.0)
    clf_mnb.fit(X_train_mnb, y_train)

    pred_mnb = clf_mnb.predict(X_test_mnb)
    proba_mnb = clf_mnb.predict_proba(X_test_mnb)
    classes_mnb = clf_mnb.classes_ # ['B', 'N']

    print("\nKết quả dự đoán bằng Sklearn MultinomialNB:")
    for i, d_name in enumerate(['D5', 'D6']):
        p_b_pct = proba_mnb[i][0] * 100
        p_n_pct = proba_mnb[i][1] * 100
        print(f"-> Văn bản {d_name}: Dự đoán Lớp [{pred_mnb[i]}] | Xác suất: P(B) = {p_b_pct:.2f}%, P(N) = {p_n_pct:.2f}%")

    # ==============================================================
    # PHẦN B: BERNOULLI NAIVE BAYES (VECTOR NHỊ PHÂN XUẤT HIỆN 0/1)
    # ==============================================================
    print("\n--- PHẦN B: MÔ HÌNH BERNOULLI NAIVE BAYES (BINARY VECTOR 0/1) ---")
    # Biểu diễn vector nhị phân (chỉ quan tâm có hay không xuất hiện)
    X_train_bnb = np.array([
        [1, 1, 1, 0, 0, 0, 0, 0, 0],  # D1 (B)
        [1, 1, 0, 1, 1, 0, 0, 0, 0],  # D2 (B)
        [0, 1, 0, 0, 1, 1, 0, 0, 0],  # D3 (B)
        [0, 1, 0, 0, 0, 0, 1, 1, 1]   # D4 (N)
    ])

    X_test_bnb = np.array([
        [1, 0, 0, 1, 0, 0, 0, 1, 0],  # D5: "Hà-Nội", "bún-chả", "hủ-tiếu"
        [0, 1, 0, 0, 0, 0, 0, 1, 1]   # D6: "phở", "hủ-tiếu", "bánh-bò"
    ])

    print("Bảng vector nhị phân trong tập Training:")
    df_train_bnb = pd.DataFrame(X_train_bnb, columns=vocab, index=['D1 (B)', 'D2 (B)', 'D3 (B)', 'D4 (N)'])
    print(df_train_bnb)

    clf_bnb = BernoulliNB(alpha=1.0)
    clf_bnb.fit(X_train_bnb, y_train)

    pred_bnb = clf_bnb.predict(X_test_bnb)
    proba_bnb = clf_bnb.predict_proba(X_test_bnb)

    print("\nKết quả dự đoán bằng Sklearn BernoulliNB:")
    for i, d_name in enumerate(['D5', 'D6']):
        p_b_pct = proba_bnb[i][0] * 100
        p_n_pct = proba_bnb[i][1] * 100
        print(f"-> Văn bản {d_name}: Dự đoán Lớp [{pred_bnb[i]}] | Xác suất: P(B) = {p_b_pct:.2f}%, P(N) = {p_n_pct:.2f}%")

    # ==============================================================
    # TỔNG KẾT SO SÁNH HAI MÔ HÌNH
    # ==============================================================
    print("\n" + "=" * 50)
    print("BẢNG TỔNG HỢP SO SÁNH DỰ ĐOÁN HAI MÔ HÌNH:")
    print("=" * 50)
    summary_df = pd.DataFrame({
        'Văn bản Test': ['D5 (Hà-Nội Hà-Nội bún-chả hủ-tiếu)', 'D6 (phở hủ-tiếu bánh-bò)'],
        'Multinomial NB Pred': [pred_mnb[0], pred_mnb[1]],
        'Multinomial P(B) / P(N)': [f"{proba_mnb[0][0]*100:.1f}% / {proba_mnb[0][1]*100:.1f}%",
                                    f"{proba_mnb[1][0]*100:.1f}% / {proba_mnb[1][1]*100:.1f}%"],
        'Bernoulli NB Pred': [pred_bnb[0], pred_bnb[1]],
        'Bernoulli P(B) / P(N)': [f"{proba_bnb[0][0]*100:.1f}% / {proba_bnb[0][1]*100:.1f}%",
                                  f"{proba_bnb[1][0]*100:.1f}% / {proba_bnb[1][1]*100:.1f}%"]
    })
    print(summary_df.to_string(index=False))

    return {
        'mnb_pred': pred_mnb,
        'mnb_proba': proba_mnb,
        'bnb_pred': pred_bnb,
        'bnb_proba': proba_bnb
    }

if __name__ == "__main__":
    run_bai3()
