"""
HỆ THỐNG MÔ PHỎNG BÀI KIỂM TRA CODE TỰ ĐỘNG CHẤM ĐIỂM — CHƯƠNG NAIVE BAYES
===========================================================================
Cách học:
1. Mở file này trong trình soạn thảo (VS Code / Antigravity).
2. Đọc hướng dẫn ở từng hàm `bai_1` đến `bai_5` và tự gõ code vào phần `# TODO`.
3. Chạy lệnh kiểm tra điểm số tại Terminal:
   d:\MachineLearning\FinalProject\work\.venv\Scripts\python.exe d:\MachineLearning\OnTapThucHanh\luyen_code_nb_cham_tu_dong.py
4. Nếu bí ở bài nào, bạn có thể xem gợi ý hoặc nhắn lên khung chat để được hướng dẫn từng dòng!
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ==============================================================================
# BÀI 1 (2.0 điểm): Tách X, y và chia Train/Test thủ công theo tỷ lệ (Pandas)
# (Giống hệt cách chia dữ liệu trong NaiveBayes_Multinomial.ipynb của thầy)
# ==============================================================================
def bai_1_chia_du_lieu(df: pd.DataFrame, target_col: str, train_ratio: float = 0.7):
    """
    Yêu cầu:
    1. Tính `train_size = round(len(df) * train_ratio)`.
    2. Lấy `train_size` dòng đầu tiên của `df` để tạo:
       - `X_train`: DataFrame đã bỏ cột `target_col` (dùng `.drop(columns=[target_col])`)
       - `y_train`: Series cột `target_col`
    3. Lấy các dòng còn lại từ `train_size` trở đi để tạo:
       - `X_test`: DataFrame đã bỏ cột `target_col` và `.reset_index(drop=True)`
       - `y_test`: Series cột `target_col` và `.reset_index(drop=True)`
    4. Trả về tuple: `(X_train, y_train, X_test, y_test)`
    """
    # TODO: Tự gõ code của bạn dưới đây (khoảng 5 dòng)
    return None, None, None, None


# ==============================================================================
# BÀI 2 (2.0 điểm): Phân loại dữ liệu số liên tục bằng GaussianNB (Iris / Cancer)
# ==============================================================================
def bai_2_gaussian_nb(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    """
    Yêu cầu:
    1. Khởi tạo mô hình `model = GaussianNB()`.
    2. Huấn luyện mô hình trên `X_train, y_train` bằng `.fit(...)`.
    3. Dự đoán nhãn `y_pred` trên `X_test` bằng `.predict(...)`.
    4. Tính độ chính xác `acc = accuracy_score(y_test, y_pred)`.
    5. Trả về `(y_pred, acc)`.
    """
    # TODO: Tự gõ code của bạn dưới đây (khoảng 4 dòng)
    return None, None


# ==============================================================================
# BÀI 3 (2.0 điểm): Dữ liệu danh mục chữ -> OneHotEncoder + MultinomialNB
# (Giống Bài Đơn hàng ẩm thực & Bài Động đất Nepal 8 cột chữ)
# ==============================================================================
def bai_3_multinomial_ohe(X_train_df: pd.DataFrame, y_train: np.ndarray, X_test_df: pd.DataFrame):
    """
    Yêu cầu:
    1. Khởi tạo `ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')`.
    2. Biến đổi tập Train: `X_tr_ohe = ohe.fit_transform(X_train_df)`.
    3. Biến đổi tập Test:  `X_te_ohe = ohe.transform(X_test_df)` (CHỈ dùng transform!).
    4. Khởi tạo `clf = MultinomialNB(alpha=1.0)`, gọi `.fit(X_tr_ohe, y_train)`.
    5. Trả về mảng nhãn dự đoán `clf.predict(X_te_ohe)`.
    """
    # TODO: Tự gõ code của bạn dưới đây (khoảng 6 dòng)
    return None


# ==============================================================================
# BÀI 4 (2.0 điểm): Dữ liệu nhị phân 0/1 -> BernoulliNB & Xác suất predict_proba
# (Giống Bài Triệu chứng bệnh & Văn bản Bắc - Nam)
# ==============================================================================
def bai_4_bernoulli_nb(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray):
    """
    Yêu cầu:
    1. Khởi tạo `clf = BernoulliNB(alpha=1.0)`.
    2. Huấn luyện trên `X_train, y_train`.
    3. Tính nhãn dự đoán `y_pred = clf.predict(X_test)`.
    4. Tính ma trận xác suất `y_proba = clf.predict_proba(X_test)`.
    5. Trả về `(y_pred, y_proba)`.
    """
    # TODO: Tự gõ code của bạn dưới đây (khoảng 4 dòng)
    return None, None


# ==============================================================================
# BÀI 5 (2.0 điểm): Tự cài đặt công thức Laplace Smoothing (From Scratch)
# (Giống hệt hàm Pxik_feature_per_class trong NaiveBayes_Multinomial.ipynb của thầy)
# ==============================================================================
def bai_5_laplace_smoothing(X_col_k, xi_val, num_nomials_M: int, alpha: float = 1.0) -> float:
    """
    Tính xác suất có điều kiện P(x_i = xi_val | y = k) với làm trơn Laplace:
        P = (count + alpha) / (N_k + num_nomials_M * alpha)
    Trong đó:
    - `X_col_k`: mảng hoặc Series chứa giá trị cột thứ i của các mẫu thuộc lớp k.
    - `xi_val`: giá trị cần đếm.
    - `count`: số phần tử trong `X_col_k` bằng với `xi_val` (gợi ý: `(np.asarray(X_col_k) == xi_val).sum()`).
    - `N_k`: tổng số phần tử của `X_col_k` (gợi ý: `len(X_col_k)`).
    """
    # TODO: Tự gõ code của bạn dưới đây (khoảng 3-4 dòng)
    return None


# ==============================================================================
# ĐÁP ÁN MẪU (Chỉ dùng để đối chiếu khi chạy với cờ --dap-an)
# ==============================================================================
def _ref_bai_1(df, target_col, train_ratio=0.7):
    train_size = round(len(df) * train_ratio)
    X_train = df.iloc[:train_size].drop(columns=[target_col])
    y_train = df[target_col].iloc[:train_size]
    X_test = df.iloc[train_size:].drop(columns=[target_col]).reset_index(drop=True)
    y_test = df[target_col].iloc[train_size:].reset_index(drop=True)
    return X_train, y_train, X_test, y_test


def _ref_bai_2(X_train, y_train, X_test, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return y_pred, acc


def _ref_bai_3(X_train_df, y_train, X_test_df):
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_tr_ohe = ohe.fit_transform(X_train_df)
    X_te_ohe = ohe.transform(X_test_df)
    clf = MultinomialNB(alpha=1.0)
    clf.fit(X_tr_ohe, y_train)
    return clf.predict(X_te_ohe)


def _ref_bai_4(X_train, y_train, X_test):
    clf = BernoulliNB(alpha=1.0)
    clf.fit(X_train, y_train)
    return clf.predict(X_test), clf.predict_proba(X_test)


def _ref_bai_5(X_col_k, xi_val, num_nomials_M, alpha=1.0):
    arr = np.asarray(X_col_k)
    count = (arr == xi_val).sum()
    N_k = len(arr)
    return float((count + alpha) / (N_k + num_nomials_M * alpha))


# ==============================================================================
# BỘ CHẤM ĐIỂM TỰ ĐỘNG (AUTOGRADER)
# ==============================================================================
def run_autograder(use_ref: bool = False):
    print("=" * 72)
    print(" HỆ THỐNG CHẤM ĐIỂM TỰ ĐỘNG — BÀI KIỂM TRA NAIVE BAYES (THANG 10.0)")
    print("=" * 72)
    score = 0.0
    rng = np.random.default_rng(42)

    f1 = _ref_bai_1 if use_ref else bai_1_chia_du_lieu
    f2 = _ref_bai_2 if use_ref else bai_2_gaussian_nb
    f3 = _ref_bai_3 if use_ref else bai_3_multinomial_ohe
    f4 = _ref_bai_4 if use_ref else bai_4_bernoulli_nb
    f5 = _ref_bai_5 if use_ref else bai_5_laplace_smoothing

    # --- Test Bài 1 ---
    try:
        df_dummy = pd.DataFrame({
            "f1": np.arange(10),
            "f2": np.arange(10, 20),
            "cuisine": ["A", "B"] * 5,
        })
        X_tr, y_tr, X_te, y_te = f1(df_dummy, "cuisine", 0.7)
        assert X_tr is not None and X_te is not None, "Hàm đang trả về None (bạn chưa viết code)."
        assert len(X_tr) == 7 and len(X_te) == 3, f"Kích thước sai: cần 7 train, 3 test nhưng nhận {len(X_tr)}, {len(X_te)}"
        assert "cuisine" not in X_tr.columns and "cuisine" not in X_te.columns, "Chưa xóa cột nhãn 'cuisine' khỏi X_train/X_test!"
        assert list(X_te.index) == [0, 1, 2], "Quên .reset_index(drop=True) trên X_test!"
        assert list(y_te.index) == [0, 1, 2], "Quên .reset_index(drop=True) trên y_test!"
        print("[Bài 1] Chia dữ liệu Pandas (.iloc, .drop, .reset_index): ✅ PASS (+2.0đ)")
        score += 2.0
    except Exception as e:
        print(f"[Bài 1] Chia dữ liệu Pandas: ❌ CHƯA ĐẠT -> {e}")

    # --- Test Bài 2 ---
    try:
        X_g = np.vstack([rng.normal(2.0, 0.5, (40, 4)), rng.normal(6.0, 0.5, (40, 4))])
        y_g = np.array([0] * 40 + [1] * 40)
        X_g_te = np.vstack([rng.normal(2.0, 0.5, (10, 4)), rng.normal(6.0, 0.5, (10, 4))])
        y_g_te = np.array([0] * 10 + [1] * 10)
        pred_g, acc_g = f2(X_g, y_g, X_g_te, y_g_te)
        ref_pred_g, ref_acc_g = _ref_bai_2(X_g, y_g, X_g_te, y_g_te)
        assert pred_g is not None, "Hàm đang trả về None (bạn chưa viết code)."
        assert np.array_equal(pred_g, ref_pred_g), "Mảng dự đoán y_pred không khớp với GaussianNB chuẩn."
        assert np.isclose(acc_g, ref_acc_g), f"Accuracy sai: kỳ vọng {ref_acc_g}, nhận {acc_g}"
        print("[Bài 2] Huấn luyện & Đánh giá GaussianNB (Sklearn):       ✅ PASS (+2.0đ)")
        score += 2.0
    except Exception as e:
        print(f"[Bài 2] Huấn luyện & Đánh giá GaussianNB:       ❌ CHƯA ĐẠT -> {e}")

    # --- Test Bài 3 ---
    try:
        X_tr_cat = pd.DataFrame({"city": ["HN", "HN", "SG", "DN", "SG"], "device": ["ios", "web", "ios", "web", "ios"]})
        y_tr_cat = np.array(["B", "B", "N", "N", "N"])
        X_te_cat = pd.DataFrame({"city": ["HN", "LA_CHUA_GAP"], "device": ["ios", "web"]})
        pred_mnb = f3(X_tr_cat, y_tr_cat, X_te_cat)
        ref_mnb = _ref_bai_3(X_tr_cat, y_tr_cat, X_te_cat)
        assert pred_mnb is not None, "Hàm đang trả về None (bạn chưa viết code)."
        assert np.array_equal(pred_mnb, ref_mnb), f"Kết quả dự đoán sai: kỳ vọng {ref_mnb}, nhận {pred_mnb}"
        print("[Bài 3] OneHotEncoder + MultinomialNB (Dữ liệu chữ):      ✅ PASS (+2.0đ)")
        score += 2.0
    except Exception as e:
        print(f"[Bài 3] OneHotEncoder + MultinomialNB:          ❌ CHƯA ĐẠT -> {e}")

    # --- Test Bài 4 ---
    try:
        X_tr_b = np.array([
            [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 1, 1],
        ])
        y_tr_b = np.array(["B", "B", "B", "N"])
        X_te_b = np.array([
            [1, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 1],
        ])
        pred_b, proba_b = f4(X_tr_b, y_tr_b, X_te_b)
        ref_pred_b, ref_proba_b = _ref_bai_4(X_tr_b, y_tr_b, X_te_b)
        assert pred_b is not None and proba_b is not None, "Hàm đang trả về None (bạn chưa viết code)."
        assert np.array_equal(pred_b, ref_pred_b), f"Nhãn dự đoán sai: kỳ vọng {ref_pred_b}, nhận {pred_b}"
        assert np.allclose(proba_b, ref_proba_b), "Xác suất predict_proba không khớp."
        print("[Bài 4] BernoulliNB & predict_proba (Văn bản/Triệu chứng):✅ PASS (+2.0đ)")
        score += 2.0
    except Exception as e:
        print(f"[Bài 4] BernoulliNB & predict_proba:            ❌ CHƯA ĐẠT -> {e}")

    # --- Test Bài 5 ---
    try:
        col_k = pd.Series(["ios", "web", "ios", "ios", "android"])
        p_ios = f5(col_k, "ios", num_nomials_M=5, alpha=1.0)
        p_unseen = f5(col_k, "windows", num_nomials_M=5, alpha=1.0)
        assert p_ios is not None, "Hàm đang trả về None (bạn chưa viết code)."
        assert np.isclose(p_ios, (3 + 1.0) / (5 + 5 * 1.0)), f"Sai xác suất cho 'ios': kỳ vọng 0.4, nhận {p_ios}"
        assert np.isclose(p_unseen, (0 + 1.0) / (5 + 5 * 1.0)), f"Sai xác suất Laplace khi count=0: kỳ vọng 0.1, nhận {p_unseen}"
        print("[Bài 5] Tự cài công thức Laplace Smoothing (From Scratch):✅ PASS (+2.0đ)")
        score += 2.0
    except Exception as e:
        print(f"[Bài 5] Tự cài công thức Laplace Smoothing:     ❌ CHƯA ĐẠT -> {e}")

    print("-" * 72)
    print(f" TỔNG ĐIỂM HIỆN TẠI: {score:.1f} / 10.0 điểm")
    print("=" * 72)


if __name__ == "__main__":
    use_ref_flag = "--dap-an" in sys.argv
    run_autograder(use_ref=use_ref_flag)
