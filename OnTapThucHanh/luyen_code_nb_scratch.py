"""
BÀI LUYỆN CODE TỰ CÀI NAIVE BAYES TỪ ĐẦU (FROM SCRATCH) — KHÔNG DÙNG SKLEARN
=============================================================================
Mỗi bài tương ứng với 1 phần kiến thức cốt lõi để tự cài đặt hoàn chỉnh:
- Bài 1: Tự chia Train/Test bằng NumPy (permutation)
- Bài 2: Tự cài Gaussian Naive Bayes thuần NumPy (học means, vars, priors & dự đoán)
- Bài 3: Tự cài Bernoulli Naive Bayes thuần NumPy (cột 0/1 + Laplace)
- Bài 4: Tự cài Multinomial Naive Bayes cho văn bản đếm từ (Laplace)
- Bài 5: Tự tính Accuracy, Precision, Recall, F1 bằng NumPy

Chạy kiểm tra bằng lệnh:
  d:\MachineLearning\FinalProject\work\.venv\Scripts\python.exe d:\MachineLearning\OnTapThucHanh\luyen_code_nb_scratch.py
Chạy xem đáp án chuẩn chấm 10/10:
  d:\MachineLearning\FinalProject\work\.venv\Scripts\python.exe d:\MachineLearning\OnTapThucHanh\luyen_code_nb_scratch.py --dap-an
"""

import sys
import numpy as np

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ==============================================================================
# BÀI 1 (2.0 điểm): Tự chia Train / Test bằng NumPy (không dùng train_test_split)
# ==============================================================================
def bai_1_chia_train_test_numpy(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8, seed: int = 42):
    """
    Yêu cầu:
    1. Gọi `np.random.seed(seed)`.
    2. Tạo mảng chỉ số xáo trộn `idx = np.random.permutation(len(X))`.
    3. Tính `split = int(train_ratio * len(X))`.
    4. Cắt trả về `(X_train, X_test, y_train, y_test)` theo `idx[:split]` và `idx[split:]`.
    """
    # TODO: Tự gõ code của bạn dưới đây
    return None, None, None, None


# ==============================================================================
# BÀI 2 (2.0 điểm): Tự cài Gaussian Naive Bayes thuần NumPy (Iris / Cancer)
# ==============================================================================
def bai_2_gaussian_nb_scratch(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray):
    """
    Yêu cầu:
    1. Lấy danh sách lớp `classes = np.unique(y_train)`.
    2. Với mỗi lớp `c` (lọc `X_c = X_train[y_train == c]`):
       - `mean_c = np.mean(X_c, axis=0)`
       - `var_c  = np.var(X_c, axis=0) + 1e-9`
       - `prior_c = len(X_c) / len(X_train)`
    3. Với mỗi mẫu `x` trong `X_test`, tính điểm từng lớp:
       - `log_pdf = -0.5 * np.log(2 * np.pi * var_c) - ((x - mean_c) ** 2) / (2 * var_c)`
       - `score = np.log(prior_c) + np.sum(log_pdf)`
       Chọn nhãn `classes[np.argmax(scores)]` và trả về mảng `np.array(y_pred)`.
    """
    # TODO: Tự gõ code của bạn dưới đây
    return None


# ==============================================================================
# BÀI 3 (2.0 điểm): Tự cài Bernoulli Naive Bayes thuần NumPy (cột nhị phân 0/1)
# ==============================================================================
def bai_3_bernoulli_nb_scratch(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, alpha: float = 1.0):
    """
    Yêu cầu:
    1. Lấy `classes = np.unique(y_train)`.
    2. Với mỗi lớp `c` (`X_c = X_train[y_train == c]`, `N_c = len(X_c)`):
       - `prior_c = N_c / len(X_train)`
       - `p_c = (np.sum(X_c == 1, axis=0) + alpha) / (N_c + 2.0 * alpha)`
    3. Với mỗi mẫu `x` trong `X_test`, tính điểm từng lớp:
       - `prob = x * p_c + (1 - x) * (1 - p_c)`
       - `score = np.log(prior_c) + np.sum(np.log(np.clip(prob, 1e-12, 1.0)))`
       Trả về mảng `np.array(y_pred)` gồm `classes[np.argmax(scores)]`.
    """
    # TODO: Tự gõ code của bạn dưới đây
    return None


# ==============================================================================
# BÀI 4 (2.0 điểm): Tự cài Multinomial Naive Bayes cho văn bản đếm tần suất từ
# ==============================================================================
def bai_4_multinomial_text_scratch(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, alpha: float = 1.0):
    """
    Yêu cầu:
    1. `classes = np.unique(y_train)`, số từ trong từ điển `d = X_train.shape[1]`.
    2. Với mỗi lớp `c` (`X_c = X_train[y_train == c]`):
       - `prior_c = len(X_c) / len(X_train)`
       - `word_counts_c = np.sum(X_c, axis=0)`
       - `total_words_c = np.sum(X_c)`
       - `p_c = (word_counts_c + alpha) / (total_words_c + alpha * d)`
    3. Với mỗi văn bản `x` trong `X_test`:
       - `score = np.log(prior_c) + np.sum(x * np.log(p_c))`
       Trả về mảng `np.array(y_pred)` gồm `classes[np.argmax(scores)]`.
    """
    # TODO: Tự gõ code của bạn dưới đây
    return None


# ==============================================================================
# BÀI 5 (2.0 điểm): Tự cài Accuracy, Precision, Recall, F1 (không dùng sklearn)
# ==============================================================================
def bai_5_metrics_scratch(y_true: np.ndarray, y_pred: np.ndarray, pos_label=1):
    """
    Yêu cầu:
    1. `acc = float(np.mean(y_pred == y_true))`
    2. `TP = np.sum((y_true == pos_label) & (y_pred == pos_label))`
       `FP = np.sum((y_true != pos_label) & (y_pred == pos_label))`
       `FN = np.sum((y_true == pos_label) & (y_pred != pos_label))`
    3. `precision = float(TP / (TP + FP))` nếu `TP + FP > 0` else `0.0`
       `recall    = float(TP / (TP + FN))` nếu `TP + FN > 0` else `0.0`
       `f1        = float(2 * precision * recall / (precision + recall))` nếu `precision + recall > 0` else `0.0`
    4. Trả về tuple `(acc, precision, recall, f1)`.
    """
    # TODO: Tự gõ code của bạn dưới đây
    return None, None, None, None


# ==============================================================================
# ĐÁP ÁN MẪU (Để đối chiếu với bộ test)
# ==============================================================================
def _ref_1(X, y, train_ratio=0.8, seed=42):
    np.random.seed(seed)
    idx = np.random.permutation(len(X))
    split = int(train_ratio * len(X))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]


def _ref_2(X_train, y_train, X_test):
    classes = np.unique(y_train)
    means = [np.mean(X_train[y_train == c], axis=0) for c in classes]
    vars_ = [np.var(X_train[y_train == c], axis=0) + 1e-9 for c in classes]
    priors = [np.mean(y_train == c) for c in classes]
    y_pred = []
    for x in X_test:
        scores = [
            np.log(priors[i]) + np.sum(-0.5 * np.log(2 * np.pi * vars_[i]) - ((x - means[i]) ** 2) / (2 * vars_[i]))
            for i in range(len(classes))
        ]
        y_pred.append(classes[np.argmax(scores)])
    return np.array(y_pred)


def _ref_3(X_train, y_train, X_test, alpha=1.0):
    classes = np.unique(y_train)
    priors = [np.mean(y_train == c) for c in classes]
    p_list = []
    for c in classes:
        X_c = X_train[y_train == c]
        p_list.append((np.sum(X_c == 1, axis=0) + alpha) / (len(X_c) + 2.0 * alpha))
    y_pred = []
    for x in X_test:
        scores = []
        for i in range(len(classes)):
            prob = x * p_list[i] + (1 - x) * (1 - p_list[i])
            scores.append(np.log(priors[i]) + np.sum(np.log(np.clip(prob, 1e-12, 1.0))))
        y_pred.append(classes[np.argmax(scores)])
    return np.array(y_pred)


def _ref_4(X_train, y_train, X_test, alpha=1.0):
    classes = np.unique(y_train)
    d = X_train.shape[1]
    priors = [np.mean(y_train == c) for c in classes]
    p_list = []
    for c in classes:
        X_c = X_train[y_train == c]
        p_list.append((np.sum(X_c, axis=0) + alpha) / (np.sum(X_c) + alpha * d))
    y_pred = []
    for x in X_test:
        scores = [np.log(priors[i]) + np.sum(x * np.log(p_list[i])) for i in range(len(classes))]
        y_pred.append(classes[np.argmax(scores)])
    return np.array(y_pred)


def _ref_5(y_true, y_pred, pos_label=1):
    acc = float(np.mean(y_pred == y_true))
    TP = np.sum((y_true == pos_label) & (y_pred == pos_label))
    FP = np.sum((y_true != pos_label) & (y_pred == pos_label))
    FN = np.sum((y_true == pos_label) & (y_pred != pos_label))
    precision = float(TP / (TP + FP)) if (TP + FP) > 0 else 0.0
    recall = float(TP / (TP + FN)) if (TP + FN) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return acc, precision, recall, f1


def run_autograder(use_ref: bool = False):
    print("=" * 72)
    print(" CHẤM ĐIỂM TỰ ĐỘNG — NAIVE BAYES TỰ CÀI TỪ ĐẦU (FROM SCRATCH)")
    print("=" * 72)
    score = 0.0
    rng = np.random.default_rng(7)

    f1 = _ref_1 if use_ref else bai_1_chia_train_test_numpy
    f2 = _ref_2 if use_ref else bai_2_gaussian_nb_scratch
    f3 = _ref_3 if use_ref else bai_3_bernoulli_nb_scratch
    f4 = _ref_4 if use_ref else bai_4_multinomial_text_scratch
    f5 = _ref_5 if use_ref else bai_5_metrics_scratch

    # --- Test 1 ---
    try:
        X0 = np.arange(20).reshape(10, 2)
        y0 = np.arange(10)
        out = f1(X0, y0, 0.8, 42)
        ref = _ref_1(X0, y0, 0.8, 42)
        assert out[0] is not None, "Hàm đang trả về None (chưa viết code)."
        assert all(np.array_equal(a, b) for a, b in zip(out, ref)), "Chia tập Train/Test chưa khớp."
        print("[Bài 1] Tự chia Train/Test bằng NumPy permutation:   ✅ PASS (+2.0đ)")
        score += 2.0
    except Exception as e:
        print(f"[Bài 1] Tự chia Train/Test bằng NumPy:               ❌ CHƯA ĐẠT -> {e}")

    # --- Test 2 ---
    try:
        Xg_tr = np.vstack([rng.normal(1.0, 0.4, (30, 3)), rng.normal(5.0, 0.4, (30, 3))])
        yg_tr = np.array(["A"] * 30 + ["B"] * 30)
        Xg_te = np.vstack([rng.normal(1.0, 0.4, (10, 3)), rng.normal(5.0, 0.4, (10, 3))])
        pred2 = f2(Xg_tr, yg_tr, Xg_te)
        ref2 = _ref_2(Xg_tr, yg_tr, Xg_te)
        assert pred2 is not None, "Hàm đang trả về None (chưa viết code)."
        assert np.array_equal(pred2, ref2), "Dự đoán Gaussian NB chưa khớp."
        print("[Bài 2] Tự cài Gaussian NB thuần NumPy:              ✅ PASS (+2.0đ)")
        score += 2.0
    except Exception as e:
        print(f"[Bài 2] Tự cài Gaussian NB thuần NumPy:              ❌ CHƯA ĐẠT -> {e}")

    # --- Test 3 ---
    try:
        Xb_tr = np.array([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 0]], dtype=float)
        yb_tr = np.array([0, 0, 1, 1])
        Xb_te = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=float)
        pred3 = f3(Xb_tr, yb_tr, Xb_te, 1.0)
        ref3 = _ref_3(Xb_tr, yb_tr, Xb_te, 1.0)
        assert pred3 is not None, "Hàm đang trả về None (chưa viết code)."
        assert np.array_equal(pred3, ref3), "Dự đoán Bernoulli NB chưa khớp."
        print("[Bài 3] Tự cài Bernoulli NB thuần NumPy:             ✅ PASS (+2.0đ)")
        score += 2.0
    except Exception as e:
        print(f"[Bài 3] Tự cài Bernoulli NB thuần NumPy:             ❌ CHƯA ĐẠT -> {e}")

    # --- Test 4 ---
    try:
        Xt_tr = np.array([
            [2, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 1, 1],
        ], dtype=float)
        yt_tr = np.array(["B", "B", "B", "N"])
        Xt_te = np.array([[2, 0, 0, 1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 1, 1]], dtype=float)
        pred4 = f4(Xt_tr, yt_tr, Xt_te, 1.0)
        ref4 = _ref_4(Xt_tr, yt_tr, Xt_te, 1.0)
        assert pred4 is not None, "Hàm đang trả về None (chưa viết code)."
        assert np.array_equal(pred4, ref4), "Dự đoán Multinomial NB văn bản chưa khớp."
        print("[Bài 4] Tự cài Multinomial NB văn bản thuần NumPy:   ✅ PASS (+2.0đ)")
        score += 2.0
    except Exception as e:
        print(f"[Bài 4] Tự cài Multinomial NB văn bản thuần NumPy:   ❌ CHƯA ĐẠT -> {e}")

    # --- Test 5 ---
    try:
        yt = np.array([4, 2, 4, 4, 2, 4])
        yp = np.array([4, 4, 4, 2, 2, 4])
        m5 = f5(yt, yp, pos_label=4)
        r5 = _ref_5(yt, yp, pos_label=4)
        assert m5[0] is not None, "Hàm đang trả về None (chưa viết code)."
        assert all(np.isclose(a, b) for a, b in zip(m5, r5)), f"Sai chỉ số: kỳ vọng {r5}, nhận {m5}"
        print("[Bài 5] Tự tính Accuracy/Precision/Recall/F1 NumPy:  ✅ PASS (+2.0đ)")
        score += 2.0
    except Exception as e:
        print(f"[Bài 5] Tự tính Accuracy/Precision/Recall/F1 NumPy:  ❌ CHƯA ĐẠT -> {e}")

    print("-" * 72)
    print(f" TỔNG ĐIỂM HIỆN TẠI: {score:.1f} / 10.0 điểm")
    print("=" * 72)


if __name__ == "__main__":
    run_autograder(use_ref="--dap-an" in sys.argv)
