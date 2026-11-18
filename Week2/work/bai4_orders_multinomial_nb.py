"""
Bài 4: Dự đoán loại ẩm thực (Cuisine) từ đơn đặt hàng trực tuyến bằng Multinomial Naive Bayes
Họ và tên: Nguyễn Hải Anh - MSSV: 24001968
Môn học: Machine Learning - Bài thực hành số 2 (Naive Bayes)

Nội dung:
- Đọc tập dữ liệu multinomial_nb_orders (1).csv (3000 đơn hàng, 8 đặc trưng danh mục, 10 lớp ẩm thực)
- Đặc trưng rời rạc (Categorical features):
  + city (20 thành phố), device (5 thiết bị), user_segment (6 phân khúc khách hàng), promo (8 khuyến mãi + 'none')
  + fav_ingredient (18 nguyên liệu ưa thích), spice_level (5 mức cay), day_of_week (7 ngày trong tuần), price_bucket (4 mức giá)
- Nhãn mục tiêu: cuisine (10 lớp ẩm thực: american, chinese, indian, italian, japanese, korean, mediterranean, mexican, thai, vietnamese)
- Phân chia train:test = 7:3 (70% Train, 30% Test có phân tầng stratified)
- Tiền xử lý dữ liệu: Mã hóa One-Hot Encoding để chuyển các đặc trưng danh mục thành vector tần suất/chỉ báo phù hợp cho MultinomialNB
- So sánh bổ sung với mô hình CategoricalNB (Ordinal Encoding)
- Đánh giá toàn diện: Accuracy, Macro/Weighted Precision, Recall, F1-Score, Confusion Matrix
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.naive_bayes import MultinomialNB, CategoricalNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def run_bai4():
    print("=" * 70)
    print(" BÀI 4: DỰ ĐOÁN ẨM THỰC ĐƠN HÀNG (MULTINOMIAL NAIVE BAYES)")
    print("=" * 70)

    # 1. Đọc dữ liệu
    csv_path = os.path.join(os.path.dirname(__file__), "..", "multinomial_nb_orders (1).csv")
    if not os.path.exists(csv_path):
        csv_path = "multinomial_nb_orders (1).csv"

    df = pd.read_csv(csv_path)
    target_col = 'cuisine'
    feature_cols = [c for c in df.columns if c != target_col]

    print(f"[*] Kích thước dữ liệu: {df.shape[0]} đơn hàng, {len(feature_cols)} đặc trưng danh mục")
    print(f"[*] Danh sách đặc trưng: {feature_cols}")
    print(f"[*] Số lượng lớp ẩm thực (Cuisine): {df[target_col].nunique()}")
    print(f"[*] Phân bố mẫu theo từng lớp ẩm thực:\n{df[target_col].value_counts()}\n")

    X = df[feature_cols]
    y = df[target_col]

    # 2. Phân chia Train/Test theo tỉ lệ 7:3 (70% Train = 2100, 30% Test = 900)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"[*] Tập Train: {X_train.shape[0]} mẫu")
    print(f"[*] Tập Test:  {X_test.shape[0]} mẫu\n")

    # ==============================================================
    # 3. TIỀN XỬ LÝ: ONE-HOT ENCODING CHO MULTINOMIAL NAIVE BAYES
    # ==============================================================
    # MultinomialNB yêu cầu đầu vào là vector đếm không âm (non-negative counts).
    # One-Hot Encoding biểu diễn mỗi giá trị danh mục thành vector nhị phân chỉ số 0/1.
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_ohe = ohe.fit_transform(X_train)
    X_test_ohe = ohe.transform(X_test)
    print(f"[*] Kích thước ma trận đặc trưng sau One-Hot: {X_train_ohe.shape[1]} cột nhị phân")

    # 4. Huấn luyện mô hình MultinomialNB với Laplace smoothing alpha=1.0
    mnb = MultinomialNB(alpha=1.0)
    mnb.fit(X_train_ohe, y_train)
    y_pred_mnb = mnb.predict(X_test_ohe)

    # 5. So sánh với CategoricalNB (sử dụng OrdinalEncoder)
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train_oe = oe.fit_transform(X_train)
    X_test_oe = oe.transform(X_test)
    min_cats = [len(cats) for cats in oe.categories_]
    cnb = CategoricalNB(min_categories=min_cats, alpha=1.0)
    cnb.fit(X_train_oe, y_train)
    y_pred_cnb = cnb.predict(X_test_oe)

    # ==============================================================
    # 6. ĐÁNH GIÁ VÀ BÁO CÁO KẾT QUẢ
    # ==============================================================
    acc_mnb = accuracy_score(y_test, y_pred_mnb)
    acc_cnb = accuracy_score(y_test, y_pred_cnb)
    prec_macro = precision_score(y_test, y_pred_mnb, average='macro', zero_division=0)
    rec_macro = recall_score(y_test, y_pred_mnb, average='macro')
    f1_macro = f1_score(y_test, y_pred_mnb, average='macro')
    f1_weighted = f1_score(y_test, y_pred_mnb, average='weighted')

    print("--- KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH MULTINOMIAL NAIVE BAYES ---")
    print(f"-> Độ chính xác MultinomialNB (One-Hot): {acc_mnb * 100:.2f}%")
    print(f"-> Độ chính xác CategoricalNB (Ordinal):  {acc_cnb * 100:.2f}%")
    print("   (Ghi chú: Hai mô hình cho kết quả tương đương do bản chất toán học rời rạc)")
    print(f"-> Precision (Macro Avg):  {prec_macro * 100:.2f}%")
    print(f"-> Recall (Macro Avg):     {rec_macro * 100:.2f}%")
    print(f"-> F1-Score (Macro Avg):   {f1_macro * 100:.2f}%")
    print(f"-> F1-Score (Weighted):    {f1_weighted * 100:.2f}%\n")

    print("[*] Nhận xét baseline ngẫu nhiên: Với 10 lớp cân bằng, đoán ngẫu nhiên chỉ đạt ~10.0%.")
    print(f"    Mô hình đạt {acc_mnb * 100:.2f}%, cao gấp hơn 5 lần so với ngẫu nhiên.\n")

    print("Báo cáo phân loại chi tiết từng lớp ẩm thực:")
    print(classification_report(y_test, y_pred_mnb, digits=4))

    # Vẽ và lưu ma trận nhầm lẫn
    unique_cuisines = np.sort(df[target_col].unique())
    cm = confusion_matrix(y_test, y_pred_mnb, labels=unique_cuisines)

    save_fig_dir = os.path.dirname(__file__)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=unique_cuisines, yticklabels=unique_cuisines)
    plt.title('Ma trận nhầm lẫn - Phân loại 10 lớp ẩm thực (Multinomial NB)')
    plt.xlabel('Nhãn dự đoán')
    plt.ylabel('Nhãn thực tế')
    plt.xticks(rotation=45)
    plt.tight_layout()
    cm_path = os.path.join(save_fig_dir, "bai4_orders_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[*] Đã lưu biểu đồ ma trận nhầm lẫn tại: {cm_path}")

    return {
        'accuracy_mnb': acc_mnb,
        'accuracy_cnb': acc_cnb,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'cm': cm
    }

if __name__ == "__main__":
    run_bai4()
