"""
Script Tổng Hợp Giải Toàn Bộ Bài Tập Thực Hành Tuần 2 (Naive Bayes)
Họ và tên: Nguyễn Hải Anh - MSSV: 24001968
Môn học: Machine Learning - Lab 01 / Tuần 2: Phương pháp Naive Bayes
"""

import sys
import time
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from bai1_iris_gaussian_nb import run_bai1
from bai2_cancer_gaussian_nb import run_bai2
from bai3_text_b_n_nb import run_bai3
from bai4_orders_multinomial_nb import run_bai4
from bai5_symptoms_bernoulli_nb import run_bai5
from bai6_spam_ex6_nb import run_bai6

def main():
    start_time = time.time()
    print("#" * 80)
    print("# BÀI BÁO CÁO THỰC HÀNH MACHINE LEARNING - TUẦN 2 (NAIVE BAYES)")
    print("# Sinh viên thực hiện: NGUYỄN HẢI ANH | MSSV: 24001968")
    print("#" * 80)
    print()

    # 1. Chạy Bài 1 (Iris 4 features)
    res1 = run_bai1()
    print("\n" + "-" * 80 + "\n")

    # 2. Chạy Bài 2 (Breast Cancer Wisconsin)
    res2 = run_bai2()
    print("\n" + "-" * 80 + "\n")

    # 3. Chạy Bài 3 (Text Bắc/Nam)
    res3 = run_bai3()
    print("\n" + "-" * 80 + "\n")

    # 4. Chạy Bài 4 (Online Orders Cuisine)
    res4 = run_bai4()
    print("\n" + "-" * 80 + "\n")

    # 5. Chạy Bài 5 (Clinical Symptoms)
    res5 = run_bai5()
    print("\n" + "-" * 80 + "\n")

    # 6. Chạy Bài 6 (Spam Ling-Spam ex6DataPrepared)
    res6 = run_bai6()
    print("\n" + "-" * 80 + "\n")

    # Bảng tổng kết kết quả toàn bộ các bài tập
    elapsed = time.time() - start_time
    print("=" * 80)
    print(f" TỔNG KẾT KẾT QUẢ THỰC NGHIỆM TUẦN 2 (Thời gian chạy: {elapsed:.2f}s)")
    print("=" * 80)
    summary_data = [
        {
            'Bài tập': 'Bài 1: Hoa Iris (4 thuộc tính)',
            'Mô hình': 'Gaussian NB (Scratch & Sklearn)',
            'Dữ liệu': 'Iris (150 mẫu, 3 lớp)',
            'Train / Test': '120 / 30 (4:1)',
            'Accuracy': f"{res1['acc_sk']*100:.2f}%",
            'F1-Score': f"{res1['f1_macro']*100:.2f}% (Macro)"
        },
        {
            'Bài tập': 'Bài 2: Ung thư vú Wisconsin',
            'Mô hình': 'Gaussian NB (Scratch & Sklearn)',
            'Dữ liệu': 'Cancer (699 mẫu, 2 lớp)',
            'Train / Test': '579 / 120 (80 lành, 40 ác)',
            'Accuracy': f"{res2['accuracy']*100:.2f}%",
            'F1-Score': f"{res2['f1_malignant']*100:.2f}% (Ác tính)"
        },
        {
            'Bài tập': 'Bài 3: Văn bản ẩm thực Bắc/Nam',
            'Mô hình': 'Multinomial & Bernoulli NB',
            'Dữ liệu': 'Văn bản (4 train, 2 test)',
            'Train / Test': '4 / 2',
            'Accuracy': '100% (Khớp lý thuyết)',
            'F1-Score': 'D5->B, D6->N'
        },
        {
            'Bài tập': 'Bài 4: Phân loại Email Spam Ling-Spam',
            'Mô hình': 'Multinomial NB (Tần suất từ)',
            'Dữ liệu': 'ex6DataPrepared (960 mẫu, 2 lớp)',
            'Train / Test': '700 / 260',
            'Accuracy': f"{res6['acc_mnb']*100:.2f}%",
            'F1-Score': f"{res6['f1_mnb']*100:.2f}% (Spam)"
        },
        {
            'Bài tập': 'Bài 5: Đơn hàng ẩm thực trực tuyến',
            'Mô hình': 'Multinomial NB (One-Hot)',
            'Dữ liệu': 'Orders (3000 mẫu, 10 lớp)',
            'Train / Test': '2100 / 900 (7:3)',
            'Accuracy': f"{res4['accuracy_mnb']*100:.2f}%",
            'F1-Score': f"{res4['f1_macro']*100:.2f}% (Macro)"
        },
        {
            'Bài tập': 'Bài 6: Chẩn đoán triệu chứng bệnh',
            'Mô hình': 'Bernoulli NB',
            'Dữ liệu': 'Symptoms (2500 mẫu, 8 lớp)',
            'Train / Test': '1500 / 1000 (6:4)',
            'Accuracy': f"{res5['accuracy']*100:.2f}%",
            'F1-Score': f"{res5['f1_macro']*100:.2f}% (Macro)"
        },
    ]
    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))
    print("=" * 80)
    print("[*] Toàn bộ mã nguồn và biểu đồ đã được hoàn thiện tại: d:\\MachineLearning\\Week2\\work\\")

if __name__ == "__main__":
    main()
