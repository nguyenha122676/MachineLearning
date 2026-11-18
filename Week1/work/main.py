# Sinh viên thực hiện: Nguyễn Hải Anh - MSSV: 24001968
"""
MENU ĐIỀU KHIỂN CHẠY TẤT CẢ VÍ DỤ VÀ BÀI TẬP (BÀI THỰC HÀNH SỐ 2)
Bạn chỉ cần chạy file này bằng lệnh:
    python main.py
sau đó gõ số thứ tự của bài muốn chạy và nhấn Enter!
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import subprocess

SCRIPTS = {
    "1": ("Ví dụ A.1: K-NN dự đoán GPA từ điểm SAT (SAT_GPA.csv)", "vidu_A1.py"),
    "2": ("Ví dụ A.2: K-NN dự đoán Áp suất theo Nhiệt độ sôi (17 mẫu)", "vidu_A2.py"),
    "3": ("Ví dụ A.3: K-NN đa chiều dự đoán tiêu thụ Xăng (fuel.txt)", "vidu_A3.py"),
    "4": ("Ví dụ B.1: Hồi quy đơn biến Nhiệt độ & Áp suất (Giải tích & pinv)", "vidu_B1.py"),
    "5": ("Ví dụ B.2: Hồi quy đa biến tiêu thụ Xăng (QR Householder & Scikit-Learn)", "vidu_B2.py"),
    "6": ("Bài tập B.1: Đánh giá Hồi quy Xăng trên cả Train & Validation (fuel.txt)", "bai_tap_B1.py"),
    "7": ("Bài tập B.3: Hồi quy SAT vs GPA, tương quan Pearson, so sánh K-NN", "bai_tap_B3.py"),
    "8": ("Bài tập B.4: Hồi quy độ dày mạch máu NTM (100 mẫu & 80/20 train/test)", "bai_tap_B4.py"),
    "9": ("Bài tập B.5: Hồi quy & K-NN dự đoán giá nhà BĐS (real_estate.csv)", "bai_tap_B5.py"),
    "0": ("Chạy TẤT CẢ các ví dụ và bài tập liên tục", "ALL"),
}

def print_menu():
    print("\n" + "="*70)
    print("      CHƯƠNG TRÌNH CHẠY BÀI THỰC HÀNH SỐ 2: K-NN & HỒI QUY TUYẾN TÍNH")
    print("="*70)
    for key, (desc, _) in SCRIPTS.items():
        if key == "0":
            print("-" * 70)
        print(f" [{key}] {desc}")
    print(" [q] Thoát chương trình")
    print("="*70)

def run_script(script_name):
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
    print(f"\n>>> ĐANG CHẠY: {script_name} ...\n")
    res = subprocess.run([sys.executable, script_path])
    print(f"\n>>> KẾT THÚC {script_name} (Mã trả về: {res.returncode})\n")

def main():
    while True:
        print_menu()
        choice = input("Nhập số bài cần chạy (0-9 hoặc q để thoát): ").strip().lower()
        if choice in ['q', 'exit', 'quit']:
            print("Đã thoát chương trình.")
            break
        elif choice == "0":
            for key in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                run_script(SCRIPTS[key][1])
                input("\nNhấn Enter để chuyển sang bài tiếp theo...")
        elif choice in SCRIPTS:
            run_script(SCRIPTS[choice][1])
            input("\nNhấn Enter để quay lại menu...")
        else:
            print("Lựa chọn không hợp lệ! Vui lòng chọn số từ 0 đến 9 hoặc q.")

if __name__ == "__main__":
    main()
