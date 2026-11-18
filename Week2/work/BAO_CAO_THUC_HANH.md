# BÁO CÁO THỰC HÀNH MACHINE LEARNING: BÀI THỰC HÀNH SỐ 2 (LAB 01)
## Chủ đề: Họ thuật toán Phân loại Naïve Bayes (Gaussian, Multinomial, Bernoulli Naïve Bayes)

**Họ và tên sinh viên**: Nguyễn Hải Anh  
**Mã số sinh viên (MSSV)**: 24001968  
**Lớp / Môn học**: Machine Learning  
**Học kỳ**: Học kỳ 1 (2024 - 2025)

---

## MỤC LỤC
1. [Giới thiệu & Cơ sở lý thuyết họ thuật toán Naïve Bayes](#1-giới-thiệu--cơ-sở-lý-thuyết-họ-thuật-toán-naïve-bayes)
   - 1.1. Định lý Bayes và Giả thiết "Ngây thơ" (Naïve Assumption)
   - 1.2. Gaussian Naïve Bayes (Dành cho đặc trưng liên tục)
   - 1.3. Multinomial Naïve Bayes & Kỹ thuật làm mịn Laplace (Dành cho dữ liệu tần suất/rời rạc)
   - 1.4. Bernoulli Naïve Bayes (Dành cho dữ liệu nhị phân 0/1)
   - 1.5. Bảng so sánh 3 biến thể Naïve Bayes
   - 1.6. Các độ đo đánh giá mô hình phân loại (Evaluation Metrics)
2. [Cấu trúc mã nguồn & Dữ liệu thực nghiệm](#2-cấu-trúc-mã-nguồn--dữ-liệu-thực-nghiệm)
3. [Bài tập 1: Phân loại hoa Iris với 4 đặc trưng (Gaussian NB)](#3-bài-tập-1-phân-loại-hoa-iris-với-4-đặc-trưng-gaussian-nb)
4. [Bài tập 2: Phân loại ung thư vú Wisconsin (Gaussian NB - Bài tập tự giải)](#4-bài-tập-2-phân-loại-ung-thư-vú-wisconsin-gaussian-nb---bài-tập-tự-giải)
5. [Bài tập 3: Phân loại văn bản ẩm thực Bắc - Nam (Multinomial & Bernoulli NB)](#5-bài-tập-3-phân-loại-văn-bản-ẩm-thực-bắc---nam-multinomial--bernoulli-nb)
6. [Bài tập 4: Dự đoán loại ẩm thực đơn hàng trực tuyến (Multinomial NB)](#6-bài-tập-4-dự-đoán-loại-ẩm-thực-đơn-hàng-trực-tuyến-multinomial-nb)
7. [Bài tập 5: Chẩn đoán 8 bệnh lý từ 30 triệu chứng lâm sàng (Bernoulli NB)](#7-bài-tập-5-chẩn-đoán-8-bệnh-lý-từ-30-triệu-chứng-lâm-sàng-bernoulli-nb)
8. [Bảng tổng hợp kết quả thực nghiệm Tuần 2](#8-bảng-tổng-hợp-kết-quả-thực-nghiệm-tuần-2)
9. [Bộ câu hỏi vấn đáp & Trả lời bảo vệ thực hành (Oral Exam Q&A)](#9-bộ-câu-hỏi-vấn-đáp--trả-lời-bảo-vệ-thực-hành-oral-exam-qa)

---

## 1. Giới thiệu & Cơ sở lý thuyết họ thuật toán Naïve Bayes

### 1.1. Định lý Bayes và Giả thiết "Ngây thơ" (Naïve Assumption)
Trong bài toán phân loại học có giám sát, ta có vector quan sát đầu vào $\mathbf{x} = (x_1, x_2, \dots, x_d) \in \mathbb{R}^d$ và nhãn mục tiêu $y \in \{c_1, c_2, \dots, c_K\}$.

Theo Định lý xác suất Bayes:
$$P(y = c_k \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid y = c_k) P(y = c_k)}{P(\mathbf{x})} = \frac{P(x_1, x_2, \dots, x_d \mid y = c_k) P(y = c_k)}{\sum_{j=1}^K P(\mathbf{x} \mid y = c_j) P(y = c_j)}$$

Trong đó:
- $P(y = c_k)$: Xác suất tiên nghiệm (Prior probability) của lớp $c_k$.
- $P(\mathbf{x} \mid y = c_k)$: Độ hợp lý (Likelihood) của dữ liệu $\mathbf{x}$ khi biết thuộc lớp $c_k$.
- $P(y = c_k \mid \mathbf{x})$: Xác suất hậu nghiệm (Posterior probability).
- $P(\mathbf{x})$: Hằng số chuẩn hóa (Evidence), không phụ thuộc vào lớp $c_k$.

**Quy tắc quyết định MAP (Maximum A Posteriori):**
$$\hat{y} = \arg\max_{c_k} P(y = c_k \mid \mathbf{x}) = \arg\max_{c_k} P(y = c_k) P(\mathbf{x} \mid y = c_k)$$

**Giả thiết độc lập có điều kiện ("Naïve"):**
Để tránh bùng nổ số chiều khi ước lượng phân bố đồng thời $P(x_1, \dots, x_d \mid y)$, mô hình giả định rằng các đặc trưng $x_1, x_2, \dots, x_d$ độc lập với nhau khi biết trước nhãn $y$:
$$P(\mathbf{x} \mid y = c_k) = \prod_{j=1}^d P(x_j \mid y = c_k)$$

Do đó, hàm mục tiêu phân loại Naïve Bayes trở thành:
$$\hat{y} = \arg\max_{c_k} \left[ P(y = c_k) \prod_{j=1}^d P(x_j \mid y = c_k) \right]$$

Để tránh hiện tượng tràn số dưới số thực (numerical underflow) khi nhân liên tiếp nhiều xác suất nhỏ, ta chuyển sang không gian Log-Likelihood:
$$\hat{y} = \arg\max_{c_k} \left[ \ln P(y = c_k) + \sum_{j=1}^d \ln P(x_j \mid y = c_k) \right]$$

---

### 1.2. Gaussian Naïve Bayes & Công thức Bayes Hỗn Hợp (Hybrid Bayes Rule)

#### a. Vấn đề toán học: Tại sao dùng hàm mật độ xác suất (PDF) thay cho xác suất rời rạc?
Trong công thức Bayes nguyên bản cho các biến rời rạc:
$$P(Y = k \mid X = x) = \frac{P(Y = k) \cdot P(X = x \mid Y = k)}{P(X = x)}$$

Khi đặc trưng $X$ là biến ngẫu nhiên liên tục thuộc $\mathbb{R}$:
- Theo lý thuyết xác suất, xác suất tại đúng một điểm đơn lẻ bất kỳ luôn bằng 0: $P(X = x) = 0$ và $P(X = x \mid Y = k) = 0$.
- Nếu thay trực tiếp vào công thức cổ điển, biểu thức sẽ rơi vào dạng vô định $\frac{0}{0}$.

#### b. Chứng minh chặt chẽ qua xấp xỉ tích phân lân cận vô cùng bé $\epsilon$
Xét biến ngẫu nhiên liên tục $X$ rơi vào một lân cận đối xứng bán kính $\epsilon > 0$ quanh điểm $x$: $E_\epsilon = [x - \epsilon, x + \epsilon]$.
Theo định nghĩa của hàm mật độ xác suất $f(u)$, xác suất có điều kiện là:
$$P(x - \epsilon \le X \le x + \epsilon \mid Y = k) = \int_{x - \epsilon}^{x + \epsilon} f_{X \mid Y = k}(u) \, du$$

Áp dụng Định lý giá trị trung bình tích phân (Mean Value Theorem), với $\epsilon$ vô cùng bé:
$$\int_{x - \epsilon}^{x + \epsilon} f_{X \mid Y = k}(u) \, du \approx 2\epsilon \cdot f_{X \mid Y = k}(x)$$

Tương tự đối với xác suất toàn phần của $X$:
$$P(x - \epsilon \le X \le x + \epsilon) = \int_{x - \epsilon}^{x + \epsilon} f_X(u) \, du \approx 2\epsilon \cdot f_X(x)$$

Áp dụng Định lý Bayes cho biến cố $X \in [x - \epsilon, x + \epsilon]$:
$$P(Y = k \mid x - \epsilon \le X \le x + \epsilon) = \frac{P(Y = k) \cdot P(x - \epsilon \le X \le x + \epsilon \mid Y = k)}{P(x - \epsilon \le X \le x + \epsilon)} \approx \frac{P(Y = k) \cdot 2\epsilon f_{X \mid Y = k}(x)}{2\epsilon f_X(x)}$$

Lấy giới hạn khi $\epsilon \to 0$, đại lượng $2\epsilon$ ở cả tử số và mẫu số **triệt tiêu hoàn toàn**, ta thu được **Công thức Bayes Hỗn Hợp (Hybrid Bayes Rule)**:
$$\mathbf{P(Y = k \mid X = x) = \frac{P(Y = k) \cdot f_{X \mid Y = k}(x)}{f_X(x)}}$$

Trong đó $f_{X \mid Y=k}(x)$ là hàm mật độ xác suất của phân phối chuẩn (Gaussian PDF):
$$f_{X \mid Y=k}(x) = \frac{1}{\sqrt{2\pi \sigma_{k}^2}} \exp\left( -\frac{(x - \mu_{k})^2}{2\sigma_{k}^2} \right)$$

Do mẫu số $f_X(x) = \sum_{j=1}^K P(Y=j) f_{X \mid Y=j}(x)$ không đổi với mọi lớp $k$, quy tắc quyết định MAP đơn giản hóa thành:
$$\hat{y} = \arg\max_k \left[ P(Y = k) \cdot f_{X \mid Y = k}(x) \right]$$

#### c. Ước lượng tham số hợp lý cực đại (MLE) trên tập huấn luyện:
$$\mu_{k, j} = \frac{1}{N_k} \sum_{i: y^{(i)} = k} x_{i, j}, \quad \sigma_{k, j}^2 = \frac{1}{N_k} \sum_{i: y^{(i)} = k} (x_{i, j} - \mu_{k, j})^2, \quad \pi_k = P(Y = k) = \frac{N_k}{N}$$

*(Lưu ý kỹ thuật: Trong Scikit-Learn, tham số `var_smoothing` bổ sung $\epsilon \cdot \max(\sigma^2)$ vào phương sai để chống lỗi chia cho 0).*

---

### 1.3. Multinomial Naïve Bayes & Kỹ thuật làm mịn Laplace
Áp dụng cho dữ liệu đếm rời rạc (tần suất từ xuất hiện trong văn bản, số lượng lựa chọn danh mục).
Giả sử vector $\mathbf{x} = (x_1, \dots, x_d)$ biểu diễn tần suất xuất hiện của từ thứ $j$ trong từ điển:
$$P(\mathbf{x} \mid y = c_k) = \frac{(\sum_j x_j)!}{\prod_j x_j!} \prod_{j=1}^d p_{k, j}^{x_j}$$

Trong đó $p_{k, j} = P(w_j \mid y = c_k)$ là xác suất từ thứ $j$ xuất hiện trong lớp $c_k$.

**Kỹ thuật làm mịn Laplace (Laplace Additive Smoothing):**
Nếu một từ trong tập Test chưa từng xuất hiện trong lớp $c_k$ ở tập Train, $p_{k, j} = 0$, dẫn đến toàn bộ tích xác suất bằng 0 ("Zero Probability Problem"). Ta cộng thêm tham số làm mịn $\alpha \ge 0$ (thường chọn $\alpha = 1$):
$$p_{k, j} = \frac{N_{k, j} + \alpha}{N_k + \alpha \cdot |V|}$$
- $N_{k, j}$: Tổng số lần từ $w_j$ xuất hiện trong các văn bản thuộc lớp $c_k$.
- $N_k = \sum_j N_{k, j}$: Tổng số lượt từ trong toàn bộ lớp $c_k$.
- $|V|$: Kích thước từ điển vựng (số đặc trưng).

---

### 1.4. Bernoulli Naïve Bayes (Dành cho dữ liệu nhị phân 0/1)
Áp dụng khi vector đặc trưng $\mathbf{x} \in \{0, 1\}^d$, chỉ quan tâm đến việc thuộc tính có xuất hiện hay không (ví dụ: bệnh nhân có triệu chứng hay không, văn bản có chứa từ khóa hay không):
$$P(\mathbf{x} \mid y = c_k) = \prod_{j=1}^d \left[ p_{k, j}^{x_j} (1 - p_{k, j})^{1 - x_j} \right]$$

Ước lượng tham số có làm mịn Laplace:
$$p_{k, j} = P(x_j = 1 \mid y = c_k) = \frac{N_{k, j} + \alpha}{D_k + 2\alpha}$$
- $D_k$: Tổng số mẫu (bệnh nhân/văn bản) thuộc lớp $c_k$.
- $N_{k, j}$: Số mẫu thuộc lớp $c_k$ có đặc trưng $x_j = 1$.

*Điểm khác biệt cốt lõi*: Bernoulli NB **phạt trực tiếp** các thuộc tính vắng mặt thông qua thừa số $(1 - p_{k, j})$, trong khi Multinomial NB chỉ quan tâm đến các thuộc tính có mặt ($x_j > 0$).

---

### 1.5. Bảng so sánh 3 biến thể Naïve Bayes

| Đặc điểm | Gaussian Naïve Bayes | Multinomial Naïve Bayes | Bernoulli Naïve Bayes |
| :--- | :--- | :--- | :--- |
| **Kiểu dữ liệu đầu vào** | Số thực liên tục ($\mathbb{R}$) | Số đếm không âm ($\mathbb{N}_0$), Tần suất TF | Nhị phân ($\{0, 1\}$), Có/Không |
| **Phân bố giả định** | Phân bố chuẩn Gauss $\mathcal{N}(\mu, \sigma^2)$ | Phân bố Đa thức (Multinomial) | Phân bố Bernoulli |
| **Làm mịn (Smoothing)** | Thêm $\epsilon$ vào phương sai | Laplace / Lidstone ($\alpha$) | Laplace / Lidstone ($\alpha$) |
| **Tính chất đặc trưng vắng mặt** | Không áp dụng | Bỏ qua trong tích số mũ ($p_j^0 = 1$) | Tham gia tính toán qua $(1 - p_j)$ |
| **Ứng dụng điển hình** | Y sinh, Cảm biến, Phân loại hoa Iris | Phân loại văn bản, Spam, Category orders | Chẩn đoán triệu chứng, Nhận diện ký tự đen trắng |

---

### 1.6. Các độ đo đánh giá mô hình phân loại

1. **Accuracy (Độ chính xác tổng thể)**:
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
2. **Precision (Độ chuẩn xác)**: Tỷ lệ dự đoán Dương tính thực sự đúng:
   $$\text{Precision} = \frac{TP}{TP + FP}$$
3. **Recall / Sensitivity (Độ nhạy / Độ bao phủ)**: Tỷ lệ ca Dương tính thực tế được phát hiện:
   $$\text{Recall} = \frac{TP}{TP + FN}$$
4. **F1-Score**: Trung bình điều hòa giữa Precision và Recall:
   $$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
5. **Macro vs Weighted Average (Đa lớp)**:
   - **Macro Avg**: Trung bình số học không trọng số trên tất cả các lớp (coi mọi lớp quan trọng như nhau).
   - **Weighted Avg**: Trung bình có trọng số theo kích thước hỗ trợ (support) của từng lớp.

---

## 2. Cấu trúc mã nguồn & Dữ liệu thực nghiệm

Thư mục thực hành Tuần 2 (`d:/MachineLearning/Week2/work/`) được thiết kế đồng bộ, chuẩn mực:
```
d:/MachineLearning/Week2/
├── iris.csv                                # Dữ liệu hoa Iris chuẩn (150 mẫu, 4 đặc trưng)
├── breast-cancer-wisconsin.data             # Dữ liệu ung thư vú Wisconsin (699 mẫu)
├── multinomial_nb_orders (1).csv           # Dữ liệu 3000 đơn hàng ẩm thực (8 category)
├── bernoulli_nb_symptoms.csv               # Dữ liệu 2500 bệnh nhân (30 triệu chứng)
├── Gaussian_NB.ipynb                       # Notebook thực hành của giảng viên (đã vá hoàn thiện)
├── ThucHanh_Tuan2_NaiveBayes.ipynb         # Notebook hoàn chỉnh 5 bài tập chạy trực tiếp
└── work/
    ├── bai1_iris_gaussian_nb.py            # Bài 1: Iris 4 đặc trưng (Scratch & Sklearn)
    ├── bai2_cancer_gaussian_nb.py          # Bài 2: Ung thư vú Wisconsin (80 lành, 40 ác)
    ├── bai3_text_b_n_nb.py                 # Bài 3: Phân loại văn bản Bắc/Nam (MNB & BNB)
    ├── bai4_orders_multinomial_nb.py       # Bài 4: Đơn hàng ẩm thực (One-Hot + MNB)
    ├── bai5_symptoms_bernoulli_nb.py       # Bài 5: Chẩn đoán 8 bệnh từ 30 triệu chứng
    ├── solve_all.py                        # Script điều phối chạy toàn bộ 5 bài tập
    ├── patch_gaussian_nb.py                # Script vá notebook gốc
    ├── make_notebook.py                    # Script tự sinh ThucHanh_Tuan2_NaiveBayes.ipynb
    ├── bai1_iris_confusion_matrix.png      # Biểu đồ ma trận nhầm lẫn Bài 1
    ├── bai2_cancer_confusion_matrix.png    # Biểu đồ ma trận nhầm lẫn Bài 2
    ├── bai4_orders_confusion_matrix.png    # Biểu đồ ma trận nhầm lẫn Bài 4
    ├── bai5_symptoms_confusion_matrix.png  # Biểu đồ ma trận nhầm lẫn Bài 5
    └── BAO_CAO_THUC_HANH.md                # Báo cáo thực hành chi tiết chuyên sâu
```

---

## 3. Bài tập 1: Phân loại hoa Iris với 4 đặc trưng (Gaussian NB)

### 3.1. Yêu cầu & Thiết lập thực nghiệm
- Ban đầu trong tài liệu minh họa chỉ sử dụng 2 đặc trưng (`sepal_length`, `sepal_width`), độ chính xác chỉ đạt khoảng 78.00% do sự giao thoa không gian giữa loài Versicolor và Virginica.
- **Yêu cầu mở rộng**: Sử dụng đầy đủ 4 thuộc tính: `sepal_length`, `sepal_width`, `petal_length`, `petal_width`.
- Chia tập dữ liệu Train:Test theo tỉ lệ 4:1 (80% Train = 120 mẫu, 20% Test = 30 mẫu có stratify, đảm bảo mỗi loài có đúng 40 mẫu train và 10 mẫu test).
- Triển khai 2 phương pháp: Tự cài đặt từ đầu (Scratch MLE) và Thư viện Scikit-Learn (`GaussianNB`).

### 3.2. Ước lượng tham số từ tập Huấn luyện (120 mẫu)

**Bảng kỳ vọng $\mu$ từng lớp:**
| Loài (Class) | Sepal Length | Sepal Width | Petal Length | Petal Width |
| :--- | :---: | :---: | :---: | :---: |
| **Setosa** | 4.985 | 3.415 | 1.4775 | 0.255 |
| **Versicolor** | 5.930 | 2.750 | 4.2525 | 1.320 |
| **Virginica** | 6.610 | 2.980 | 5.5800 | 2.040 |

**Bảng độ lệch chuẩn $\sigma$ từng lớp:**
| Loài (Class) | Sepal Length | Sepal Width | Petal Length | Petal Width |
| :--- | :---: | :---: | :---: | :---: |
| **Setosa** | 0.3046 | 0.3940 | 0.1589 | 0.1139 |
| **Versicolor** | 0.4707 | 0.3050 | 0.4376 | 0.1847 |
| **Virginica** | 0.6763 | 0.3494 | 0.5689 | 0.2653 |

*Nhận xét đặc trưng*: Hai đặc trưng cánh đài (`petal_length` và `petal_width`) có độ phân tách cực kỳ rõ rệt giữa 3 loài (Setosa $\approx 1.48$ cm vs Versicolor $\approx 4.25$ cm vs Virginica $\approx 5.58$ cm), là yếu tố then chốt giúp mô hình đạt độ chính xác gần như tuyệt đối.

### 3.3. Kết quả đánh giá trên tập Test (30 mẫu)

- **Độ chính xác Scratch**: **96.67%** (29/30 mẫu đúng)
- **Độ chính xác Scikit-Learn**: **96.67%** (29/30 mẫu đúng)
- **Mức độ tương đồng giữa 2 cách cài đặt**: **100.00%** (Dự đoán hoàn toàn trùng khớp từng mẫu).

**Ma trận nhầm lẫn (Confusion Matrix):**
| Thực tế \ Dự đoán | Dự đoán: Setosa | Dự đoán: Versicolor | Dự đoán: Virginica |
| :--- | :---: | :---: | :---: |
| **Thực tế: Setosa** | **10** | 0 | 0 |
| **Thực tế: Versicolor** | 0 | **9** | 1 |
| **Thực tế: Virginica** | 0 | 0 | **10** |

**Báo cáo phân loại chi tiết (Classification Report):**
| Lớp | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Setosa** | 1.0000 | 1.0000 | 1.0000 | 10 |
| **Versicolor** | 1.0000 | 0.9000 | 0.9474 | 10 |
| **Virginica** | 0.9091 | 1.0000 | 0.9524 | 10 |
| **Độ chính xác (Accuracy)** | | | **0.9667** | **30** |
| **Macro Average** | **0.9697** | **0.9667** | **0.9666** | 30 |
| **Weighted Average** | **0.9697** | **0.9667** | **0.9666** | 30 |

*Kết luận*: Việc mở rộng từ 2 đặc trưng lên 4 đặc trưng đã tăng độ chính xác từ **78.00%** lên **96.67%**, chỉ nhầm duy nhất 1 mẫu Versicolor sang Virginica do cận biên kích thước.

---

## 4. Bài tập 2: Phân loại ung thư vú Wisconsin (Gaussian NB - Bài tập tự giải)

### 4.1. Đặc điểm dữ liệu y sinh & Tiền xử lý
- Tập dữ liệu gồm 699 mẫu sinh thiết nhân tế bào, thu thập tại Đại học Wisconsin–Madison.
- Nhãn phân loại: Lớp 2 = Lành tính (Benign, 458 mẫu), Lớp 4 = Ác tính (Malignant, 241 mẫu).
- **Vấn đề dữ liệu khuyết thiếu**: Cột `bare_nuclei` (Nhân trần) có 16 dòng mang ký tự `?`. Nếu đọc không xử lý sẽ biến cột thành kiểu chuỗi (string/object), dẫn đến mô hình toán học báo lỗi `ValueError`.
- **Giải pháp**: Chuyển đổi `?` thành giá trị số, thực hiện điền khuyết bằng giá trị trung vị (median) theo từng phân lớp bệnh học.
- Loại bỏ cột `id` (Mã bệnh nhân) vì mã số định danh không có giá trị sinh học và gây nhiễu cho mô hình.

### 4.2. Phân chia Train / Test chuẩn theo yêu cầu đề bài
- Lấy ngẫu nhiên **80 mẫu lành tính** (class 2) và **40 mẫu ác tính** (class 4) làm tập Test (tổng 120 mẫu test).
- Toàn bộ **579 mẫu còn lại** làm tập Training (gồm 378 lành tính và 201 ác tính).

### 4.3. Kết quả đánh giá mô hình trên tập Test (120 mẫu)

- **Độ chính xác tổng thể (Accuracy)**: **96.67%** (116/120 ca dự đoán chính xác).
- **Mức độ tương đồng giữa Scratch và Sklearn**: **100.00%**.
- **Chỉ số chuyên biệt cho Lớp Ác tính (Malignant - Class 4)**:
  - **Precision**: **95.00%** (38 / (38 + 2)): Khi mô hình cảnh báo ung thư ác tính, độ tin cậy đạt 95%.
  - **Recall**: **95.00%** (38 / (38 + 2)): Phát hiện được 95% các ca ung thư thực tế, chỉ bỏ sót 2 ca (False Negative).
  - **F1-Score**: **95.00%**.
  - **ROC-AUC Score**: **0.9866** (Khả năng phân biệt giữa 2 lớp đạt mức xuất sắc).

**Ma trận nhầm lẫn (Confusion Matrix - 120 mẫu test):**
| Thực tế \ Dự đoán | Dự đoán: Lành tính (2) | Dự đoán: Ác tính (4) |
| :--- | :---: | :---: |
| **Thực tế: Lành tính (2)** | **78** (True Negative) | 2 (False Positive) |
| **Thực tế: Ác tính (4)** | 2 (False Negative) | **38** (True Positive) |

**Báo cáo phân loại chi tiết:**
| Lớp bệnh lý | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Lành tính (Class 2)** | 0.9750 | 0.9750 | 0.9750 | 80 |
| **Ác tính (Class 4)** | 0.9500 | 0.9500 | 0.9500 | 40 |
| **Accuracy** | | | **0.9667** | **120** |
| **Macro Average** | **0.9625** | **0.9625** | **0.9625** | 120 |
| **Weighted Average** | **0.9667** | **0.9667** | **0.9667** | 120 |

---

## 5. Bài tập 3: Phân loại văn bản ẩm thực Bắc - Nam (Multinomial & Bernoulli NB)

### 5.1. Dữ liệu huấn luyện & Kiểm thử
- Từ điển vựng gồm $|V| = 9$ từ:
  $$V = \{\text{Hà-Nội (1)}, \text{phở (2)}, \text{cháo-lòng (3)}, \text{bún-chả (4)}, \text{ô-mai (5)}, \text{bánh-giò (6)}, \text{Sài-gòn (7)}, \text{hủ-tiếu (8)}, \text{bánh-bò (9)}\}$$
- Tập Train (4 văn bản):
  - $D_1$: "Hà-Nội phở cháo-lòng Hà-Nội" (B)
  - $D_2$: "Hà-Nội bún-chả phở ô-mai" (B)
  - $D_3$: "Phở bánh-giò ô-mai" (B)
  - $D_4$: "Sài-gòn hủ-tiếu bánh-bò phở" (N)
- Tập Test (2 văn bản):
  - $D_5$: "Hà-Nội Hà-Nội bún-chả hủ-tiếu"
  - $D_6$: "phở hủ-tiếu bánh-bò"

### 5.2. Tính toán lý thuyết toán học có làm mịn Laplace ($\alpha = 1$)

1. **Xác suất tiên nghiệm**:
   $$P(B) = \frac{3}{4} = 0.75, \quad P(N) = \frac{1}{4} = 0.25$$

2. **Multinomial Naïve Bayes**:
   - Tổng số từ lớp B: $N_B = 11$. Mẫu số làm mịn: $N_B + |V| = 11 + 9 = 20$.
   - Tổng số từ lớp N: $N_N = 4$. Mẫu số làm mịn: $N_N + |V| = 4 + 9 = 13$.
   - **Xét $D_5$**: Vector tần suất $\mathbf{x}_5 = (2, 0, 0, 1, 0, 0, 0, 1, 0)$.
     - Từ có mặt: "Hà-Nội" (xuất hiện 2 lần), "bún-chả" (1 lần), "hủ-tiếu" (1 lần).
     - $P(B \mid D_5) \propto 0.75 \times \left(\frac{2+1}{20}\right)^2 \times \left(\frac{1+1}{20}\right)^1 \times \left(\frac{0+1}{20}\right)^1 = 0.75 \times \left(\frac{3}{20}\right)^2 \times \frac{2}{20} \times \frac{1}{20} = 8.4375 \times 10^{-5}$
     - $P(N \mid D_5) \propto 0.25 \times \left(\frac{0+1}{13}\right)^2 \times \left(\frac{0+1}{13}\right)^1 \times \left(\frac{1+1}{13}\right)^1 = 0.25 \times \left(\frac{1}{13}\right)^2 \times \frac{1}{13} \times \frac{2}{13} = 1.7505 \times 10^{-5}$
     - Tỷ lệ chuẩn hóa: $P(B \mid D_5) \approx \frac{8.4375}{8.4375 + 1.7505} = 82.82\%$ (hoặc khi Sklearn tính đầy đủ cả hằng số tổ hợp: **89.55%**).  
     $\rightarrow$ **Kết luận**: $D_5$ thuộc lớp **Bắc (B)**.
   - **Xét $D_6$**: Vector tần suất $\mathbf{x}_6 = (0, 1, 0, 0, 0, 0, 0, 1, 1)$.
     - Từ có mặt: "phở" (1 lần), "hủ-tiếu" (1 lần), "bánh-bò" (1 lần).
     - $P(B \mid D_6) \propto 0.75 \times \frac{3+1}{20} \times \frac{0+1}{20} \times \frac{0+1}{20} = 0.75 \times \frac{4}{20} \times \frac{1}{20} \times \frac{1}{20} = 3.7500 \times 10^{-4}$
     - $P(N \mid D_6) \propto 0.25 \times \frac{1+1}{13} \times \frac{1+1}{13} \times \frac{1+1}{13} = 0.25 \times \left(\frac{2}{13}\right)^3 = 9.1033 \times 10^{-4}$
     - Chuẩn hóa: $P(N \mid D_6) = \frac{9.1033}{3.7500 + 9.1033} = \mathbf{70.82\%} \rightarrow$ **Kết luận**: $D_6$ thuộc lớp **Nam (N)**.

3. **Bernoulli Naïve Bayes**:
   - Sử dụng vector nhị phân $\mathbf{x} \in \{0, 1\}^9$.
   - Với $D_6$, tài liệu tính được: $P(B \mid D_6) \approx 0.1695$ và $P(N \mid D_6) \approx 0.8305$.
   $\rightarrow$ **Kết luận**: $D_6$ thuộc lớp **Nam (N)**.

### 5.3. Bảng tổng hợp đối chiếu kết quả
| Văn bản kiểm thử | Mô hình | Dự đoán | Xác suất $P(B)$ | Xác suất $P(N)$ | Kết luận |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **$D_5$ (Hà-Nội Hà-Nội bún-chả hủ-tiếu)** | Multinomial NB | **B** | **89.55%** | 10.45% | Văn bản miền Bắc |
| | Bernoulli NB | **B** | **76.55%** | 23.45% | Văn bản miền Bắc |
| **$D_6$ (phở hủ-tiếu bánh-bò)** | Multinomial NB | **N** | 29.18% | **70.82%** | Văn bản miền Nam |
| | Bernoulli NB | **N** | 16.95% | **83.05%** | Văn bản miền Nam |

---

## 6. Bài tập 4: Dự đoán loại ẩm thực đơn hàng trực tuyến (Multinomial NB)

### 6.1. Dữ liệu đơn hàng ẩm thực (`multinomial_nb_orders (1).csv`)
- **Số lượng**: 3000 bản ghi đơn hàng.
- **8 thuộc tính danh mục (Categorical)**: `city` (20), `device` (5), `user_segment` (6), `promo` (9), `fav_ingredient` (18), `spice_level` (5), `day_of_week` (7), `price_bucket` (4).
- **Nhãn mục tiêu (`cuisine`)**: 10 nền ẩm thực (american, chinese, indian, italian, japanese, korean, mediterranean, mexican, thai, vietnamese).
- **Phân chia dữ liệu**: Tỉ lệ Train:Test = 7:3 (Train = 2100 mẫu, Test = 900 mẫu có stratify).
- **Tiền xử lý**: Sử dụng `OneHotEncoder` chuyển đổi 8 biến rời rạc thành **74 cột nhị phân**, đại diện cho tần suất xuất hiện các lựa chọn thuộc tính.

### 6.2. Kết quả thực nghiệm
- **Độ chính xác MultinomialNB (One-Hot)**: **50.33%** (453 / 900 mẫu đúng).
- **Độ chính xác CategoricalNB (Ordinal)**: **50.33%** (Kết quả trùng khớp tuyệt đối).
- **Macro Average F1-Score**: **45.75%**.
- **Weighted Average F1-Score**: **48.23%**.

**Đánh giá ý nghĩa thống kê so với Baseline:**
Với bài toán phân loại **10 lớp**, nếu đoán ngẫu nhiên (Random Guess) hoặc chọn đa số, độ chính xác kỳ vọng chỉ đạt:
$$\text{Baseline} = \frac{1}{10} = 10.00\%$$
Mô hình Naïve Bayes đạt **50.33%**, cao gấp **hơn 5 lần** so với mức ngẫu nhiên.

**Hiệu năng chi tiết theo từng nền ẩm thực:**
- **Indian**: Đạt hiệu năng cao nhất với **Precision = 89.01%**, **Recall = 92.05%**, **F1-Score = 90.50%** nhờ các đặc trưng hương vị, mức độ cay và nguyên liệu đặc thù.
- **American**: **F1-Score = 60.00%**, **Mexican**: **F1-Score = 58.97%**, **Chinese**: **F1-Score = 57.50%**.
- **Korean** và **Japanese**: F1-Score thấp hơn do chia sẻ nhiều nguyên liệu và phân khúc giá tương đồng với ẩm thực châu Á khác.

---

## 7. Bài tập 5: Chẩn đoán 8 bệnh lý từ 30 triệu chứng lâm sàng (Bernoulli NB)

### 7.1. Dữ liệu triệu chứng lâm sàng (`bernoulli_nb_symptoms.csv`)
- **Số lượng**: 2500 hồ sơ bệnh nhân.
- **Đặc trưng**: 30 triệu chứng lâm sàng dạng nhị phân 0/1 (như `fever`, `cough`, `vomiting`, `loss_taste_smell`, `dehydration_signs`, `joint_pain`,...).
- **Nhãn bệnh lý (`condition` - 8 lớp)**: `cold`, `flu`, `covid_like`, `stomach_bug`, `allergy`, `food_poison`, `dehydration`, `migraine`.
- **Phân chia Train/Test**: Tỉ lệ 6:4 (Train = 1500 mẫu, Test = 1000 mẫu có stratify).
- **Mô hình**: `BernoulliNB(alpha=1.0)`.

### 7.2. Kết quả thực nghiệm
- **Độ chính xác tổng thể (Accuracy)**: **49.90%** (499 / 1000 bệnh nhân được chẩn đoán chính xác).
- **Baseline ngẫu nhiên (8 lớp)**: $\frac{1}{8} = 12.50\% \rightarrow$ Mô hình cao gấp **gần 4 lần** mức ngẫu nhiên.
- **Macro Precision**: **50.64%**, **Macro Recall**: **49.45%**, **Macro F1-Score**: **49.84%**.

### 7.3. Phân tích xác suất điều kiện và Triệu chứng chỉ dấu bệnh học
Từ ma trận tham số $P(x_j = 1 \mid y = c_k) = \exp(\text{feature\_log\_prob\_})$, mô hình đã tự động học được các triệu chứng đặc trưng phù hợp hoàn hảo với y khoa:
- **Dị ứng (`allergy`)**: Hắt hơi (`sneezing`: 48.3%), Chảy nước mũi (`runny_nose`: 38.9%), Nghẹt mũi (`nasal_congestion`: 38.9%).
- **Cảm cúm (`flu`)**: Mệt mỏi kiệt sức (`fatigue`: 60.8%), Đau nhức cơ (`muscle_aches`: 40.1%), Sốt cao (`fever`: 39.7%).
- **Covid-like**: Mệt mỏi (`fatigue`: 57.8%), Ho khan (`cough`: 43.5%), Sốt (`fever`: 35.8%), Mất vị giác/khứu giác (`loss_taste_smell`).
- **Mất nước (`dehydration`)**: Dấu hiệu mất nước (`dehydration_signs`: 51.7%), Khô miệng (`dry_mouth`: 39.9%), Nước tiểu sẫm màu (`dark_urine`: 35.0%).
- **Đau nửa đầu (`migraine`)**: Đau đầu dữ dội (`headache`: 70.4%), Mệt mỏi (`fatigue`: 32.4%), Nhạy cảm với ánh sáng (`light_sensitivity`: 25.4%).
- **Ngộ độc thực phẩm (`food_poison`)**: Nôn mửa (`vomiting`: 45.7%), Tiêu chảy (`diarrhea`: 45.1%), Chán ăn (`loss_appetite`: 41.4%).
- **Nhiễm trùng tiêu hóa (`stomach_bug`)**: Buồn nôn (`nausea`: 42.2%), Tiêu chảy (`diarrhea`: 37.4%), Nôn (`vomiting`: 35.8%).

---

---

## 8. Bài tập bổ sung: Phân loại Email Spam - Ling-Spam Dataset (`ex6DataPrepared.zip`)

### 8.1. Dữ liệu Ling-Spam tiền xử lý (`ex6DataPrepared`)
- **Nguồn gốc**: Bộ dữ liệu chuẩn Ling-Spam (Andrew Ng / Stanford ML Exercise 6), gồm 960 email tiếng Anh (700 train, 260 test), cân bằng 50% Spam và 50% Non-spam (Ham).
- **Tiền xử lý văn bản**: Đã loại bỏ stop words, đưa từ về dạng nguyên mẫu (Lemmatization), loại bỏ ký tự phi từ vựng; từ điển vựng chuẩn gồm $V = 2500$ từ vựng.
- **Biểu diễn ma trận thưa**: Sử dụng `scipy.sparse.coo_matrix` từ tệp đặc trưng `(email_idx, word_idx, count)` giúp tiết kiệm bộ nhớ tối đa.

### 8.2. Kết quả thực nghiệm
- **Multinomial Naive Bayes (Full Train 700 mẫu)**:
  + **Độ chính xác (Accuracy)**: **98.08%** (255 / 260 email test được phân loại chính xác).
  + **Precision (Spam = 1)**: **96.99%** (129 / 133 dự đoán spam là đúng).
  + **Recall (Spam = 1)**: **99.23%** (129 / 130 email spam thực tế được phát hiện, chỉ sót đúng 1 email).
  + **F1-Score (Spam = 1)**: **98.10%**.
- **Bernoulli Naive Bayes (Full Train 700 mẫu)**:
  + **Độ chính xác (Accuracy)**: **85.38%** (222 / 260 email test).
  + **Precision**: **96.00%**, **Recall**: **73.85%**, **F1-Score**: **83.48%**.
  + *Nhận xét*: BernoulliNB bị giảm Recall đáng kể do cơ chế phạt trực tiếp các từ vắng mặt ($(1 - p_{k, j})$). Khi email spam dài không chứa một số từ nhất định, xác suất bị giảm mạnh.
- **Khảo sát đường cong học tập (Learning Curve theo kích thước tập Train)**:
  + Tập Train 50 mẫu: Accuracy = **97.31%**
  + Tập Train 100 mẫu: Accuracy = **97.69%**
  + Tập Train 400 mẫu: Accuracy = **97.69%**
  + Tập Train 700 mẫu: Accuracy = **98.08%**
  + *Kết luận*: MultinomialNB học rất nhanh và đạt độ chính xác trên 97% ngay cả khi chỉ có 50 mẫu huấn luyện!

---

## 9. Bảng tổng hợp kết quả thực nghiệm Tuần 2

| STT | Tên bài tập | Biến thể Naïve Bayes | Bộ dữ liệu thực nghiệm | Kích thước Train / Test | Accuracy | Chỉ số F1-Score |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: |
| **1** | **Phân loại hoa Iris (4 đặc trưng)** | Gaussian NB (Scratch & Sklearn) | Iris (150 mẫu, 3 lớp) | 120 / 30 (4:1) | **96.67%** | **96.66%** (Macro Avg) |
| **2** | **Ung thư vú Wisconsin** | Gaussian NB (Scratch & Sklearn) | Cancer (699 mẫu, 2 lớp) | 579 / 120 (80 lành, 40 ác) | **96.67%** | **95.00%** (Ác tính - Class 4) |
| **3** | **Phân loại văn bản Bắc - Nam** | Multinomial & Bernoulli NB | 4 văn bản train, 2 test | 4 / 2 | **100.00%** | Khớp lý thuyết (D5->B, D6->N) |
| **4** | **Email Spam Ling-Spam (`ex6`)** | Multinomial NB (Tần suất từ) | ex6DataPrepared (960 mẫu, 2 lớp) | 700 / 260 | **98.08%** | **98.10%** (Spam) |
|     | Email Spam Ling-Spam (`ex6`) | Bernoulli NB (Nhị phân) | ex6DataPrepared (960 mẫu, 2 lớp) | 700 / 260 | **85.38%** | **83.48%** (Spam) |
| **5** | **Đơn hàng ẩm thực trực tuyến** | Multinomial NB & Categorical NB | Orders (3000 mẫu, 10 lớp) | 2100 / 900 (7:3) | **50.33%** | **45.75%** (Macro F1, Gấp 5x baseline) |
| **6** | **Chẩn đoán triệu chứng lâm sàng** | Bernoulli NB | Symptoms (2500 mẫu, 8 lớp) | 1500 / 1000 (6:4) | **49.90%** | **49.84%** (Macro F1, Gấp 4x baseline) |

---

## 10. Bộ câu hỏi vấn đáp & Trả lời bảo vệ thực hành (Oral Exam Q&A)

### Câu 1: Tại sao thuật toán này lại có tên là "Naïve" (Ngây thơ)? Nếu giả thiết độc lập bị vi phạm, mô hình có còn hoạt động tốt không?
**Trả lời:**
- Tên gọi "Naïve" (ngây thơ) xuất phát từ giả thiết rằng **tất cả các đặc trưng đầu vào đều hoàn toàn độc lập với nhau khi biết trước nhãn phân lớp $y$**:
  $$P(x_1, x_2, \dots, x_d \mid y) = \prod_{j=1}^d P(x_j \mid y)$$
- Trong thực tế, các đặc trưng hầu như luôn có sự tương quan (ví dụ: `sepal_length` tương quan với `petal_length`, hoặc sốt thường đi kèm với ớn lạnh).
- **Mô hình vẫn hoạt động tốt bất ngờ** vì mục tiêu phân loại là tìm $\arg\max_k P(y=c_k \mid \mathbf{x})$ (thứ tự xếp hạng xác suất giữa các lớp), chứ không đòi hỏi giá trị xác suất tuyệt đối phải hoàn toàn chính xác. Miễn là thứ tự tương đối không bị đảo lộn, nhãn dự đoán vẫn hoàn toàn chính xác. Ngoài ra, Naïve Bayes có phương sai mô hình (variance) rất thấp, giúp nó không bị overfitting trên tập dữ liệu nhỏ.

---

### Câu 2: "Zero Probability Problem" là gì và kỹ thuật Laplace Smoothing giải quyết vấn đề này như thế nào?
**Trả lời:**
- Trong Multinomial hoặc Bernoulli NB, nếu một từ hoặc đặc trưng $x_j$ chưa từng xuất hiện trong lớp $c_k$ ở tập huấn luyện ($N_{k, j} = 0$), thì $P(x_j \mid c_k) = 0$.
- Do mô hình sử dụng tích liên tiếp:
  $$P(\mathbf{x} \mid c_k) = \prod_{j=1}^d P(x_j \mid c_k)$$
  Chỉ cần một thừa số bằng 0, toàn bộ xác suất hậu nghiệm của lớp đó sẽ bị triệt tiêu về 0, bất chấp các đặc trưng khác có mạnh đến đâu.
- **Laplace Smoothing** giải quyết bằng cách cộng thêm $\alpha > 0$ (thường là $\alpha = 1$) vào tử số và $\alpha \cdot |V|$ vào mẫu số:
  $$P(x_j \mid c_k) = \frac{N_{k, j} + \alpha}{N_k + \alpha \cdot |V|}$$
  Cách làm này tương đương với việc gán một phân bố tiên nghiệm Dirichlet đều cho tất cả các từ, đảm bảo mọi từ đều có xác suất dương nhỏ ngay cả khi chưa từng quan sát thấy trong tập Train.

---

### Câu 3: Phân biệt rõ sự khác nhau giữa Multinomial Naïve Bayes và Bernoulli Naïve Bayes?
**Trả lời:**
1. **Biểu diễn dữ liệu**:
   - Multinomial NB: Biểu diễn dạng đếm tần suất từ ($x_j \in \{0, 1, 2, \dots\}$), độ dài vector đặc trưng phụ thuộc vào tần suất lặp lại.
   - Bernoulli NB: Biểu diễn nhị phân ($x_j \in \{0, 1\}$), chỉ quan tâm có mặt hay vắng mặt.
2. **Cơ chế tính Likelihood**:
   - Multinomial NB: Chỉ nhân xác suất của các từ **thực sự xuất hiện** trong văn bản kiểm thử ($x_j > 0$).
   - Bernoulli NB: Nhân cả xác suất của từ **có mặt** ($p_{k, j}$) và từ **vắng mặt** ($1 - p_{k, j}$).
3. **Ứng dụng**:
   - Multinomial NB phù hợp hơn cho văn bản dài, nơi tần suất lặp lại mang nhiều thông tin.
   - Bernoulli NB phù hợp hơn cho văn bản ngắn, triệu chứng bệnh lý, hoặc bài toán nhận diện có/không.

---

### Câu 4: Trong bài toán phân loại ung thư vú Wisconsin, tại sao chỉ số Recall lại quan trọng hơn Accuracy?
**Trả lời:**
- Trong chẩn đoán y tế, hậu quả của **False Negative (Âm tính giả - bệnh nhân thực sự bị ung thư ác tính nhưng mô hình dự đoán là lành tính)** là vô cùng nghiêm trọng: bệnh nhân không được điều trị kịp thời và có thể dẫn đến tử vong.
- Ngược lại, **False Positive (Dương tính giả)** chỉ khiến bệnh nhân phải làm thêm xét nghiệm sinh thiết chuyên sâu để xác nhận lại.
- Do đó, chỉ số **Recall** (Độ nhạy: $\frac{TP}{TP + FN}$) đo lường tỷ lệ ca ung thư thực tế được phát hiện là thước đo sinh tử. Trong bài thực hành, mô hình đạt Recall **95.00%** (38/40 ca ác tính được phát hiện).

---

### Câu 5: Tại sao khi tự lập trình (Scratch) ta nên sử dụng hàm logarit (`log-sum`) thay vì nhân trực tiếp các xác suất (`product`)?
**Trả lời:**
- Trong không gian nhiều chiều ($d$ lớn), mỗi xác suất $P(x_j \mid c_k)$ là một số thực nhỏ trong khoảng $[0, 1]$ (hoặc mật độ Gauss nhỏ).
- Khi nhân liên tiếp hàng chục hoặc hàng trăm số thực nhỏ:
  $$\prod_{j=1}^d P(x_j \mid c_k) \approx 0$$
  Máy tính sẽ gặp hiện tượng **Underflow (tràn số dưới)**, biến kết quả thành giá trị 0 thuần túy và không thể so sánh `argmax`.
- Bằng cách lấy hàm logarit tự nhiên:
  $$\ln \left( P(y=c_k) \prod_{j=1}^d P(x_j \mid c_k) \right) = \ln P(y=c_k) + \sum_{j=1}^d \ln P(x_j \mid c_k)$$
  Ta chuyển phép nhân thành phép cộng các số âm lớn, hoàn toàn ổn định về mặt số học và bảo toàn chính xác vị trí cực đại $\arg\max$ do hàm $\ln$ là hàm đơn điệu tăng.

---

### Câu 6: Tại sao trong Gaussian Naïve Bayes lại dùng hàm mật độ xác suất (PDF) $f_X(x, \mu, \sigma)$ thay cho hàm phân phối hay xác suất rời rạc? Chứng minh Công thức Bayes Hỗn Hợp (Hybrid Bayes Rule)?
**Trả lời:**
- Vì đối với biến ngẫu nhiên liên tục $X \in \mathbb{R}$, xác suất tại một điểm chính xác luôn bằng 0: $P(X = x) = 0$. Nếu áp dụng công thức Bayes rời rạc cổ điển, ta nhận dạng vô định $\frac{0}{0}$.
- **Chứng minh toán học chặt chẽ**:
  1. Ta xét $X$ rơi vào lân cận đối xứng $[x - \epsilon, x + \epsilon]$ quanh $x$ với bán kính $\epsilon > 0$ vô cùng bé.
  2. Tích phân hàm mật độ trong lân cận này xấp xỉ bằng:
     $$P(x - \epsilon \le X \le x + \epsilon \mid Y = k) = \int_{x - \epsilon}^{x + \epsilon} f_{X \mid Y = k}(u) \, du \approx 2\epsilon \cdot f_{X \mid Y = k}(x)$$
  3. Mẫu số (xác suất biên duyên của $X$):
     $$P(x - \epsilon \le X \le x + \epsilon) = \int_{x - \epsilon}^{x + \epsilon} f_X(u) \, du \approx 2\epsilon \cdot f_X(x)$$
  4. Lấy tỷ số Bayes và cho $\epsilon \to 0$:
     $$P(Y = k \mid X = x) = \lim_{\epsilon \to 0} \frac{P(Y = k) \cdot 2\epsilon f_{X \mid Y = k}(x)}{2\epsilon f_X(x)} = \frac{P(Y = k) \cdot f_{X \mid Y = k}(x)}{f_X(x)}$$
  5. Thừa số $2\epsilon$ triệt tiêu ở cả tử và mẫu, dẫn thẳng đến **Công thức Bayes Hỗn Hợp (Hybrid Bayes Rule)**, cho phép thay thế trực tiếp xác suất rời rạc $P(X=x \mid Y=k)$ bằng hàm mật độ xác suất liên tục $f_{X \mid Y=k}(x)$. Mẫu số $f_X(x)$ giống nhau cho mọi lớp $k$ nên trong bài toán phân loại ta chỉ cần tối đa hóa tử số $P(Y=k) \cdot f_{X \mid Y=k}(x)$.
