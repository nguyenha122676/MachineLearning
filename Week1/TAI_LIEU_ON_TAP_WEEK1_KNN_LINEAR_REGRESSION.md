# GIÁO TRÌNH ÔN TẬP TOÀN DIỆN TUẦN 1 (WEEK 1)
## THUẬT TOÁN K-NEAREST NEIGHBORS (K-NN) CHO HỒI QUY & HỒI QUY TUYẾN TÍNH (LINEAR REGRESSION)

> **Sinh viên**: Nguyễn Hải Anh (MSSV: 24001968)  
> **Học phần**: Machine Learning (Học máy)  
> **Chủ đề**: Dự báo giá trị liên tục (Hồi quy - Regression)  
> **Thư mục liên quan**: `d:\MachineLearning\Week1`  
> **Các bài tập thực hành**: Bài B.1 (Nhiên liệu `fuel.txt`), Bài B.3 (SAT & GPA `SAT_GPA.csv`), Bài B.4 (Độ dày nội trung mạc NTM `vidu4_lin_reg.txt`), Bài B.5 (Bất động sản `real_estate.csv`).

---

## MỤC LỤC
1. [Bản chất bài toán Hồi quy (Regression) trong Machine Learning](#1-bản-chất-bài-toán-hồi-quy-regression-trong-machine-learning)
2. [Thuật toán 1: K-NN cho Hồi quy (K-Nearest Neighbors Regression)](#2-thuật-toán-1-k-nn-cho-hồi-quy-k-nearest-neighbors-regression)
   - 2.1. Trực giác đời sống
   - 2.2. Toán học chuyên sâu: Khoảng cách Euclid & Chuẩn hóa
   - 2.3. Quy trình từng bước & Thuật toán
   - 2.4. Bản chất "Học lười" (Lazy Learning) & Hạn chế không ngoại suy
3. [Thuật toán 2: Hồi quy tuyến tính (Linear Regression)](#3-thuật-toán-2-hồi-quy-tuyến-tính-linear-regression)
   - 3.1. Trực giác hình học: Tìm đường thẳng tối ưu
   - 3.2. Mô hình 1 chiều: Nghiệm giải tích phương sai / hiệp phương sai
   - 3.3. Mô hình đa biến tổng quát: Normal Equation & Ma trận giả nghịch đảo
   - 3.4. Phân rã QR để tránh ma trận kỳ dị
4. [Các độ đo đánh giá mô hình Hồi quy (Evaluation Metrics)](#4-các-độ-đo-đánh-giá-mô-hình-hồi-quy-evaluation-metrics)
   - MSE, MAE, RMSE, R-squared ($R^2$), RSS, Phân tích phần dư Residuals
5. [Giải phẫu mã nguồn từng dòng (Line-by-Line Code Breakdown)](#5-giải-phẫu-mã-nguồn-từng-dòng-line-by-line-code-breakdown)
   - Bẫy tính khoảng cách `axis` trong NumPy
   - Kỹ thuật thêm cột số 1 (Bias trick)
   - Giải ma trận bằng `pinv` vs dùng `scikit-learn`
6. [Chi tiết 4 bài tập thực hành Tuần 1](#6-chi-tiết-4-bài-tập-thực-hành-tuần-1)
7. [Bộ 10 câu hỏi vấn đáp bảo vệ bài tập Tuần 1 kèm đáp án chuẩn](#7-bộ-10-câu-hỏi-vấn-đáp-bảo-vệ-bài-tập-tuần-1-kèm-đáp-án-chuẩn)

---

## 1. BẢN CHẤT BÀI TOÁN HỒI QUY (REGRESSION) TRONG MACHINE LEARNING

Trong học máy có giám sát (Supervised Learning), ta có một tập dữ liệu gồm $N$ mẫu huấn luyện:
$$\mathcal{D} = \{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \dots, (\mathbf{x}_N, y_N)\}$$
- $\mathbf{x}_i \in \mathbb{R}^d$: Vector đặc trưng đầu vào gồm $d$ chiều (ví dụ: diện tích, số phòng, khoảng cách đến tàu điện).
- $y_i$: Giá trị mục tiêu đầu ra.

> **Định nghĩa Hồi quy (Regression)**:  
> Là bài toán mà đầu ra $y$ là một **số thực liên tục** ($y \in \mathbb{R}$), ví dụ:
> - Dự đoán giá bán một căn nhà (triệu USD).
> - Dự đoán mức tiêu thụ nhiên liệu của một chiếc xe (lít/100km).
> - Dự đoán điểm thi GPA đại học từ điểm thi SAT.
> - Dự đoán độ dày màng nội trung mạc động mạch cảnh NTM (mm).
>
> *(Nếu $y$ là nhãn rời rạc như 0 hoặc 1, đỗ hoặc trượt, ta gọi đó là bài toán **Phân loại - Classification**, sẽ học ở Tuần 2, 3, 4).*

Tuần 1 giới thiệu cho bạn 2 cách tiếp cận hồi quy hoàn toàn đối lập nhau về mặt triết học:
1. **K-NN**: Tiếp cận cục bộ, phi tham số, không cần giả định hàm số.
2. **Linear Regression**: Tiếp cận toàn cục, có tham số, giả định trước quan hệ là đường thẳng/mặt phẳng tuyến tính.

---

## 2. THUẬT TOÁN 1: K-NN CHO HỒI QUY (K-NEAREST NEIGHBORS REGRESSION)

### 2.1. Trực giác đời sống
Hãy tưởng tượng bạn chuyển đến một thành phố mới và muốn biết giá thuê một căn hộ chung cư 65m². Bạn không có công thức tính giá bất động sản. Bạn làm thế nào?
- Bạn hỏi giá thuê của **$K = 5$ căn hộ chung cư khác** ngay gần đó, cũng có diện tích sàn sàn 65m² (gọi là 5 "láng giềng" gần nhất).
- Bạn thấy 5 căn đó có giá thuê lần lượt là: 10 triệu, 11 triệu, 10.5 triệu, 12 triệu, 11.5 triệu.
- Bạn tính **trung bình cộng**: $\frac{10 + 11 + 10.5 + 12 + 11.5}{5} = 11$ triệu đồng.
- Bạn kết luận căn hộ của mình có giá thuê khoảng 11 triệu đồng.

Đó chính là nguyên lý $100\%$ của **K-NN Regression**!

---

### 2.2. Toán học chuyên sâu: Đo khoảng cách & Chuẩn hóa dữ liệu

#### A. Hàm khoảng cách Euclid (Chuẩn $L_2$)
Để biết căn hộ nào "gần" căn hộ của ta nhất trong không gian đa chiều, ta dùng khoảng cách Euclid:
Giả sử mẫu cần dự đoán là $\mathbf{x} = [x_1, x_2, \dots, x_d]^T$ và mẫu huấn luyện là $\mathbf{x}_i = [x_{i1}, x_{i2}, \dots, x_{id}]^T$:
$$d(\mathbf{x}, \mathbf{x}_i) = \|\mathbf{x} - \mathbf{x}_i\|_2 = \sqrt{\sum_{j=1}^d (x_j - x_{ij})^2}$$

- Khi $d = 1$ (1 chiều): $d(x, x_i) = |x - x_i|$ (trị tuyệt đối hiệu số).
- Khi $d > 1$ (đa chiều): Căn bậc hai của tổng bình phương độ lệch trên từng chiều đặc trưng.

#### B. Tại sao BẮT BUỘC phải Chuẩn hóa dữ liệu (Feature Scaling)?
Giả sử ta dự đoán giá nhà dựa trên 2 đặc trưng:
- $x_1$: Diện tích nhà (từ $50$ đến $300$ m²).
- $x_2$: Số phòng ngủ (từ $1$ đến $5$ phòng).

Nếu lấy khoảng cách trực tiếp:
$$d = \sqrt{(150 - 100)^2 + (4 - 2)^2} = \sqrt{2500 + 4} = \sqrt{2504} \approx 50.04$$
Con số chênh lệch diện tích ($50^2 = 2500$) nuốt chửng hoàn toàn con số chênh lệch phòng ngủ ($2^2 = 4$). Mô hình K-NN sẽ coi số phòng ngủ như vô hình!

**Cách giải quyết**: Đưa tất cả các biến về cùng một thang đo trước khi tính khoảng cách:
1. **Chuẩn hóa Z-Score (Standardization)**:
   $$z = \frac{x - \mu}{\sigma} \quad (\text{đưa về trung bình } 0 \text{ và phương sai } 1)$$
2. **Chuẩn hóa Min-Max (Min-Max Scaling)**:
   $$x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}} \quad (\text{đưa về đoạn } [0, 1])$$

---

### 2.3. Quy trình từng bước & Thuật toán K-NN Regression

Cho một điểm kiểm thử $\mathbf{x}_{\text{test}}$:
1. **Bước 1**: Tính khoảng cách $d(\mathbf{x}_{\text{test}}, \mathbf{x}_i)$ tới tất cả các điểm $\mathbf{x}_i$ trong tập huấn luyện ($i = 1, 2, \dots, N$).
2. **Bước 2**: Sắp xếp các khoảng cách theo thứ tự tăng dần.
3. **Bước 3**: Lấy ra chỉ số của $K$ điểm có khoảng cách nhỏ nhất, gọi tập hợp các chỉ số này là $\mathcal{N}_K(\mathbf{x}_{\text{test}})$.
4. **Bước 4**: Giá trị dự đoán là trung bình cộng của các nhãn thực tế $y_i$ thuộc $K$ láng giềng đó:
   $$\hat{y} = \frac{1}{K} \sum_{i \in \mathcal{N}_K(\mathbf{x}_{\text{test}})} y_i$$
   *(Phiên bản nâng cao có thể dùng trung bình có trọng số theo nghịch đảo khoảng cách $w_i = \frac{1}{d_i}$).*

---

### 2.4. Bản chất "Học lười" (Lazy Learning) & Hạn chế không ngoại suy

> [!IMPORTANT]
> **Hai câu hỏi "tủ" giảng viên cực kỳ hay hỏi khi vấn đáp về K-NN:**
>
> 1. **Tại sao gọi K-NN là Lazy Learning?**  
>    Vì giai đoạn huấn luyện (Training phase) thực chất chẳng làm gì ngoài việc copy dữ liệu vào RAM! Nó không giải phương trình, không tối ưu trọng số. Toàn bộ gánh nặng tính toán bị đẩy dồn sang giai đoạn dự đoán (Prediction phase) — mỗi lần muốn đoán 1 điểm, nó phải chạy vòng lặp tính khoảng cách tới hàng ngàn, hàng triệu điểm cũ trong bộ nhớ.
>
> 2. **K-NN có ngoại suy (Extrapolation) được không?**  
>    **Tuyệt đối KHÔNG!** Vì giá trị dự đoán $\hat{y}$ là trung bình cộng của các điểm trong tập train, nên:
>    $$\min(Y_{\text{train}}) \le \hat{y} \le \max(Y_{\text{train}})$$
>    Nếu trong tương lai giá nhà tăng vọt lên 200 triệu/m² trong khi lịch sử quá khứ giá cao nhất chỉ 100 triệu/m², K-NN chỉ có thể đoán tối đa là 100 triệu/m². Nó không thể nhìn thấy xu hướng đi lên vượt ngoài biên dữ liệu đã có.

---

## 3. THUẬT TOÁN 2: HỒI QUY TUYẾN TÍNH (LINEAR REGRESSION)

### 3.1. Trực giác hình học: Tìm đường thẳng tối ưu
Khác với K-NN (hỏi ý kiến hàng xóm xung quanh), **Hồi quy tuyến tính** tin rằng toàn bộ thế giới dữ liệu tuân theo một quy luật đường thẳng:
$$\text{Giá nhà} = w_0 + w_1 \times \text{Diện tích}$$
Mục tiêu là cầm một cây thước kẻ, đặt lên trang giấy có các điểm dữ liệu sao cho **tổng khoảng cách thẳng đứng từ các điểm đến cây thước kẻ là nhỏ nhất**.

```
    y (GPA)
     ^
 4.0 │                         * (Điểm thực tế y_i)
     │                       / |
     │                      /  |  Sai số e_i = y_i - y^_i
 3.0 │                     /---* (Điểm dự đoán y^_i trên đường thẳng)
     │            *       /
     │           /|      /   Đường thẳng hồi quy: y^ = w0 + w1 * x
 2.0 │          /-*     /
     │         /       /
 1.0 │        *       /
     └────────┼───────┼────────────────────────> x (SAT)
             1000    1500
```

---

### 3.2. Mô hình 1 chiều: Nghiệm giải tích phương sai / hiệp phương sai
Khi dữ liệu chỉ có 1 biến đầu vào $x$ ($d = 1$), phương trình đường thẳng là:
$$\hat{y} = w_0 + w_1 x$$
Hàm mất mát bình phương tối thiểu (Ordinary Least Squares - OLS):
$$J(w_0, w_1) = \sum_{i=1}^N (y_i - \hat{y}_i)^2 = \sum_{i=1}^N (y_i - (w_0 + w_1 x_i))^2$$
Bằng cách lấy đạo hàm riêng theo $w_0$ và $w_1$ rồi cho bằng $0$, ta có công thức nghiệm giải tích đẹp đẽ:
$$w_1 = \frac{S_{XY}}{S_{XX}} = \frac{\sum_{i=1}^N (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^N (x_i - \bar{x})^2}$$
$$w_0 = \bar{y} - w_1 \bar{x}$$
Trong đó:
- $\bar{x}, \bar{y}$: Giá trị trung bình của $x$ và $y$.
- $S_{XY}$: Hiệp phương sai (Covariance) giữa $x$ và $y$ (đo mức độ cùng tăng cùng giảm).
- $S_{XX}$: Phương sai (Variance) của $x$.

---

### 3.3. Mô hình đa biến tổng quát: Normal Equation & Ma trận giả nghịch đảo

Khi bài toán có $d$ đặc trưng: $\mathbf{x} = [x_1, x_2, \dots, x_d]^T$:
$$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_d x_d$$

#### Kỹ thuật Bias Trick (Bổ sung biến giả $x_0 = 1$)
Để không phải viết riêng hệ số tự do $w_0$, ta định nghĩa:
$$\bar{\mathbf{x}} = [1, x_1, x_2, \dots, x_d]^T \in \mathbb{R}^{d+1}, \quad \mathbf{w} = [w_0, w_1, w_2, \dots, w_d]^T \in \mathbb{R}^{d+1}$$
Khi đó:
$$\hat{y} = \bar{\mathbf{x}}^T \mathbf{w} = \mathbf{w}^T \bar{\mathbf{x}}$$

Với toàn bộ tập dữ liệu gồm $N$ mẫu, ghép thành ma trận thiết kế (Design Matrix) $\bar{\mathbf{X}} \in \mathbb{R}^{N \times (d+1)}$:
$$\bar{\mathbf{X}} = \begin{bmatrix} 1 & x_{11} & x_{12} & \dots & x_{1d} \\ 1 & x_{21} & x_{22} & \dots & x_{2d} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & x_{N1} & x_{N2} & \dots & x_{Nd} \end{bmatrix}, \quad \mathbf{y} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_N \end{bmatrix}$$

Vector các giá trị dự đoán:
$$\hat{\mathbf{y}} = \bar{\mathbf{X}}\mathbf{w}$$

Hàm mất mát toàn thể:
$$J(\mathbf{w}) = \frac{1}{2N} \|\mathbf{y} - \bar{\mathbf{X}}\mathbf{w}\|_2^2 = \frac{1}{2N} (\mathbf{y} - \bar{\mathbf{X}}\mathbf{w})^T (\mathbf{y} - \bar{\mathbf{X}}\mathbf{w})$$

Lấy đạo hàm ma trận theo vector $\mathbf{w}$ và đặt bằng 0:
$$\nabla_\mathbf{w} J(\mathbf{w}) = \bar{\mathbf{X}}^T (\bar{\mathbf{X}}\mathbf{w} - \mathbf{y}) = 0 \implies (\bar{\mathbf{X}}^T \bar{\mathbf{X}}) \mathbf{w} = \bar{\mathbf{X}}^T \mathbf{y}$$

> **Phương trình pháp tuyến (Normal Equation)**:
> $$\mathbf{w} = (\bar{\mathbf{X}}^T \bar{\mathbf{X}})^{-1} \bar{\mathbf{X}}^T \mathbf{y}$$
> Cụm đại lượng $\bar{\mathbf{X}}^\dagger = (\bar{\mathbf{X}}^T \bar{\mathbf{X}})^{-1} \bar{\mathbf{X}}^T$ được gọi là **Ma trận giả nghịch đảo Moore-Penrose (Pseudo-inverse)**.

---

### 3.4. Phân rã QR để tránh lỗi ma trận kỳ dị
Trong thực tế, phép tính $(\bar{\mathbf{X}}^T \bar{\mathbf{X}})^{-1}$ trực tiếp rất nguy hiểm vì:
1. Nếu có hai cột trong dữ liệu phụ thuộc tuyến tính nhau (Đa cộng tuyến - Multicollinearity), ma trận $\bar{\mathbf{X}}^T \bar{\mathbf{X}}$ sẽ bị suy biến (kỳ dị, không có nghịch đảo).
2. Số điều kiện (Condition number) của $\bar{\mathbf{X}}^T \bar{\mathbf{X}}$ bị bình phương lên, gây sai số dấu phẩy động rất lớn trong máy tính.

**Giải pháp phân rã QR**:
Phân tích $\bar{\mathbf{X}} = \mathbf{Q} \mathbf{R}$, trong đó $\mathbf{Q}$ là ma trận trực giao ($\mathbf{Q}^T \mathbf{Q} = \mathbf{I}$) và $\mathbf{R}$ là ma trận tam giác trên khả nghịch.
Khi đó:
$$\bar{\mathbf{X}}^T \bar{\mathbf{X}} \mathbf{w} = \bar{\mathbf{X}}^T \mathbf{y} \iff (\mathbf{R}^T \mathbf{Q}^T \mathbf{Q} \mathbf{R}) \mathbf{w} = \mathbf{R}^T \mathbf{Q}^T \mathbf{y} \iff \mathbf{R}^T \mathbf{R} \mathbf{w} = \mathbf{R}^T \mathbf{Q}^T \mathbf{y}$$
Nhân hai vế với $(\mathbf{R}^T)^{-1}$:
$$\mathbf{R} \mathbf{w} = \mathbf{Q}^T \mathbf{y}$$
Vì $\mathbf{R}$ là ma trận tam giác trên, ta giải tìm $\mathbf{w}$ cực nhanh bằng phép thế ngược (Back-substitution) mà không cần nghịch đảo trực tiếp bất kỳ ma trận nào! Đây chính là cách hàm `np.linalg.lstsq` và thư viện LAPACK hoạt động bên dưới.

---

## 4. CÁC ĐỘ ĐO ĐÁNH GIÁ MÔ HÌNH HỒI QUY (EVALUATION METRICS)

Để biết mô hình dự đoán tốt hay dở, ta so sánh giá trị thực tế $y_i$ và giá trị dự đoán $\hat{y}_i$ qua 5 thước đo chuẩn mực:

### 1. MSE (Mean Squared Error - Sai số bình phương trung bình)
$$\text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$
- **Đặc trưng**: Do có lũy thừa bậc 2, MSE **phạt cực nặng các sai số lớn**. Nếu mô hình dự đoán lệch 1 đơn vị thì phạt 1, nhưng lệch 10 đơn vị thì phạt tới 100.
- **Nhược điểm**: Đơn vị đo bị bình phương lên (ví dụ giá nhà triệu USD thì đơn vị của MSE là triệu USD bình phương, khó hình dung).

### 2. RMSE (Root Mean Squared Error)
$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
- Đưa đơn vị sai số về cùng đơn vị đo thực tế của biến mục tiêu $y$.

### 3. MAE (Mean Absolute Error - Sai số tuyệt đối trung bình)
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
- **Đặc trưng**: Rất trực quan và dễ hiểu với người bình thường. Ít bị ảnh hưởng bởi một vài điểm dữ liệu dị biệt (Outliers) hơn MSE.

### 4. $R^2$ (Hệ số xác định - Coefficient of Determination)
$$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$
Trong đó:
- $\text{SS}_{\text{res}}$ (Residual Sum of Squares): Tổng bình phương sai số của mô hình.
- $\text{SS}_{\text{tot}}$ (Total Sum of Squares): Tổng bình phương sai số khi ta dùng mô hình "ngớ ngẩn nhất" — luôn luôn đoán giá trị trung bình $\bar{y}$.
- **Ý nghĩa**:
  - $R^2 = 1.0$: Mô hình hoàn hảo, đoán đúng $100\%$ mọi điểm.
  - $R^2 = 0.85$: Mô hình giải thích được $85\%$ sự biến thiên của $y$.
  - $R^2 = 0$: Mô hình chỉ ngang ngửa việc đoán giá trị trung bình.
  - $R^2 < 0$: Mô hình cực kỳ tệ, dự đoán còn sai lệch hơn cả việc đoán bừa giá trị trung bình.

### 5. Phân tích phần dư sai số (Residual Analysis)
Đặt sai số cho từng điểm là $e_i = y_i - \hat{y}_i$:
- **Kỳ vọng sai số**: $E[e] = \frac{1}{N} \sum e_i$. Với mô hình Hồi quy tuyến tính tối ưu, $E[e]$ phải xấp xỉ bằng $0$ (không bị lệch thiên vị).
- **Phương sai sai số**: $\text{Var}(e) = \frac{1}{N} \sum (e_i - E[e])^2$.

---

## 5. GIẢI PHẪU MÃ NGUỒN TỪNG DÒNG (LINE-BY-LINE CODE BREAKDOWN)

### A. Bẫy tính khoảng cách `axis` trong NumPy (CỰC KỲ DỄ BỊ LỖI)

Xem đoạn code tính khoảng cách trong bài tập K-NN:
```python
def distance(X_train, x_val):
    # X_train có kích thước: (N, d) = (số dòng, số cột)
    # x_val là một điểm kiểm thử có kích thước: (d,)
    diff = X_train - x_val  # NumPy tự động trừ từng dòng nhờ cơ chế Broadcasting
    
    # BẪY TỬ HUYỆT NẰM Ở ĐÂY:
    return np.linalg.norm(diff, ord=2, axis=1)
```

> **Giải thích tại sao `axis=1` chứ không phải `axis=0`?**
> - Ma trận `diff` có kích thước $(N, d)$ (ví dụ 100 dòng, 4 cột).
> - Ta cần tính khoảng cách giữa điểm `x_val` với **từng dòng** của `X_train`.
> - Do đó ta phải bình phương và cộng dồn theo **chiều ngang (cộng các cột lại với nhau trên cùng một dòng)** $\implies$ Trong NumPy, thao tác dọc theo các cột trên cùng 1 dòng là `axis=1`.
> - Nếu bạn ghi nhầm thành `axis=0`, NumPy sẽ cộng dồn tất cả các dòng lại theo chiều dọc, và kết quả trả về chỉ là một vector 4 phần tử vô nghĩa!

---

### B. Tìm K láng giềng và dự đoán bằng `np.argsort`
```python
# 1. Tính khoảng cách từ x_val đến toàn bộ tập train
dist_array = distance(X_train, x_val)  # Trả về mảng 1D độ dài N

# 2. Sắp xếp mảng khoảng cách
# np.argsort KHÔNG trả về giá trị khoảng cách, mà trả về MẢNG CHỈ SỐ ban đầu
sorted_indices = np.argsort(dist_array)

# 3. Lấy ra K chỉ số đầu tiên (tương ứng với K khoảng cách nhỏ nhất)
nearest_k_indices = sorted_indices[:K]

# 4. Trích xuất nhãn thực tế của K láng giềng đó và lấy trung bình
y_pred = np.mean(Y_train[nearest_k_indices])
```

---

### C. Kỹ thuật Bias Trick và giải Normal Equation bằng NumPy
```python
import numpy as np

# Giả sử X có shape (N, d) và y có shape (N,)
N = X.shape[0]

# 1. Tạo cột toàn số 1 có shape (N, 1)
ones = np.ones((N, 1))

# 2. Ghép cột số 1 vào bên trái ma trận X để tạo ma trận Xbar (N, d + 1)
Xbar = np.concatenate((ones, X), axis=1)

# 3. Tính ma trận giả nghịch đảo và tìm vector trọng số w
# Cách tính tường minh: w = (Xbar^T * Xbar)^(-1) * Xbar^T * y
w = np.linalg.pinv(Xbar).dot(y)

# 4. Trọng số bias w0 và các trọng số đặc trưng w1, w2...
w0 = w[0]
weights = w[1:]

# 5. Dự đoán cho mẫu mới x_test
# Thêm số 1 vào đầu x_test:
x_test_bar = np.insert(x_test, 0, 1)
y_hat = np.dot(x_test_bar, w)
```

---

## 6. CHI TIẾT 4 BÀI TẬP THỰC HÀNH TUẦN 1

Dưới đây là tóm tắt bản chất kỹ thuật của 4 bài tập thực hành trong thư mục `Week1/work/`:

### 1. Bài tập B.1: Tiêu thụ nhiên liệu (`fuel.txt`)
- **Dữ liệu**: Gồm 51 bang của Mỹ, theo dõi mức tiêu thụ xăng dầu dựa trên thuế nhiên liệu, thu nhập bình quân, số lượng giấy phép lái xe...
- **Thuật toán**: Áp dụng hồi quy tuyến tính đa biến để tìm bộ trọng số $\mathbf{w}$.
- **Kết quả cốt lõi**: Xác định được biến nào có trọng số dương lớn nhất (tác động làm tăng tiêu thụ xăng) và biến nào có trọng số âm (thuế cao làm giảm tiêu thụ xăng).

### 2. Bài tập B.3: Điểm SAT & GPA đại học (`SAT_GPA.csv`)
- **Dữ liệu**: 84 sinh viên với 2 cột: Điểm SAT ($x$) và điểm GPA đại học ($y$).
- **Mục tiêu**: Hồi quy tuyến tính 1 biến.
- **Công thức tính tay**: Áp dụng trực tiếp $w_1 = \frac{S_{XY}}{S_{XX}}$ và $w_0 = \bar{y} - w_1 \bar{x}$.
- **Vẽ đồ thị**: Vẽ đường thẳng hồi quy màu đỏ cắt ngang đám mây điểm dữ liệu phân tán.

### 3. Bài tập B.4: Độ dày nội trung mạc động mạch cảnh NTM (`vidu4_lin_reg.txt`)
- **Dữ liệu**: 100 bệnh nhân với các chỉ số tuổi, huyết áp, đường huyết, cholesterol để dự đoán bề dày màng mạch NTM (dấu hiệu xơ vữa động mạch).
- **So sánh**: Chạy cả K-NN Regression (với $K=3, 5, 7$) và Linear Regression.
- **Nhận xét**: So sánh MSE và $R^2$ của hai mô hình để thấy Linear Regression tổng quát hóa xu hướng toàn cục tốt hơn khi dữ liệu có nhiễu y tế.

### 4. Bài tập B.5: Giá bất động sản Tân Bắc (`real_estate.csv`)
- **Dữ liệu**: 414 giao dịch bất động sản với 6 đặc trưng: Tuổi nhà, khoảng cách đến ga tàu điện ngầm gần nhất (MRT), số lượng cửa hàng tiện lợi xung quanh, vĩ độ, kinh độ...
- **Kết quả thực nghiệm**: Khoảng cách đến ga MRT là biến quan trọng nhất (trọng số âm lớn — nhà càng xa ga tàu thì giá càng giảm mạnh).

---

## 7. BỘ 10 CÂU HỎI VẤN ĐÁP BẢO VỆ BÀI TẬP TUẦN 1 KÈM ĐÁP ÁN CHUẨN

1. **Câu 1: Điểm khác biệt căn bản nhất giữa Hồi quy (Regression) và Phân loại (Classification) là gì?**  
   *Đáp*: Là dạng của biến mục tiêu đầu ra $y$. Trong Hồi quy, $y$ là số thực liên tục ($y \in \mathbb{R}$ như giá tiền, điểm số). Trong Phân loại, $y$ là nhãn rời rạc thuộc các nhóm riêng biệt ($y \in \{0, 1\}$ hoặc các lớp).

2. **Câu 2: Tại sao K-NN lại được gọi là mô hình "Phi tham số" (Non-parametric)?**  
   *Đáp*: "Phi tham số" không có nghĩa là không có tham số nào, mà có nghĩa là mô hình không giả định trước một cấu trúc hàm số toán học cố định với số lượng tham số cố định độc lập với kích thước dữ liệu. Số lượng "tham số" ngầm định của K-NN chính là toàn bộ các điểm dữ liệu huấn luyện.

3. **Câu 3: Chọn tham số $K$ trong K-NN như thế nào? $K=1$ và $K=N$ sẽ dẫn đến hiện tượng gì?**  
   *Đáp*: 
   - $K=1$: Mô hình chỉ nhìn đúng 1 láng giềng gần nhất. Ranh giới cực kỳ ngoằn ngoèo, bắt chước cả các điểm nhiễu $\implies$ **Overfitting (Quá khớp)**.
   - $K=N$: Mô hình lấy trung bình cộng của toàn bộ tập dữ liệu huấn luyện, dự đoán ra một đường nằm ngang bất động $\implies$ **Underfitting (Chưa khớp)**.
   - Giá trị $K$ tối ưu thường được tìm bằng kỹ thuật Cross-Validation.

4. **Câu 4: K-NN có khả năng ngoại suy (Extrapolation) không? Cho ví dụ.**  
   *Đáp*: Không bao giờ. Giá trị dự đoán của K-NN luôn bị chặn trong đoạn $[\min(Y_{\text{train}}), \max(Y_{\text{train}})]$. Nếu tập train có mức lương cao nhất là 50 triệu, thì dù một ứng viên sau này giỏi gấp 10 lần, K-NN cũng chỉ có thể dự đoán tối đa là 50 triệu.

5. **Câu 5: Tại sao trong Hồi quy tuyến tính ta phải thêm cột số 1 vào ma trận thiết kế $\bar{\mathbf{X}}$?**  
   *Đáp*: Để biểu diễn hệ số chệch $w_0$ (bias / intercept). Việc bổ sung biến giả $x_0 = 1$ giúp gộp $w_0$ vào chung vector trọng số $\mathbf{w}$, chuyển phương trình từ $\hat{y} = w_0 + \sum w_j x_j$ thành phép nhân vô hướng ngắn gọn $\hat{y} = \bar{\mathbf{x}}^T \mathbf{w}$.

6. **Câu 6: Phương trình pháp tuyến (Normal Equation) là gì? Công thức tính $\mathbf{w}$ ra sao?**  
   *Đáp*: Là phương trình tìm điểm cực trị của hàm mất mát bình phương sai số: $\mathbf{w} = (\bar{\mathbf{X}}^T \bar{\mathbf{X}})^{-1} \bar{\mathbf{X}}^T \mathbf{y}$.

7. **Câu 7: Khi nào ma trận $\bar{\mathbf{X}}^T \bar{\mathbf{X}}$ không nghịch đảo được? Cách giải quyết trong code là gì?**  
   *Đáp*: Khi xảy ra hiện tượng Đa cộng tuyến hoàn hảo (các cột trong $X$ phụ thuộc tuyến tính lẫn nhau) hoặc khi số thuộc tính $d$ lớn hơn số mẫu $N$. Trong code, ta giải quyết bằng cách dùng hàm ma trận giả nghịch đảo `np.linalg.pinv` (dựa trên phân rã SVD) hoặc phân rã QR (`np.linalg.lstsq`).

8. **Câu 8: Hệ số xác định $R^2$ có thể âm không? Nếu $R^2 = -0.2$ thì có ý nghĩa gì?**  
   *Đáp*: Hoàn toàn có thể âm trên tập dữ liệu kiểm thử (Test set). $R^2 < 0$ có nghĩa là mô hình dự đoán cực kỳ tệ, sai số của nó thậm chí còn lớn hơn cả sai số khi ta chỉ đơn giản lấy giá trị trung bình $\bar{y}$ để dự đoán cho mọi mẫu.

9. **Câu 9: So sánh ưu nhược điểm của MAE và MSE.**  
   *Đáp*: 
   - MSE phạt rất nặng sai số lớn do có bình phương, giúp mô hình cố gắng triệt tiêu những sai lệch nghiêm trọng, nhưng rất nhạy cảm và dễ bị kéo lệch bởi các điểm dị biệt (Outliers).
   - MAE phản ánh sai số trung bình tuyến tính, cùng đơn vị với $y$, ít bị ảnh hưởng bởi Outliers, nhưng đạo hàm tại điểm 0 không liên tục.

10. **Câu 10: Nếu dữ liệu nhà đất có diện tích từ 50-300 m² và số nhà vệ sinh từ 1-4, chuyện gì xảy ra nếu quên chuẩn hóa dữ liệu trước khi chạy K-NN?**  
    *Đáp*: Thuộc tính diện tích với giá trị hàng trăm sẽ chi phối gần như $100\%$ công thức khoảng cách Euclid. Số nhà vệ sinh chỉ chênh nhau 1-2 đơn vị nên bình phương lên chỉ là 1-4, hoàn toàn bị triệt tiêu trước đại lượng diện tích bình phương hàng chục ngàn. Mô hình sẽ coi như số nhà vệ sinh không hề tồn tại.
