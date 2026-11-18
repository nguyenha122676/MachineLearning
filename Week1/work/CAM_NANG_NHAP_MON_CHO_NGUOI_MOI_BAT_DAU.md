# CẨM NANG TOÀN DIỆN MACHINE LEARNING CHO NGƯỜI MỚI BẮT ĐẦU (ZERO KNOWLEDGE)
## Hướng Dẫn & Giải Thích Chi Tiết TẤT CẢ Ví Dụ (A.1, A.2, A.3, B.1, B.2) và Bài Tập (B.1, B.3, B.4, B.5)

> **Lời nhắn nhủ gửi bạn**: Đừng lo lắng nếu bạn chưa từng học Machine Learning hay cảm thấy sợ các công thức toán học! Cẩm nang này được viết bằng **ngôn ngữ đời thường, hình tượng hóa dễ hiểu, giải thích từng dòng code và chỉ rõ từng con số kết quả nghĩa là gì**. Bạn chỉ cần đọc cẩm nang này là sẽ nắm trọn vẹn kiến thức của bài thực hành số 2.

---

## MỤC LỤC
1. [Khái niệm cốt lõi bằng ngôn ngữ đời thường](#1-khái-niệm-cốt-lõi-bằng-ngôn-ngữ-đời-thường)
2. [Hiểu bản chất 2 thuật toán: K-NN và Hồi quy tuyến tính](#2-hiểu-bản-chất-2-thuật-toán-k-nn-và-hồi-quy-tuyến-tính)
3. [Làm sao biết máy đoán đúng hay sai? (MAE, MSE, R2, RSS)](#3-làm-sao-biết-máy-đoán-đúng-hay-sai)
4. [Hai cái bẫy kỹ thuật trong tài liệu cần lưu ý](#4-hai-cái-bẫy-kỹ-thuật-trong-tài-liệu-cần-lưu-ý)
5. [PHẦN 1: GIẢI THÍCH CHI TIẾT TỪNG VÍ DỤ](#5-phần-1-giải-thích-chi-tiết-từng-ví-dụ)
   - [5.1. Ví dụ A.1: K-NN dự đoán điểm GPA từ điểm SAT (`vidu_A1.py`)](#51-ví-dụ-a1-k-nn-dự-đoán-điểm-gpa-từ-điểm-sat)
   - [5.2. Ví dụ A.2: K-NN dự đoán Áp suất khí quyển theo Nhiệt độ sôi (`vidu_A2.py`)](#52-ví-dụ-a2-k-nn-dự-đoán-áp-suất-khí-quyển-theo-nhiệt-độ-sôi)
   - [5.3. Ví dụ A.3: K-NN đa chiều dự đoán mức tiêu thụ xăng (`vidu_A3.py`)](#53-ví-dụ-a3-k-nn-đa-chiều-dự-đoán-mức-tiêu-thụ-xăng)
   - [5.4. Ví dụ B.1: Hồi quy đơn biến Nhiệt độ & Áp suất (`vidu_B1.py`)](#54-ví-dụ-b1-hồi-quy-đơn-biến-nhiệt-độ--áp-suất)
   - [5.5. Ví dụ B.2: Hồi quy đa biến Xăng bằng QR Householder & Scikit-Learn (`vidu_B2.py`)](#55-ví-dụ-b2-hồi-quy-đa-biến-xăng-bằng-qr-householder--scikit-learn)
6. [PHẦN 2: GIẢI THÍCH CHI TIẾT TỪNG BÀI TẬP BẮT BUỘC](#6-phần-2-giải-thích-chi-tiết-từng-bài-tập-bắt-buộc)
   - [6.1. Bài tập B.1: Đánh giá Hồi quy tiêu thụ Xăng trên Train & Validation (`bai_tap_B1.py`)](#61-bài-tập-b1-đánh-giá-hồi-quy-tiêu-thụ-xăng-trên-train--validation)
   - [6.2. Bài tập B.3: Hồi quy SAT_GPA, Pearson correlation, so sánh K-NN (`bai_tap_B3.py`)](#62-bài-tập-b3-hồi-quy-sat_gpa-pearson-correlation-so-sánh-k-nn)
   - [6.3. Bài tập B.4: Đoán độ dày thành mạch máu NTM, kỳ vọng & phương sai (`bai_tap_B4.py`)](#63-bài-tập-b4-đoán-độ-dày-thành-mạch-máu-ntm-kỳ-vọng--phương-sai)
   - [6.4. Bài tập B.5: Dự đoán giá BĐS, so sánh RSS giữa LR và K-NN (`bai_tap_B5.py`)](#64-bài-tập-b5-dự-đoán-giá-bđs-so-sánh-rss-giữa-lr-và-k-nn)
7. [Hướng dẫn thao tác chạy code bằng Menu `main.py`](#7-hướng-dẫn-thao-tác-chạy-code-bằng-menu-mainpy)
8. [Bộ câu hỏi "cứu cánh" khi thầy cô vấn đáp](#8-bộ-câu-hỏi-cứu-cánh-khi-thầy-cô-vấn-đáp)

---

## 1. Khái niệm cốt lõi bằng ngôn ngữ đời thường

### 1.1. Machine Learning (Học máy) là gì?
- **Lập trình truyền thống**: Con người tự viết ra công thức (ví dụ: `Tiền_xăng = Quãng_đường * 0.05`).
- **Machine Learning**: Con người đưa cho máy tính một tập dữ liệu quá khứ (ví dụ: thông tin 100 chuyến đi kèm tiền xăng thực tế). Máy tính tự "ngâm cứu" dữ liệu và **tự tìm ra quy luật**. Sau này khi có một chuyến đi mới toanh, máy tính sẽ tự đoán xem tốn bao nhiêu tiền xăng.

### 1.2. Hồi quy (Regression) là gì?
- Machine Learning có 2 nhiệm vụ chính:
  1. **Phân loại (Classification)**: Đoán nhãn (Chó hay Mèo? Đậu hay Rớt?).
  2. **Hồi quy (Regression)**: **Đoán một con số liên tục** (Đoán điểm GPA là bao nhiêu? Đoán giá nhà là mấy tỷ? Đoán áp suất là bao nhiêu atm?).
- 👉 **Toàn bộ bài thực hành này là bài toán Hồi quy (Regression)**.

### 1.3. Tập Train (Học) và Tập Validation/Test (Thi) là gì?
- **Tập Train (Training set)**: Giống như **Sách bài tập có sẵn đáp án giải chi tiết** ở nhà. Máy tính học thuộc quy luật từ tập này.
- **Tập Validation / Test set**: Giống như **Đề thi thật trong phòng thi**. Máy tính chưa từng thấy các câu hỏi này. Ta giấu đáp án đi, bắt máy đoán, rồi chấm điểm xem máy đoán đúng hay sai.

---

## 2. Hiểu bản chất 2 thuật toán: K-NN và Hồi quy tuyến tính

### Thuật toán 1: K-NN (K-Nearest Neighbors - K người hàng xóm gần nhất)
- **Ý tưởng**: *"Hãy cho tôi biết 5 người bạn thân của bạn, tôi sẽ nói bạn là ai"*.
- **Cách làm**:
  1. Khi cần đoán giá trị cho một mẫu mới $x$, máy tính đo khoảng cách từ $x$ đến tất cả các mẫu cũ đã học.
  2. Chọn ra **$K$ mẫu gần nhất** (ví dụ $K = 4$ hay $K = 8$).
  3. Lấy **trung bình cộng** đáp án của $K$ mẫu đó $\to$ Đó chính là kết quả dự đoán $\hat{y}$!
- **Điểm yếu của K-NN**: K-NN **chỉ biết nội suy, không biết ngoại suy**! Nó chỉ biết quanh quẩn trong những con số nó từng thấy, không thể đoán được các giá trị vượt ngưỡng ngoài tầm mắt.

### Thuật toán 2: Hồi quy tuyến tính (Linear Regression)
- **Ý tưởng**: Đặt một cây thước kẻ lên trang giấy vẽ các điểm dữ liệu sao cho cây thước **đi gần nhất với tất cả các điểm**.
- Phương trình đường thẳng là:
  $$\hat{y} = w_0 + w_1 \cdot x$$
  *(Hồi quy đơn biến: 1 đầu vào $x$)*
- Nếu có nhiều yếu tố đầu vào $x_1, x_2, \dots, x_d$ (Hồi quy đa biến):
  $$\hat{y} = w_0 + w_1 \cdot x_1 + w_2 \cdot x_2 + \dots + w_d \cdot x_d$$
- **Điểm mạnh của Hồi quy tuyến tính**: Rất nhanh, biết rõ yếu tố nào quan trọng hơn nhờ nhìn vào trọng số $w$, và **có thể ngoại suy** (kéo dài đường thẳng ra vô tận để đoán các trường hợp cực lớn hoặc cực nhỏ).

---

## 3. Làm sao biết máy đoán đúng hay sai?

Khi máy đoán ra giá trị $\hat{y}$ (y_pred), ta lấy ra so với giá trị thật $y$ (y_true):

1. **MAE (Mean Absolute Error - Sai số tuyệt đối trung bình)**:
   - *"Trung bình ra, máy đoán lệch bao nhiêu đơn vị so với thực tế?"*
   - Công thức: $\text{MAE} = \frac{1}{n}\sum |y_i - \hat{y}_i|$.
   - 👉 **MAE càng bé thì càng tốt!**
2. **MSE (Mean Squared Error - Sai số bình phương trung bình)**:
   - Bình phương độ lệch lên rồi mới chia trung bình: $\text{MSE} = \frac{1}{n}\sum (y_i - \hat{y}_i)^2$.
   - **Tác dụng**: Phạt cực nặng những cú đoán sai quá xa.
   - 👉 **MSE càng bé thì càng tốt!**
3. **$R^2$ (R-squared - Hệ số xác định)**:
   - Điểm số chấm cho mô hình trên thang điểm 100% (từ 0.0 đến 1.0).
   - $R^2 = 0.6$ nghĩa là mô hình giải thích được 60% biến thiên của dữ liệu.
   - **$R^2$ có âm không?**: Có! Khi mô hình đoán quá tệ (tệ hơn cả việc bạn đoán bừa bằng giá trị trung bình), $R^2$ sẽ mang giá trị âm.
4. **RSS (Residual Sum of Squares - Tổng bình phương phần dư)**:
   - $\text{RSS} = \sum (y_i - \hat{y}_i)^2 = n \times \text{MSE}$. Đây chính là tổng số điểm phạt sai số của mô hình.

---

## 4. Hai cái bẫy kỹ thuật trong tài liệu cần lưu ý

1. **Cái bẫy hàm khoảng cách trong PDF (trang 5)**:
   - Code mẫu ghi: `np.linalg.norm(array - value, ord=2, axis=0)` kèm nhắc nhở: *(Cần tự xem xét axis = 0 hay bằng 1!)*.
   - 👉 **Đáp án chuẩn**: Bắt buộc phải là **`axis=1`**! Vì `array` có $N$ dòng, ta cần tính chuẩn vector theo chiều ngang cho từng dòng (từng mẫu dữ liệu) chứ không phải tính dọc theo cột.
2. **Cái bẫy file điểm `SAT_GPA.csv`**:
   - File này được xếp thứ tự tăng dần theo điểm GPA.
   - 64 người đầu toàn điểm trung bình ($2.40 \to 3.50$).
   - 20 người cuối làm bài thi (Validation) toàn học sinh giỏi xuất sắc ($3.51 \to 3.81$).
   - Điều này làm cho $R^2$ trên tập Validation bị âm. K-NN thất bại vì không thể ngoại suy vượt quá $3.50$, trong khi Linear Regression vẫn vươn lên đoán được điểm cao.

---

## 5. PHẦN 1: GIẢI THÍCH CHI TIẾT TỪNG VÍ DỤ

---

### 5.1. Ví dụ A.1: K-NN dự đoán điểm GPA từ điểm SAT
- **File chạy**: [vidu_A1.py](file:///d:/MachineLearning/Week1/work/vidu_A1.py)
- **Tập dữ liệu**: `SAT_GPA.csv` (84 sinh viên).
- **Mục tiêu**: Lấy điểm thi cấp 3 (SAT) để đoán điểm đại học (GPA) bằng thuật toán K-NN ($k=8$).

#### Giải thích từng đoạn code:
1. **Đọc dữ liệu**:
   ```python
   data = pd.read_csv("SAT_GPA.csv")
   x = data['SAT'].values # Cột điểm SAT đầu vào
   y = data['GPA'].values # Cột điểm GPA đầu ra cần đoán
   ```
2. **Chia 64 người đầu học (Train), 20 người sau thi (Test)**:
   ```python
   X_train, Y_train = x[:64], y[:64]
   X_test, Y_test = x[64:], y[64:]
   ```
3. **Hàm tính khoảng cách 1 chiều và tìm 8 người gần nhất**:
   ```python
   def distance(array, value):
       return abs(array - value) # Khoảng cách giữa 2 điểm là trị tuyệt đối

   def find_nearest_index(array, value, k):
       array_D = distance(array, value)
       return np.argsort(array_D)[:k] # argsort xếp từ nhỏ đến lớn, lấy k phần tử đầu
   ```
4. **Dự đoán và tính điểm phạt**:
   - Với mỗi sinh viên trong 20 người kiểm tra, tìm 8 sinh viên trong tập học có điểm SAT gần nhất với bạn này, rồi lấy trung bình điểm GPA của 8 bạn đó làm kết quả dự đoán.

#### Kết quả số liệu:
- **MSE**: `0.081378`
- **MAE**: `0.271188` (Trung bình K-NN đoán lệch khoảng $0.27$ điểm GPA).
- **$R^2$**: `-10.9515` (Bị âm vì 20 sinh viên này toàn điểm giỏi $3.51 \to 3.81$, K-NN chỉ học điểm tối đa 3.50 nên đoán hụt).
- Đồ thị minh họa được lưu tại [hinh_A1_knn_sat_gpa.png](file:///d:/MachineLearning/Week1/work/hinh_A1_knn_sat_gpa.png).

---

### 5.2. Ví dụ A.2: K-NN dự đoán Áp suất khí quyển theo Nhiệt độ sôi
- **File chạy**: [vidu_A2.py](file:///d:/MachineLearning/Week1/work/vidu_A2.py)
- **Ý nghĩa thực tế**: Ở các độ cao khác nhau (trên núi cao hay dưới đồng bằng), áp suất khí quyển thay đổi làm nước sôi ở nhiệt độ khác nhau. Ví dụ này dùng 17 lần thí nghiệm đo nhiệt độ sôi ($X$ - độ F) để ước lượng áp suất ($y$ - atm).
- **Cách chia**: 16 mẫu đầu làm Train, **mẫu thứ 17** ($212.2^\circ\text{F}$) làm mẫu kiểm thử (Validation).
- **Thuật toán**: K-NN với $k = 4$.

#### Diễn giải cách K-NN suy luận:
- Điểm kiểm thử là $212.2^\circ\text{F}$. Áp suất thực tế đo được là **`30.06 atm`**.
- Thuật toán tìm trong 16 mẫu trước đó xem 4 mẫu nào có nhiệt độ sôi gần $212.2^\circ\text{F}$ nhất:
  - 4 điểm gần nhất là: $211.9^\circ\text{F}$ (áp suất 29.88), $210.7^\circ\text{F}$ (29.04), $209.5^\circ\text{F}$ (28.49), và $208.6^\circ\text{F}$ (27.76).
- **K-NN tính trung bình**:
  $$\hat{y} = \frac{29.88 + 29.04 + 28.49 + 27.76}{4} = \mathbf{28.7925 \text{ atm}}$$
- **Đánh giá sai số**:
  - Sai số tuyệt đối $\text{MAE} = |30.06 - 28.7925| = \mathbf{1.2675 \text{ atm}}$.
  - Sai số bình phương $\text{MSE} = (1.2675)^2 = \mathbf{1.6066}$.
- Đồ thị trực quan được lưu tại [hinh_A2_knn_temp_press.png](file:///d:/MachineLearning/Week1/work/hinh_A2_knn_temp_press.png).

---

### 5.3. Ví dụ A.3: K-NN đa chiều dự đoán mức tiêu thụ xăng
- **File chạy**: [vidu_A3.py](file:///d:/MachineLearning/Week1/work/vidu_A3.py)
- **Tập dữ liệu**: `fuel.txt` (51 bang tại Mỹ).
- **Bài toán**: Đoán lượng xăng tiêu thụ (`Fuel`) dựa vào **4 yếu tố cùng lúc**:
  1. `Tax` (Thuế xăng)
  2. `Dlic` (Số bằng lái trên 1000 dân)
  3. `Income` (Thu nhập)
  4. `logMiles` (Logarit chiều dài đường cao tốc)
- **Cách chia**: 40 bang đầu làm Train, 11 bang sau làm Validation.
- **Chọn $k = 6$**: Tìm 6 bang có đặc điểm giống nhất.

#### Cách tính khoảng cách 4 chiều:
- Với 1 chiều, khoảng cách là $|x - x_i|$.
- Với 4 chiều, dùng định lý Pytago mở rộng (khoảng cách Euclid):
  $$d = \sqrt{(Tax - Tax_i)^2 + (Dlic - Dlic_i)^2 + (Income - Income_i)^2 + (logMiles - logMiles_i)^2}$$
- Trong code Numpy, ta dùng lệnh:
  ```python
  np.linalg.norm(array - value, ord=2, axis=1) # Nhớ dùng axis=1!
  ```
#### Kết quả trên 11 bang kiểm tra:
- **MSE**: `6956.36`
- **MAE**: `65.32` (Trung bình đoán lệch khoảng 65.3 đơn vị xăng trên 1000 dân).
- **$R^2$**: `-0.1394`.

---

### 5.4. Ví dụ B.1: Hồi quy đơn biến Nhiệt độ & Áp suất
- **File chạy**: [vidu_B1.py](file:///d:/MachineLearning/Week1/work/vidu_B1.py)
- Lấy lại bài toán nhiệt độ sôi & áp suất ở Ví dụ A.2, nhưng lần này **kẻ một đường thẳng hồi quy** thay vì dùng K-NN.

#### a) Tính bằng công thức toán tay (Giải tích):
- Tính trung bình của Nhiệt độ: $\bar{x} = 202.9529^\circ\text{F}$
- Tính trung bình của Áp suất: $\bar{y} = 25.0588\text{ atm}$
- Tính độ phân tán $S_{XX} = 530.7824$ và hiệp phương sai $S_{XY} = 277.5421$.
- **Hệ số góc (độ dốc của đường thẳng)**:
  $$\theta_1 = \frac{S_{XY}}{S_{XX}} = \frac{277.5420588}{530.7823529} = \mathbf{0.5228924}$$
- **Hệ số chặn (điểm cắt trục tung)**:
  $$\theta_0 = \bar{y} - \theta_1 \cdot \bar{x} = 25.0588 - 0.5228924 \times 202.9529 = \mathbf{-81.0637}$$
- 👉 **Phương trình đường hồi quy**:
  $$\text{Áp suất} = -81.0637 + 0.5229 \times \text{Nhiệt độ}$$

#### b) Giải bằng Python bằng ma trận giả nghịch đảo (`np.linalg.pinv`):
- Để máy tính tìm được cả $\theta_0$ và $\theta_1$, người ta gắn thêm một cột toàn số 1 vào ma trận $X$ (gọi là `Xbar`).
- Sau đó áp dụng công thức ma trận: $\mathbf{w} = (\bar{\mathbf{X}}^T\bar{\mathbf{X}})^{-1} \bar{\mathbf{X}}^T\mathbf{y}$.
- Lệnh Python:
  ```python
  A = np.dot(Xbar.T, Xbar)
  b = np.dot(Xbar.T, y)
  w = np.dot(np.linalg.pinv(A), b)
  ```
- Kết quả in ra màn hình: $w_0 = -81.0637$, $w_1 = 0.5229$ **hoàn toàn trùng khớp 100% với công thức tính tay**!
- Đồ thị đường thẳng hồi quy đi xuyên qua các chấm đỏ được lưu tại [hinh_B1_lin_reg_temp_press.png](file:///d:/MachineLearning/Week1/work/hinh_B1_lin_reg_temp_press.png).

---

### 5.5. Ví dụ B.2: Hồi quy đa biến Xăng bằng QR Householder & Scikit-Learn
- **File chạy**: [vidu_B2.py](file:///d:/MachineLearning/Week1/work/vidu_B2.py)
- **Đề bài**: Tìm mối liên hệ của mức tiêu thụ xăng với 4 yếu tố trên toàn bộ 51 bang:
  $$\text{Fuel} = \theta_0 + \theta_1\text{Tax} + \theta_2\text{Dlic} + \theta_3\text{Income} + \theta_4\text{logMiles}$$
- Tài liệu đưa ra 2 cách giải để sinh viên so sánh:
  - **Cách 1**: Tự viết giải thuật phân rã ma trận **QR Householder** (xoay trục tọa độ để đưa ma trận về dạng tam giác, giúp máy tính tính nghịch đảo mà không bị lỗi số học).
  - **Cách 2**: Dùng thư viện có sẵn `scikit-learn`: `LinearRegression(fit_intercept=False)`.

#### Kết quả đối chiếu giữa 2 cách:
| Hệ số | Thuật toán tự viết (QR Householder) | Thư viện Scikit-Learn | Ý nghĩa thực tế |
| :--- | :---: | :---: | :--- |
| **Intercept ($w_0$)** | **154.192845** | **154.192845** | Hệ số chặn cơ bản |
| **Tax ($w_1$)** | **-4.227983** | **-4.227983** | Thuế xăng tăng 1 cent thì tiêu thụ giảm 4.23 đơn vị |
| **Dlic ($w_2$)** | **0.471871** | **0.471871** | Tỷ lệ người có bằng lái tăng thì tiêu thụ xăng tăng |
| **Income ($w_3$)** | **-0.006135** | **-0.006135** | Thu nhập ảnh hưởng nhẹ |
| **LogMiles ($w_4$)** | **18.545275** | **18.545275** | Đường cao tốc càng dài thì xe chạy càng tốn xăng |

👉 **Kết luận**: Hai phương pháp cho ra **kết quả giống nhau đến từng chữ số thập phân**, chứng minh thuật toán tự viết hoạt động hoàn hảo.

---

## 6. PHẦN 2: GIẢI THÍCH CHI TIẾT TỪNG BÀI TẬP BẮT BUỘC

---

### 6.1. Bài tập B.1: Đánh giá Hồi quy tiêu thụ Xăng trên Train & Validation
- **File chạy**: [bai_tap_B1.py](file:///d:/MachineLearning/Week1/work/bai_tap_B1.py)
- **Đề bài**: Lấy 40 bang đầu học (Train), 11 bang sau kiểm tra (Validation). Đánh giá bằng MSE, MAE, R-square trên cả 2 tập.
- **Phương trình tìm được từ 40 bang**:
  $$\text{Fuel} = 135.1420 - 1.9030 \cdot \text{Tax} + 0.4186 \cdot \text{Dlic} - 0.0062 \cdot \text{Income} + 19.5939 \cdot \text{logMiles}$$
- **Bảng điểm đánh giá**:
  - **Tập Train (40 mẫu)**: $\text{MSE} = \mathbf{3344.79}$, $\text{MAE} = \mathbf{44.21}$, $R^2 = \mathbf{0.5516}$ (mô hình học giải thích được 55.2% dữ liệu).
  - **Tập Validation (11 mẫu)**: $\text{MSE} = \mathbf{6253.36}$, $\text{MAE} = \mathbf{60.62}$, $R^2 = \mathbf{-0.0242}$.
  - *So sánh với K-NN ở Ví dụ A.3*: K-NN có $\text{MSE} = 6956.36$ và $\text{MAE} = 65.32$. Như vậy **Hồi quy tuyến tính cho kết quả tốt hơn K-NN** trên dữ liệu tiêu thụ nhiên liệu.

---

### 6.2. Bài tập B.3: Hồi quy SAT_GPA, Pearson correlation, so sánh K-NN
- **File chạy**: [bai_tap_B3.py](file:///d:/MachineLearning/Week1/work/bai_tap_B3.py)
- **Đề bài**: 
  1. Tính tương quan Pearson giữa SAT và GPA.
  2. Chia 64 train, 20 validation. Lập phương trình hồi quy và vẽ hình.
  3. Đánh giá MSE, MAE, R2 trên validation và so sánh với K-NN ($k=8$) ở Ví dụ A.1.

#### Kết quả:
1. **Hệ số Pearson**: $r \approx \mathbf{0.63718}$ (tương quan thuận khá chặt chẽ).
2. **Phương trình hồi quy**:
   $$\text{GPA} = 0.838210 + 0.001318 \times \text{SAT}$$
3. **Bảng so sánh đối đầu giữa Linear Regression và K-NN**:

| Thước đo | Hồi quy tuyến tính (Linear Regression) | K-NN ($k=8$) | Nhận xét |
| :--- | :---: | :---: | :--- |
| **MSE** | **0.072638** | **0.081378** | Hồi quy tuyến tính sai số thấp hơn |
| **MAE** | **0.248804** | **0.271188** | Hồi quy tuyến tính đoán sát hơn |
| **$R^2$** | **-9.667975** | **-10.951519** | Hồi quy tuyến tính đỡ tệ hơn |

- **Tại sao Hồi quy tuyến tính thắng?**: Do 20 sinh viên tập thi toàn người điểm cao ($3.51 \to 3.81$). K-NN chỉ học điểm tối đa 3.50 nên không thể đoán điểm cao hơn (không biết ngoại suy). Ngược lại, Hồi quy tuyến tính có độ dốc $w_1 = 0.0013$ nên hễ điểm SAT cao là đường thẳng tự động vươn lên đoán được điểm $3.6 - 3.7$.
- Đồ thị so sánh được lưu tại [hinh_B3_sat_gpa_regression.png](file:///d:/MachineLearning/Week1/work/hinh_B3_sat_gpa_regression.png).

---

### 6.3. Bài tập B.4: Đoán độ dày thành mạch máu NTM, kỳ vọng & phương sai
- **File chạy**: [bai_tap_B4.py](file:///d:/MachineLearning/Week1/work/bai_tap_B4.py)
- **Dữ liệu**: `vidu4_lin_reg.txt` (100 bệnh nhân).
- **Yếu tố ảnh hưởng**: Tuổi, Cholesterol, Glucose (đường huyết), HA (huyết áp), BMI.

#### Câu a: Hệ số hồi quy với toàn bộ 100 bệnh nhân:
$$\text{Bề dày NTM} = -0.1080 + 0.0099 \cdot \text{TUOI} + 0.0385 \cdot \text{CHOLESTEROL} + 0.0077 \cdot \text{GLUCOSE} + 0.0022 \cdot \text{HA} - 0.0002 \cdot \text{BMI}$$
- *Nhận xét y khoa*: `CHOLESTEROL` có hệ số dương lớn nhất ($0.0385$), cho thấy mỡ trong máu là yếu tố nguy cơ hàng đầu làm xơ vữa và dày thành mạch máu!

#### Câu b: Huấn luyện trên 80 người đầu, kiểm tra trên 20 người sau:
- Tính sai số từng người: $e_i = y_i - \hat{y}_i$.
- **Kỳ vọng sai số $E[e]$ (Độ lệch trung bình)**: **`0.079989`** $\approx 0.08$ mm (rất gần 0, cho thấy mô hình không bị lệch hệ thống).
- **Phương sai sai số $\text{Var}(e)$ (Độ phân tán của sai số)**: **`0.221003`**.
- **Độ lệch chuẩn sai số $\text{Std}(e)$**: **`0.470109`**.

---

### 6.4. Bài tập B.5: Dự đoán giá BĐS, so sánh RSS giữa LR và K-NN
- **File chạy**: [bai_tap_B5.py](file:///d:/MachineLearning/Week1/work/bai_tap_B5.py)
- **Dữ liệu**: `real_estate.csv` (414 căn nhà).
- **Cách chia**: 350 căn đầu làm Train, 64 căn sau làm Validation.
- **So sánh**: Hồi quy tuyến tính vs K-NN ($K = 18$).

#### Bảng kết quả so sánh:
| Thuật toán | Tổng bình phương sai số (RSS) | Sai số trung bình (MAE) | Kết luận |
| :--- | :---: | :---: | :--- |
| **Linear Regression (Hồi quy tuyến tính)** | **4228.1469** | 6.1255 | Kém hơn |
| **K-NN ($K = 18$)** | **2809.2935** | **4.7818** | 🏆 **THẮNG ÁP ĐẢO** |

#### Tại sao ở bài này K-NN lại thắng áp đảo Hồi quy tuyến tính?
- **Bản chất bất động sản**: Giá nhà đất phụ thuộc mạnh vào **vị trí địa lý cục bộ** (cùng một con phố, cùng gần một ga tàu điện ngầm thì giá sẽ tương đồng nhau).
- Thuật toán **K-NN** tìm 18 căn nhà láng giềng gần nhất để lấy trung bình giá $\to$ Nắm bắt cực kỳ chuẩn các khu vực giá nhà cục bộ (tính phi tuyến không gian).
- Trong khi đó, **Hồi quy tuyến tính** cố tình ép toàn bộ thành phố theo một siêu phẳng phẳng lì, nên dự đoán bị sai lệch lớn ở các khu phố đắt đỏ đột biến.
- Đồ thị so sánh được lưu tại [hinh_B5_real_estate_comparison.png](file:///d:/MachineLearning/Week1/work/hinh_B5_real_estate_comparison.png).

---

## 7. Hướng dẫn thao tác chạy code bằng Menu `main.py`

Để không bao giờ bị gõ nhầm hay bị kẹt trong màn hình `>>>` của Python:

1. Mở Terminal trong VS Code (hoặc PowerShell).
2. Gõ lệnh:
   ```powershell
   python d:\MachineLearning\Week1\work\main.py
   ```
3. Màn hình sẽ hiện Menu:
   - Gõ `1` $\to$ Chạy Ví dụ A.1
   - Gõ `2` $\to$ Chạy Ví dụ A.2
   - Gõ `3` $\to$ Chạy Ví dụ A.3
   - Gõ `4` $\to$ Chạy Ví dụ B.1
   - Gõ `5` $\to$ Chạy Ví dụ B.2
   - Gõ `6` $\to$ Chạy Bài tập B.1
   - Gõ `7` $\to$ Chạy Bài tập B.3
   - Gõ `8` $\to$ Chạy Bài tập B.4
   - Gõ `9` $\to$ Chạy Bài tập B.5
   - Gõ `0` $\to$ Chạy toàn bộ từ đầu đến cuối!

---

## 8. Bộ câu hỏi "cứu cánh" khi thầy cô vấn đáp

**1. "K-NN hoạt động như thế nào?"**
> *"Dạ thưa thầy, K-NN tính khoảng cách từ mẫu cần dự đoán đến các mẫu trong tập huấn luyện, tìm ra K mẫu gần nhất và lấy trung bình cộng đầu ra của K mẫu đó làm kết quả."*

**2. "Tại sao ở bài SAT_GPA thì Linear Regression tốt hơn, nhưng ở bài Bất động sản K-NN lại tốt hơn?"**
> *"Dạ thưa thầy: Ở bài SAT_GPA dữ liệu mang tính tuyến tính đơn giản, tập test toàn điểm cao ngoài dải tập train nên Linear Regression thắng nhờ khả năng **ngoại suy** theo độ dốc, còn K-NN chỉ biết **nội suy** nên bị giới hạn trần. Ngược lại ở bài Bất động sản, giá nhà mang tính **cục bộ địa lý phi tuyến**, K-NN lấy trung bình 18 căn nhà láng giềng nên dự đoán sát thực tế khu vực hơn là một siêu phẳng toàn cục của Linear Regression."*

**3. "Tại sao trong hàm khoảng cách đa chiều phải dùng `axis=1`?"**
> *"Dạ thưa thầy, vì ma trận hiệu có kích thước (N, d). Ta cần tính chuẩn Euclid gom theo chiều ngang của từng mẫu dữ liệu (từng hàng) để ra N khoảng cách, nên bắt buộc phải dùng `axis=1` ạ."*
