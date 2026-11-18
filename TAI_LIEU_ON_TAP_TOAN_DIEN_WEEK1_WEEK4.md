# CẨM NANG ÔN TẬP TOÀN DIỆN MÔN HỌC MÁY (MACHINE LEARNING)
## TỪ MẤT GỐC ĐẾN LÀM CHỦ BẢN CHẤT: HÀNH TRÌNH TỪ WEEK 1 ĐẾN WEEK 4

> **Tài liệu dành cho**: Nguyễn Hải Anh (MSSV: 24001968)  
> **Học phần**: Machine Learning (Học máy)  
> **Phạm vi kiến thức**: Tuần 1 đến Tuần 4.
>
> 📚 **BỘ TÀI LIỆU ÔN TẬP CHUYÊN SÂU TỪNG TUẦN & THỰC HÀNH CODE**:
> 1. [Tuần 1: K-NN Hồi quy & Hồi quy tuyến tính Linear Regression (Toán OLS, Normal Eq, 4 bài tập)](file:///d:/MachineLearning/Week1/TAI_LIEU_ON_TAP_WEEK1_KNN_LINEAR_REGRESSION.md)
> 2. [Tuần 2 & 2b: Phân loại xác suất Naïve Bayes (Gaussian, Multinomial, Bernoulli, Spam, Laplace)](file:///d:/MachineLearning/Week2/TAI_LIEU_ON_TAP_WEEK2_NAIVE_BAYES.md)
> 3. [Tuần 3: K-NN Nhị phân & Hồi quy Logistic (Sigmoid, Cross-Entropy, SGD, Đánh giá y tế)](file:///d:/MachineLearning/Week3/TAI_LIEU_ON_TAP_WEEK3_LOGISTIC_REGRESSION.md)
> 4. [Tuần 4: K-NN Đa lớp & Hồi quy Softmax (One-hot, Softmax-stable, Nhận dạng chữ số MNIST)](file:///d:/MachineLearning/week4/TAI_LIEU_ON_TAP_WEEK4_MULTICLASS_SOFTMAX.md)
> 5. 🚀 **[CẨM NANG THỰC CHIẾN: VIẾT CODE ĐỘ ĐO & HỒI QUY (SCIKIT-LEARN, NUMPY, BẪY LỖI)](file:///d:/MachineLearning/CAM_NANG_VIET_CODE_DO_DO_VA_HOI_QUY.md)**

---

## LỜI NÓI ĐẦU: VƯỢT QUA CẢM GIÁC "NGỢP VÌ AI LÀM HẾT"

Nếu bạn đang cảm thấy hoang mang vì trước đây các bài tập thực hành (`.ipynb`, `.py`) đều do AI sinh ra mã nguồn và bạn chỉ việc nộp mà không hiểu bản chất bên dưới, **đừng lo lắng!**

Bản chất của Học máy (Machine Learning) trong 4 tuần đầu thực ra chỉ xoay quanh một câu hỏi triết học rất tự nhiên:  
> **"Làm thế nào để một cỗ máy nhìn vào các dữ liệu trong quá khứ ($X, Y$) và học được quy luật để tự đưa ra dự đoán cho một dữ liệu mới toanh ($x_{\text{new}}$)?"**

Tài liệu này được biên soạn đặc biệt theo phương pháp **"Top-Down + Line-by-Line Code Breakdown"**:
1. **Trực giác đời thực (Intuition)**: Hiểu trước bằng tư duy thông thường.
2. **Mô hình Toán học (Math Formula)**: Hiểu từng ký hiệu có ý nghĩa gì.
3. **Giải phẫu từng dòng Code (Code Dissection)**: Hiểu tại sao code viết như vậy (tại sao phải `.T`, tại sao phải thêm số `1`, tại sao phải trừ `max`).
4. **Bộ câu hỏi vấn đáp "sống còn" (Oral Exam Q&A)**: Giúp bạn tự tin trả lời bất kỳ câu hỏi nào của giảng viên khi bảo vệ bài tập hoặc thi vấn đáp.

---

```
                       BẢN ĐỒ KIẾN THỨC TỪNG TUẦN (WEEKS 1 - 4)
                       
               ┌────────────────────────────────────────────────────────┐
               │              BÀI TOÁN HỌC CÓ GIÁM SÁT                   │
               │               (SUPERVISED LEARNING)                    │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                 ┌────────────────────────┴────────────────────────┐
                 ▼                                                 ▼
      BÀI TOÁN HỒI QUY (REGRESSION)               BÀI TOÁN PHÂN LOẠI (CLASSIFICATION)
      (Dự đoán số liên tục: điểm, giá, độ dày)     (Dự đoán nhãn rời rạc: đỗ/trượt, loài hoa)
                 │                                                 │
        ┌────────┴────────┐                       ┌────────────────┼────────────────┐
        ▼                 ▼                       ▼                ▼                ▼
   [TUẦN 1]           [TUẦN 1]                [TUẦN 2 & 2b]     [TUẦN 3]        [TUẦN 4]
K-NN Regression   Linear Regression            Naïve Bayes    Binary LogReg  Multiclass Softmax
(Học lười,         (Đường thẳng tối ưu,        (Xác suất,     (Sigmoid,      (Softmax, One-hot,
 khoảng cách)      nghiệm giải tích)          Laplace smooth) Cross-Entropy)  MNIST chữ số)
```

---

# PHẦN 1: BỨC TRANH TOÀN CẢNH — CÁC KHÁI NIỆM NỀN TẢNG CẦN KHẮC CỐT GHI TÂM

Trước khi đi vào từng tuần, bạn cần hiểu chuẩn 4 thuật ngữ cốt lõi mà giảng viên luôn dùng:

1. **Đặc trưng (Feature - $X$)**: Các thông số đầu vào dùng để mô tả một đối tượng.  
   - Ví dụ: Chiều dài cánh hoa, số phòng ngủ của ngôi nhà, hàm lượng kim loại trong kính.  
   - Ký hiệu: $\mathbf{x} = [x_1, x_2, \dots, x_d]^T$ ($d$ là số chiều/số thuộc tính).
2. **Nhãn / Biến mục tiêu (Label / Target - $y$)**: Cái mà ta muốn mô hình dự đoán.  
   - Nếu $y$ là số thực liên tục ($y \in \mathbb{R}$ như giá nhà 45.5 triệu/m²): Đây là **Bài toán Hồi quy (Regression)**.  
   - Nếu $y$ là nhãn rời rạc ($y \in \{0, 1\}$ như đỗ/trượt, hoặc $y \in \{0, 1, \dots, C-1\}$ như chữ số 0 đến 9): Đây là **Bài toán Phân loại (Classification)**.
3. **Mô hình tham số (Parametric) vs Phi tham số (Non-parametric)**:
   - *Mô hình tham số* (Linear Regression, Logistic Regression): Giả định trước một dạng hàm số (ví dụ $y = \mathbf{w}^T \mathbf{x} + b$). Quá trình "học" chính là đi tìm bộ số trọng số $\mathbf{w}$. Học xong chỉ cần lưu $\mathbf{w}$, có thể vứt bỏ dữ liệu huấn luyện.
   - *Mô hình phi tham số* (K-NN): Không giả định hàm số cố định. Quá trình "học" thực chất là lưu toàn bộ dữ liệu huấn luyện vào bộ nhớ. Khi có mẫu mới, nó lôi toàn bộ dữ liệu cũ ra so sánh.
4. **Hàm mất mát (Loss Function / Cost Function - $J$ hoặc $\mathcal{L}$)**:
   - Thước đo mức độ "ngu ngốc" hay "sai sót" của mô hình.
   - Mục tiêu của Machine Learning là tìm tham số sao cho hàm mất mát này **nhỏ nhất có thể** (thông qua giải tích toán học hoặc giải thuật leo dốc giảm dần Gradient Descent).

---

# PHẦN 2: CHI TIẾT NỘI DUNG TỪNG TUẦN HỌC

---

## TUẦN 1: K-NN CHO HỒI QUY & HỒI QUY TUYẾN TÍNH (LINEAR REGRESSION)

### 1. K-NN cho bài toán Hồi quy (K-Nearest Neighbors Regression)

#### A. Trực giác đời sống
Bạn muốn định giá một căn hộ chung cư 70m² ở quận Cầu Giấy. Cách tự nhiên nhất bạn làm là gì?  
Bạn đi hỏi giá của $K = 5$ căn hộ chung cư khác cũng tầm 70m² ở ngay khu vực Cầu Giấy gần đó. Sau đó bạn lấy **trung bình cộng** giá của 5 căn đó để ước lượng cho căn của mình. Đó chính là K-NN Regression!

#### B. Bản chất Toán học
1. **Khoảng cách**: Đo mức độ "giống nhau" giữa điểm mới $\mathbf{x}$ và điểm dữ liệu trong tập train $\mathbf{x}_i$:
   - Khoảng cách Euclid (Chuẩn $L_2$):
     $$d(\mathbf{x}, \mathbf{x}_i) = \|\mathbf{x} - \mathbf{x}_i\|_2 = \sqrt{\sum_{j=1}^d (x_j - x_{i,j})^2}$$
2. **Tìm láng giềng**: Sắp xếp khoảng cách tăng dần, chọn ra tập $\mathcal{N}_K(\mathbf{x})$ gồm $K$ điểm có khoảng cách nhỏ nhất.
3. **Dự đoán**:
   $$\hat{y} = \frac{1}{K} \sum_{i \in \mathcal{N}_K(\mathbf{x})} y_i$$

#### C. Giải phẫu Code Python (Week 1)
```python
# 1. Hàm tính khoảng cách nhiều chiều
def distance(X_train, x_val):
    # X_train có shape (N, d), x_val có shape (d,)
    # Phép trừ (X_train - x_val) tự động broadcast
    # np.linalg.norm tính căn bậc hai tổng bình phương theo trục axis=1
    return np.linalg.norm(X_train - x_val, ord=2, axis=1)

# 2. Tìm K láng giềng gần nhất
distances = distance(X_train, x_val_j)
# np.argsort trả về MẢNG CHỈ SỐ sau khi sắp xếp tăng dần
nearest_indices = np.argsort(distances)[:k]

# 3. Dự đoán bằng trung bình cộng
y_pred = np.mean(Y_train[nearest_indices])
```

> **Cạm bẫy thi vấn đáp về K-NN**:
> - **Hỏi**: K-NN có ngoại suy (extrapolation) được không?
> - **Đáp**: **KHÔNG BAO GIỜ**. Vì kết quả dự đoán $\hat{y}$ là trung bình cộng của các nhãn trong tập train, nên $\hat{y}$ luôn bị chặn trong khoảng $[\min(Y_{\text{train}}), \max(Y_{\text{train}})]$. Nếu tương lai giá nhà tăng gấp đôi mức cao nhất lịch sử, K-NN hoàn toàn bất lực.
> - **Hỏi**: Tại sao phải chuẩn hóa dữ liệu (Feature Scaling) trước khi dùng K-NN?
> - **Đáp**: Khoảng cách Euclid tính theo bình phương hiệu $(x_j - x_{i,j})^2$. Nếu thuộc tính $x_1$ là "Diện tích" (từ 50 đến 500 m²) và thuộc tính $x_2$ là "Số phòng ngủ" (từ 1 đến 5 phòng), thì biến diện tích với giá trị hàng trăm sẽ lấn át hoàn toàn biến số phòng ngủ, biến khoảng cách bị sai lệch nghiêm trọng.

---

### 2. Hồi quy tuyến tính (Linear Regression)

#### A. Trực giác đời sống
Bạn đo điểm SAT ($x$) và điểm đại học GPA ($y$). Bạn thấy điểm SAT càng cao thì GPA có xu hướng càng cao. Bạn muốn vẽ **một đường thẳng đẹp nhất** đi xuyên qua đám mây điểm dữ liệu đó, sao cho khoảng cách từ các điểm thực tế đến đường thẳng là bé nhất. Đường thẳng đó có phương trình: $\hat{y} = w_0 + w_1 x$.

#### B. Bản chất Toán học
1. **Dạng mô hình tổng quát ($d$ chiều)**:
   $$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_d x_d$$
   Để gọn gàng, ta thêm hệ số giả $x_0 = 1$ (bias trick) vào đầu vector đặc trưng $\bar{\mathbf{x}} = [1, x_1, x_2, \dots, x_d]^T$:
   $$\hat{y} = \bar{\mathbf{x}}^T \mathbf{w}$$
2. **Hàm mất mát (Mean Squared Error - MSE)**:
   $$J(\mathbf{w}) = \frac{1}{2N} \sum_{i=1}^N (y_i - \hat{y}_i)^2 = \frac{1}{2N} \|\mathbf{y} - \bar{\mathbf{X}}\mathbf{w}\|_2^2$$
3. **Nghiệm giải tích phương trình pháp tuyến (Normal Equation)**:
   Để $J(\mathbf{w})$ nhỏ nhất, ta lấy đạo hàm theo $\mathbf{w}$ và cho bằng $0$:
   $$\frac{\partial J}{\partial \mathbf{w}} = 0 \implies \mathbf{w} = (\bar{\mathbf{X}}^T \bar{\mathbf{X}})^{-1} \bar{\mathbf{X}}^T \mathbf{y}$$
   $(\bar{\mathbf{X}}^T \bar{\mathbf{X}})^{-1} \bar{\mathbf{X}}^T$ chính là **Ma trận giả nghịch đảo (Pseudo-inverse)** của $\bar{\mathbf{X}}$, ký hiệu $\bar{\mathbf{X}}^\dagger$.

#### C. Giải phẫu Code Python (Week 1)
```python
# Bổ sung cột số 1 vào ma trận dữ liệu (cho hệ số bias w0)
ones = np.ones((X.shape[0], 1))
Xbar = np.concatenate((ones, X), axis=1)

# Cách 1: Giải trực tiếp bằng ma trận giả nghịch đảo (pinv)
w = np.linalg.pinv(Xbar).dot(y)

# Cách 2: Sử dụng scikit-learn
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
model.fit(X, y)
# model.intercept_ chính là w0, model.coef_ chính là [w1, w2, ...]
```

#### D. Các độ đo đánh giá hồi quy quan trọng
1. **MSE (Mean Squared Error)**: Trung bình bình phương sai số. Càng bé càng tốt. Phạt rất nặng các điểm sai số lớn (do có bình phương).
2. **MAE (Mean Absolute Error)**: Trung bình trị tuyệt đối sai số. Cùng đơn vị đo với $y$, ít bị ảnh hưởng bởi điểm ngoại lai (outliers).
3. **$R^2$ (Hệ số xác định - R-squared)**:
   $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
   - $R^2 = 1.0$: Mô hình dự đoán đúng tuyệt đối $100\%$.
   - $R^2 = 0$: Mô hình chỉ dự đoán tương đương với việc lấy giá trị trung bình $\bar{y}$.
   - $R^2 < 0$: Mô hình cực kỳ tệ, dự đoán còn tệ hơn cả mức trung bình.

---

## TUẦN 2 & 2B: MÔ HÌNH PHÂN LOẠI XÁC SUẤT NAÏVE BAYES

### 1. Trực giác cốt lõi & Giả thiết "Ngây thơ"

#### A. Định lý Bayes
Khi bạn nhìn thấy các triệu chứng của bệnh nhân $\mathbf{x} = [\text{sốt, ho, mất vị giác}]$, bạn muốn tính xác suất người đó bị mắc bệnh cúm ($y = \text{Cúm}$) hay Covid ($y = \text{Covid}$):
$$P(y = c \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid y = c) \cdot P(y = c)}{P(\mathbf{x})}$$
- **$P(y = c)$ (Prior - Tiên nghiệm)**: Tỷ lệ người mắc bệnh $c$ trong dân số tự nhiên.
- **$P(\mathbf{x} \mid y = c)$ (Likelihood - Độ hợp lý)**: Nếu đã bị bệnh $c$ rồi, thì xác suất xuất hiện tổ hợp triệu chứng $\mathbf{x}$ là bao nhiêu?
- **$P(y = c \mid \mathbf{x})$ (Posterior - Hậu nghiệm)**: Sau khi nhìn thấy triệu chứng $\mathbf{x}$, xác suất thực tế mắc bệnh $c$ là bao nhiêu?

#### B. Tại sao gọi là "Ngây thơ" (Naïve)?
Để tính $P(\mathbf{x} \mid y) = P(x_1, x_2, \dots, x_d \mid y)$ cho hàng chục thuộc tính cùng lúc là điều bất khả thi vì đòi hỏi lượng dữ liệu khổng lồ.  
Naïve Bayes đưa ra một giả thiết cực kỳ "ngây thơ": **Tất cả các thuộc tính $x_1, x_2, \dots, x_d$ đều hoàn toàn độc lập với nhau khi đã biết nhãn $y$**:
$$P(x_1, x_2, \dots, x_d \mid y) = P(x_1 \mid y) \times P(x_2 \mid y) \times \dots \times P(x_d \mid y) = \prod_{j=1}^d P(x_j \mid y)$$
Mặc dù trên thực tế các thuộc tính thường phụ thuộc nhau (ví dụ: "sốt" và "ho" hay đi kèm nhau), nhưng giả thiết này làm cho việc tính toán trở nên cực kỳ nhanh chóng và kết quả phân loại trong thực tế vẫn hoạt động ấn tượng đến bất ngờ!

---

### 2. Ba biến thể Naïve Bayes kinh điển

| Tiêu chí | Gaussian Naïve Bayes (GNB) | Multinomial Naïve Bayes (MNB) | Bernoulli Naïve Bayes (BNB) |
| :--- | :--- | :--- | :--- |
| **Kiểu dữ liệu đầu vào ($X$)** | Biến liên tục (Số thực: chiều dài, cân nặng, huyết áp) | Biến đếm tần suất (Số nguyên không âm: tần số từ $0, 1, 5, 10$) | Biến nhị phân (Chỉ có $0$ hoặc $1$: có/không) |
| **Giả định phân phối** | Phân phối chuẩn Gaussian $\mathcal{N}(\mu, \sigma^2)$ | Phân phối đa thức (Multinomial Distribution) | Phân phối Bernoulli |
| **Ứng dụng tiêu biểu** | Phân loại hoa Iris, Ung thư vú Wisconsin | Phân loại thư rác (Spam email), Phân loại văn bản tin tức | Phát hiện triệu chứng bệnh (có/không), Khảo sát có/không |
| **Lớp trong `scikit-learn`** | `sklearn.naive_bayes.GaussianNB` | `sklearn.naive_bayes.MultinomialNB` | `sklearn.naive_bayes.BernoulliNB` |

---

### 3. Hai kỹ thuật "sống còn" trong Naïve Bayes

#### A. Kỹ thuật chống tràn số dưới (Log-trick to prevent Underflow)
- **Vấn đề**: Mỗi xác suất $P(x_j \mid y)$ là một số nhỏ hơn 1 (ví dụ $0.001$). Khi văn bản có 1000 từ, việc nhân 1000 số nhỏ hơn 1 với nhau:
  $$0.001 \times 0.002 \times \dots \times 0.0005 \approx 0.00000000... \to 0$$
  Máy tính sẽ bị tràn số dưới (floating-point underflow) và làm tròn thành `0.0`, khiến mô hình không thể so sánh được!
- **Giải pháp**: Lấy logarit tự nhiên ($\ln$). Phép nhân biến thành phép cộng:
  $$\ln \left( P(y) \prod_{j=1}^d P(x_j \mid y) \right) = \ln P(y) + \sum_{j=1}^d \ln P(x_j \mid y)$$
  Máy tính cộng các số âm vừa phải, hoàn toàn ổn định và chính xác tuyệt đối.

#### B. Kỹ thuật làm trơn Laplace (Laplace Smoothing)
- **Vấn đề**: Trong tập train phân loại email, từ "bitcoin" chưa từng xuất hiện trong các email bình thường ($y = \text{Non-Spam}$). Khi đó $P(\text{"bitcoin"} \mid \text{Non-Spam}) = 0$.
  Nếu một ngày có email công việc vô tình nhắc đến từ "bitcoin", tích xác suất sẽ bị:
  $$P(\text{email} \mid \text{Non-Spam}) = P(\dots) \times 0 \times P(\dots) = 0$$
  Một thuộc tính chưa từng thấy lập tức xóa sạch cơ hội của toàn bộ lớp!
- **Giải pháp**: Cộng thêm một đại lượng ảo $\alpha$ (thường chọn $\alpha = 1$):
  $$P(x_j \mid y = c) = \frac{N_{jc} + \alpha}{N_c + \alpha \cdot d}$$
  ($N_{jc}$: số lần từ $j$ xuất hiện trong lớp $c$, $N_c$: tổng số từ của lớp $c$, $d$: kích thước từ điển).  
  Nhờ đó xác suất không bao giờ bị bằng 0.

---

## TUẦN 3: K-NN CHO PHÂN LOẠI NHỊ PHÂN & HỒI QUY LOGISTIC (LOGISTIC REGRESSION)

### 1. K-NN cho Phân loại Nhị phân (Binary Classification)
- Khác với Hồi quy (lấy trung bình cộng): Trong bài toán phân loại, ta đếm xem trong $K$ láng giềng gần nhất, **nhãn nào chiếm đa số** (Majority Voting) thì dự đoán điểm mới theo nhãn đó.
- Code cốt lõi:
  ```python
  # Tìm K láng giềng gần nhất, lấy nhãn của chúng:
  y_nearest = [Y_train[idx] for idx in nearest_indices[:k]]
  # Nhãn xuất hiện nhiều nhất:
  y_pred = max(set(y_nearest), key=y_nearest.count)
  ```

---

### 2. Hồi quy Logistic Nhị phân (Binary Logistic Regression)

#### A. Trực giác đời sống
Tại sao không dùng Hồi quy tuyến tính ($\hat{y} = \mathbf{w}^T \mathbf{x}$) để phân loại đỗ ($y=1$) hay trượt ($y=0$)?  
Bởi vì đường thẳng tuyến tính có thể bắn ra các giá trị $-\infty$ hoặc $+\infty$ (ví dụ: dự đoán điểm đỗ bằng $-3.5$ hoặc $+10.2$), trong khi xác suất bắt buộc phải nằm nghiêm ngặt trong đoạn $[0, 1]$.  
Giải pháp: Ta lấy giá trị số thực $z = \mathbf{w}^T \mathbf{x}$ bóp (ép) nó qua một "cái phễu hình chữ S" để ép kết quả luôn nằm trong khoảng $(0, 1)$. "Cái phễu" đó chính là **Hàm Sigmoid**.

```
                HÀM SIGMOID: σ(z) = 1 / (1 + e^(-z))
                
         1.0 ┤                         . - - ' ' ' 
             │                     . '
             │                  . '
         0.5 ┤ - - - - - - - - ┼ - - - - - - - - - - Ngưỡng 0.5 (z = 0)
             │              .  │
             │           . '   │
         0.0 ┤ . . . - '       │
             └─────────────────┴───────────────────
                              z = 0
             [Dự đoán: y = 0]     [Dự đoán: y = 1]
```

#### B. Bản chất Toán học
1. **Hàm kích hoạt Sigmoid**:
   $$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad \text{với } z = \mathbf{w}^T \mathbf{x}$$
   - Khi $z \to +\infty \implies \sigma(z) \to 1$.
   - Khi $z \to -\infty \implies \sigma(z) \to 0$.
   - Khi $z = 0 \implies \sigma(z) = 0.5$.
2. **Quy tắc ra quyết định (Decision Rule)**:
   $$\hat{y} = \begin{cases} 1 & \text{nếu } \sigma(\mathbf{w}^T \mathbf{x}) \ge 0.5 \iff \mathbf{w}^T \mathbf{x} \ge 0 \\ 0 & \text{ngược lại} \end{cases}$$
   Phương trình $\mathbf{w}^T \mathbf{x} = 0$ chính là **Mặt phẳng siêu ranh giới quyết định (Decision Boundary)** phân chia 2 nửa không gian.
3. **Hàm mất mát Binary Cross-Entropy (Log-loss)**:
   $$J(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
   - Nếu nhãn thực tế $y_i = 1$: Mất mát là $-\log(\hat{y}_i)$. Nếu mô hình đoán $\hat{y}_i \to 1$, mất mát tiến về $0$. Nếu đoán $\hat{y}_i \to 0$, mất mát tiến đến $+\infty$ (phạt cực nặng)!
   - Nếu nhãn thực tế $y_i = 0$: Mất mát là $-\log(1 - \hat{y}_i)$.
4. **Giải thuật tối ưu Gradient Descent**:
   Đạo hàm của hàm mất mát theo trọng số có dạng cực kỳ đẹp:
   $$\frac{\partial J}{\partial \mathbf{w}} = \mathbf{x}_i (\hat{y}_i - y_i)$$
   Quy tắc cập nhật trọng số lặp (với tốc độ học $\eta$):
   $$\mathbf{w}_{\text{new}} = \mathbf{w}_{\text{old}} - \eta (\hat{y}_i - y_i) \mathbf{x}_i = \mathbf{w}_{\text{old}} + \eta (y_i - \hat{y}_i) \mathbf{x}_i$$

#### C. Các độ đo đánh giá bài toán phân loại (Evaluation Metrics)

```
                            MA TRẬN NHẦM LẪN (CONFUSION MATRIX)
                                    Dự đoán: 0         Dự đoán: 1
                     Thực tế: 0  ┌───────────────┬───────────────┐
                                 │ TN (True Neg) │ FP (False Pos)│  (Sai lầm loại I)
                                 ├───────────────┼───────────────┤
                     Thực tế: 1  │ FN (False Neg)│ TP (True Pos) │  (Sai lầm loại II - Rất nguy hiểm trong y tế)
                                 └───────────────┴───────────────┘
```

1. **Accuracy (Độ chính xác tổng thể)**: $\frac{TP + TN}{TP + TN + FP + FN}$.  
   *Cảnh báo*: Vô dụng khi dữ liệu bị mất cân bằng (Imbalanced data). Ví dụ: 99% người không mắc bệnh hiểm nghèo, mô hình luôn đoán bừa "0" vẫn đạt Accuracy 99% nhưng bỏ lọt tất cả người bệnh!
2. **Precision (Độ chuẩn xác)**: $\frac{TP}{TP + FP}$. Trong số những ca mô hình dự đoán là Dương tính, có bao nhiêu ca thực sự đúng? (Quan trọng trong lọc thư rác, tránh ném nhầm email quan trọng vào Spam).
3. **Recall (Độ nhạy)**: $\frac{TP}{TP + FN}$. Trong số tất cả các ca thực sự Dương tính, mô hình bắt được bao nhiêu ca? (Cực kỳ quan trọng trong chẩn đoán y tế ung thư/tim mạch, thà bắt nhầm còn hơn bỏ sót).
4. **F1-Score**: Trung bình điều hòa giữa Precision và Recall: $F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$.

---

## TUẦN 4: K-NN PHÂN LOẠI NHIỀU LỚP & HỒI QUY SOFTMAX (MULTINOMIAL LOGISTIC REGRESSION)

### 1. K-NN cho Phân loại Nhiều lớp (Multiclass K-NN)
- Nguyên lý không đổi: Với mẫu mới $\mathbf{x}$, tính khoảng cách tới tất cả các điểm trong tập huấn luyện. Tìm $K$ điểm gần nhất, và nhãn xuất hiện nhiều nhất trong $K$ điểm đó sẽ là nhãn dự đoán (áp dụng cho $C = 3$ lớp như hoa Iris, hoặc $C = 10$ lớp như số MNIST).

---

### 2. Hồi quy Softmax (Multinomial Logistic Regression)

#### A. Trực giác đời sống
Khi phân loại nhị phân (2 lớp), ta dùng hàm Sigmoid để biến đổi một số thực thành xác suất của lớp 1 (xác suất lớp 0 tự động là $1 - P$).  
Nhưng khi có $C$ lớp ($C \ge 3$, ví dụ nhận dạng chữ số từ 0 đến 9):  
Với mỗi bức ảnh, mô hình tính ra $C$ điểm số thô $z_0, z_1, \dots, z_{C-1}$ (gọi là logits). Làm thế nào biến $C$ điểm số thô bất kỳ (có thể âm, dương, rất lớn) thành $C$ con số xác suất vừa **dương** vừa **có tổng bằng đúng 1.0 ($100\%$)**?  
Chiếc chìa khóa thần kỳ ở đây là **Hàm Softmax**!

#### B. Bản chất Toán học
1. **Hàm kích hoạt Softmax**:
   $$a_c = P(y = c \mid \mathbf{x}) = \frac{e^{z_c}}{\sum_{k=0}^{C-1} e^{z_k}}, \quad \text{với } z_c = \mathbf{w}_c^T \mathbf{x}$$
   - Tử số $e^{z_c}$ luôn luôn dương với mọi giá trị của $z_c$.
   - Mẫu số là tổng tất cả các $e^{z_k}$, đảm bảo $\sum_{c=0}^{C-1} a_c = 1$.
2. **Kỹ thuật ổn định số học (Softmax Stable Trick - BẮT BUỘC PHẢI BIẾT)**:
   - Nếu một điểm số $z_c = 1000$, máy tính tính $e^{1000}$ sẽ bị tràn số dương (`Overflow -> inf/nan`).
   - Khắc phục bằng tính chất bất biến: Trừ tất cả các giá trị $z$ cho giá trị lớn nhất $z_{\max} = \max(z)$:
     $$\text{softmax\_stable}(\mathbf{z}) = \frac{e^{\mathbf{z} - \max(\mathbf{z})}}{\sum e^{\mathbf{z} - \max(\mathbf{z})}}$$
   - Khi đó số lớn nhất trở thành $e^0 = 1$, các số còn lại đều nhỏ hơn 1, máy tính không bao giờ bị tràn số!
3. **Mã hóa nhãn One-hot (One-hot Encoding)**:
   Nếu bài toán có 3 lớp ($C = 3$):
   - Nhãn $y = 0 \implies \mathbf{y} = [1, 0, 0]^T$
   - Nhãn $y = 1 \implies \mathbf{y} = [0, 1, 0]^T$
   - Nhãn $y = 2 \implies \mathbf{y} = [0, 0, 1]^T$
4. **Hàm mất mát Categorical Cross-Entropy**:
   $$J(\mathbf{W}) = -\sum_{i=1}^N \sum_{c=0}^{C-1} y_{ic} \ln(a_{ic})$$
   Vì vector $\mathbf{y}_i$ chỉ có duy nhất vị trí nhãn đúng bằng 1, các vị trí khác bằng 0, nên thực chất hàm mất mát chỉ đơn giản là: $-\ln(\text{xác suất của lớp đúng})$.
5. **Đạo hàm và Cập nhật Gradient Descent (SGD)**:
   $$\frac{\partial J}{\partial \mathbf{W}} = \mathbf{x}_i (\mathbf{a}_i - \mathbf{y}_i)^T$$
   Công thức cập nhật:
   $$\mathbf{W}_{\text{new}} = \mathbf{W}_{\text{old}} + \eta \mathbf{x}_i (\mathbf{y}_i - \mathbf{a}_i)^T$$
   (Trong đó $\mathbf{y}_i$ là nhãn one-hot thực tế, $\mathbf{a}_i$ là xác suất dự đoán qua softmax).

---

### 3. Giải phẫu từng dòng Code NumPy trong Tuần 4

Đây là đoạn mã nguồn cốt lõi tự xây dựng từ đầu (from scratch) bằng NumPy trong bài thực hành Tuần 4:

```python
import numpy as np
from scipy import sparse

# 1. Hàm chuyển nhãn số nguyên thành ma trận One-hot
def convert_labels(y, C):
    # Tạo ma trận thưa shape (C, N)
    # y là chỉ số dòng, np.arange(len(y)) là chỉ số cột
    Y = sparse.coo_matrix((np.ones_like(y), 
                           (y, np.arange(len(y)))), shape=(C, len(y))).toarray()
    return Y

# 2. Hàm Softmax ổn định số học
def softmax_stable(Z):
    # Z có shape (C, N) - mỗi cột là điểm số của một mẫu dữ liệu
    # Trừ giá trị max của mỗi cột (axis=0) để tránh tràn số exp
    e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    A = e_Z / e_Z.sum(axis=0)
    return A

# 3. Thuật toán huấn luyện Softmax Regression bằng Stochastic Gradient Descent (SGD)
def softmax_regression(X, y, W_init, eta, tol=1e-4, max_count=10000):
    W = [W_init]
    C = W_init.shape[1]
    Y = convert_labels(y, C)
    N = X.shape[1]
    d = X.shape[0]
    count = 0
    check_w_after = 20
    
    while count < max_count:
        # Xáo trộn thứ tự các mẫu (Stochastic)
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)      # Vector đặc trưng mẫu thứ i (d, 1)
            yi = Y[:, i].reshape(C, 1)      # Vector nhãn one-hot của mẫu i (C, 1)
            ai = softmax_stable(np.dot(W[-1].T, xi))  # Dự đoán xác suất (C, 1)
            
            # Cập nhật trọng số theo hướng giảm gradient
            W_new = W[-1] + eta * xi.dot((yi - ai).T)
            count += 1
            
            # Tiêu chuẩn dừng: Trọng số không thay đổi đáng kể sau 20 bước
            if count % check_w_after == 0:
                if np.linalg.norm(W_new - W[-check_w_after]) < tol:
                    return W
            W.append(W_new)
    return W

# 4. Hàm dự đoán nhãn cho mẫu mới
def pred(W, X):
    # Tính xác suất softmax
    A = softmax_stable(W.T.dot(X))
    # Nhãn dự đoán là chỉ số có xác suất lớn nhất (argmax theo cột)
    return np.argmax(A, axis=0)
```

---

### 4. Bài toán thực tế lớn: Nhận dạng chữ số viết tay MNIST
- **Tập dữ liệu MNIST**: Gồm 60,000 ảnh huấn luyện và 10,000 ảnh kiểm tra của 10 chữ số từ 0 đến 9.
- **Biểu diễn dữ liệu**:
  - Mỗi ảnh có kích thước $28 \times 28$ pixel.
  - Duỗi phẳng (flatten) ma trận $28 \times 28$ thành 1 vector đặc trưng có $d = 784$ chiều.
  - Giá trị mỗi pixel từ $0$ (nền trắng) đến $255$ (nét mực đen). Thường chia cho $255.0$ để chuẩn hóa về khoảng $[0, 1]$.
  - Ma trận trọng số $\mathbf{W}$ của Softmax Regression sẽ có kích thước: $(785, 10)$ (784 thuộc tính + 1 bias cho 10 lớp).
- **So sánh với `scikit-learn`**:
  ```python
  from sklearn.linear_model import LogisticRegression
  # solver='sag' (Stochastic Average Gradient) chuyên dùng cho dữ liệu lớn nhiều lớp
  clf = LogisticRegression(multi_class='multinomial', solver='sag', max_iter=5000)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  ```

---

# PHẦN 3: BẢNG SO SÁNH TỔNG HỢP CÁC THUẬT TOÁN ĐÃ HỌC (WEEKS 1 - 4)

| Tiêu chí so sánh | K-NN (K-Nearest Neighbors) | Linear Regression | Naïve Bayes | Logistic / Softmax Regression |
| :--- | :--- | :--- | :--- | :--- |
| **Loại bài toán** | Cả Hồi quy & Phân loại | Chỉ Hồi quy | Chỉ Phân loại | Chỉ Phân loại (Binary / Multiclass) |
| **Bản chất mô hình** | Phi tham số (Non-parametric), Lazy learning | Tham số (Parametric) | Sinh (Generative), Dựa trên Xác suất | Phân biệt (Discriminative), Tìm siêu phẳng phân chia |
| **Thời gian Train** | Gần như bằng 0 (chỉ lưu dữ liệu vào RAM) | Rất nhanh (Nghiệm giải tích ma trận) | Rất nhanh (Đếm tần suất & tính mean/var) | Trung bình (Lặp Gradient Descent để hội tụ) |
| **Thời gian Predict** | Rất chậm khi dữ liệu lớn (phải duyệt qua toàn bộ tập train) | Cực nhanh (chỉ nhân ma trận $\mathbf{w}^T \mathbf{x}$) | Rất nhanh (chỉ nhân cộng các xác suất) | Cực nhanh (chỉ nhân ma trận $\mathbf{W}^T \mathbf{x}$ rồi argmax) |
| **Khả năng ngoại suy** | **Không thể** | Rất tốt (theo xu hướng đường thẳng) | Không áp dụng cho hồi quy | Không áp dụng cho hồi quy |
| **Độ nhạy với thang đo (Feature Scaling)** | **Cực kỳ nhạy cảm** (bắt buộc phải chuẩn hóa) | Ít nhạy với nghiệm giải tích, nhạy với Gradient Descent | Không bị ảnh hưởng | **Nhạy cảm** (chuẩn hóa giúp Gradient Descent hội tụ nhanh) |
| **Ưu điểm lớn nhất** | Đơn giản, mô hình hóa quan hệ phi tuyến cục bộ xuất sắc | Nghiệm giải tích rõ ràng, dễ giải thích ý nghĩa hệ số | Huấn luyện cực nhanh, rất mạnh với phân loại văn bản | Dự đoán đưa ra xác suất tin cậy, ranh giới phân chia mượt mà |
| **Nhược điểm lớn nhất** | Tốn bộ nhớ RAM, rất chậm khi inference | Giả định tuyến tính quá cứng nhắc | Giả thiết các thuộc tính độc lập thường sai trong thực tế | Khó phân loại tốt nếu dữ liệu có biên phi tuyến phức tạp |

---

# PHẦN 4: BỘ 20 CÂU HỎI VẤN ĐÁP "SỐNG CÒN" (ORAL EXAM DEFENSE PREP)

Dưới đây là 20 câu hỏi mà các giảng viên bộ môn Machine Learning thường xuyên hỏi nhất khi sinh viên báo cáo bài tập hoặc thi vấn đáp, kèm theo câu trả lời chuẩn xác, gãy gọn:

### Nhóm 1: Thuật toán K-NN (Tuần 1, Tuần 3, Tuần 4)
1. **Câu 1: K-NN là thuật toán "học lười" (Lazy Learning), điều đó có nghĩa là gì?**
   - *Trả lời*: Quá trình "huấn luyện" của K-NN thực chất không học bất kỳ trọng số nào mà chỉ đơn giản là lưu trữ toàn bộ dữ liệu huấn luyện vào bộ nhớ. Mọi tính toán phức tạp (tính khoảng cách, tìm láng giềng) bị trì hoãn cho đến khi có một mẫu mới cần dự đoán (Inference time).
2. **Câu 2: Tham số $K$ ảnh hưởng như thế nào đến mô hình? Chọn $K=1$ hay $K$ rất lớn sẽ bị hiện tượng gì?**
   - *Trả lời*: 
     - Khi $K$ quá nhỏ ($K = 1$): Ranh giới phân chia rất gồ ghề, nhạy cảm với từng điểm nhiễu $\implies$ Dễ bị **Overfitting** (quá khớp).
     - Khi $K$ quá lớn ($K \approx N$): Mô hình dự đoán nhãn theo lớp chiếm đa số trong tập train $\implies$ Bị **Underfitting** (quá đơn giản, bỏ qua cấu trúc cục bộ).
     - Thường chọn $K$ là số lẻ (để tránh hòa điểm khi vote nhị phân) và tối ưu qua kỹ thuật Cross-Validation.
3. **Câu 3: Tại sao trong K-NN bắt buộc phải chuẩn hóa dữ liệu (Standardization / Min-Max Scaling)?**
   - *Trả lời*: Vì K-NN dựa trên khoảng cách hình học Euclid $\sqrt{\sum (x_j - x'_{j})^2}$. Thuộc tính nào có dải giá trị lớn (ví dụ: Thu nhập 10 triệu - 50 triệu) sẽ lấn át hoàn toàn các thuộc tính có dải giá trị nhỏ (ví dụ: Số con từ 0 - 4), khiến khoảng cách bị chi phối phiến diện. Chuẩn hóa giúp mọi đặc trưng đóng vai trò bình đẳng.
4. **Câu 4: K-NN cho Hồi quy và K-NN cho Phân loại khác nhau ở bước nào?**
   - *Trả lời*: Các bước tính khoảng cách và chọn $K$ láng giềng gần nhất là giống hệt nhau. Chỉ khác ở bước ra kết quả cuối cùng:
     - Hồi quy (Regression): Lấy **trung bình cộng** nhãn của $K$ láng giềng.
     - Phân loại (Classification): Lấy **biểu quyết đa số** (Majority Voting - nhãn có tần suất xuất hiện nhiều nhất) của $K$ láng giềng.

---

### Nhóm 2: Hồi quy Tuyến tính & Độ đo (Tuần 1)
5. **Câu 5: Tại sao trong Hồi quy tuyến tính ta phải thêm cột số 1 vào ma trận dữ liệu $\bar{\mathbf{X}}$?**
   - *Trả lời*: Cột số 1 này đại diện cho biến giả $x_0 = 1$. Nó cho phép gộp hệ số chệch (bias / intercept $w_0$) vào cùng vector trọng số $\mathbf{w} = [w_0, w_1, \dots, w_d]^T$, giúp chuyển phương trình từ $\hat{y} = w_0 + \sum w_j x_j$ thành tích vô hướng ma trận ngắn gọn $\hat{y} = \bar{\mathbf{x}}^T \mathbf{w}$.
6. **Câu 6: Normal Equation $\mathbf{w} = (\bar{\mathbf{X}}^T \bar{\mathbf{X}})^{-1} \bar{\mathbf{X}}^T \mathbf{y}$ có nhược điểm gì? Khi nào không nghịch đảo được?**
   - *Trả lời*: 
     - Chi phí tính nghịch đảo ma trận là $\mathcal{O}(d^3)$. Nếu số thuộc tính $d$ quá lớn (hàng chục nghìn), việc nghịch đảo ma trận cực kỳ chậm và tốn bộ nhớ.
     - Ma trận $\bar{\mathbf{X}}^T \bar{\mathbf{X}}$ không nghịch đảo được khi nó bị kỳ dị (Singular matrix), xảy ra khi có hai hoặc nhiều thuộc tính tương quan tuyến tính hoàn hảo với nhau (Đa cộng tuyến - Multicollinearity), hoặc số mẫu $N < d$. Khi đó ta phải dùng ma trận giả nghịch đảo (Pseudo-inverse) qua phân rã SVD/QR.
7. **Câu 7: Hệ số xác định $R^2$ có thể âm không? Ý nghĩa của $R^2 = 0.85$ là gì?**
   - *Trả lời*: 
     - $R^2$ hoàn toàn **có thể âm** nếu mô hình dự đoán tệ hơn cả việc luôn đoán giá trị trung bình $\bar{y}$.
     - $R^2 = 0.85$ có nghĩa là: Mô hình đã giải thích được $85\%$ sự biến thiên (phương sai) của biến mục tiêu $y$ dựa trên các đặc trưng đầu vào $X$.

---

### Nhóm 3: Naïve Bayes (Tuần 2 & 2b)
8. **Câu 8: Tại sao lại gọi thuật toán này là "Naïve" (Ngây thơ)?**
   - *Trả lời*: Vì nó đưa ra một giả thiết cực kỳ ngây thơ và phi thực tế: Tất cả các thuộc tính độc lập có điều kiện với nhau khi biết nhãn lớp ($P(\mathbf{x} \mid y) = \prod P(x_j \mid y)$).
9. **Câu 9: Hiện tượng "Zero Probability" trong Naïve Bayes là gì và cách khắc phục?**
   - *Trả lời*: Nếu một từ hoặc một giá trị thuộc tính chưa từng xuất hiện trong tập train của một lớp nào đó, xác suất của nó bằng 0. Khi nhân tích chuỗi lại, cả biểu thức xác suất hậu nghiệm sẽ bị bằng 0. Khắc phục bằng **Làm trơn Laplace (Laplace Smoothing)**: cộng $\alpha = 1$ vào tử số và $\alpha \times d$ vào mẫu số.
10. **Câu 10: Tại sao phải lấy Logarithm của xác suất trong Naïve Bayes?**
    - *Trả lời*: Để tránh hiện tượng **Tràn số dưới (Floating-point Underflow)**. Tích của hàng trăm xác suất nhỏ hơn 1 sẽ tạo ra một số cực kỳ nhỏ khiến máy tính làm tròn thành số 0. Lấy logarit biến phép nhân các số nhỏ thành phép cộng các số âm, đảm bảo tính toán ổn định và giữ nguyên thứ tự độ lớn.
11. **Câu 11: Khi nào thì dùng Gaussian NB, khi nào dùng Multinomial NB, khi nào dùng Bernoulli NB?**
    - *Trả lời*: 
      - Gaussian NB: Khi đặc trưng là số thực liên tục (chiều cao, cân nặng, kích thước hoa).
      - Multinomial NB: Khi đặc trưng là số đếm tần suất rời rạc (tần số xuất hiện của từ trong văn bản).
      - Bernoulli NB: Khi đặc trưng là dạng nhị phân 0 hoặc 1 (từ đó có xuất hiện hay không, bệnh nhân có triệu chứng sốt hay không).

---

### Nhóm 4: Hồi quy Logistic & Softmax Regression (Tuần 3 & Tuần 4)
12. **Câu 12: Tên gọi là "Hồi quy Logistic" (Logistic Regression), nhưng tại sao nó lại là bài toán Phân loại (Classification)?**
    - *Trả lời*: Về bản chất toán học, mô hình vẫn thực hiện "hồi quy" để ước lượng một giá trị liên tục, nhưng giá trị đó là **Log-odds (logit)**: $\ln \left( \frac{p}{1-p} \right) = \mathbf{w}^T \mathbf{x}$. Tuy nhiên, sau đó ta đưa giá trị này qua hàm Sigmoid để thu được xác suất và áp dụng ngưỡng (thresholding) để gán nhãn rời rạc $0$ hoặc $1$. Do đó, mục đích sử dụng cuối cùng của nó là **Phân loại**.
13. **Câu 13: Hàm Sigmoid và Hàm Softmax khác nhau như thế nào?**
    - *Trả lời*: 
      - Hàm Sigmoid: Dùng cho phân loại nhị phân ($C = 2$ lớp). Nhận vào 1 điểm số $z$ và trả về xác suất của lớp 1.
      - Hàm Softmax: Là sự tổng quát hóa tự nhiên của Sigmoid cho bài toán phân loại nhiều lớp ($C \ge 3$). Nhận vào một vector điểm số $C$ chiều và trả về một vector phân phối xác suất có tổng đúng bằng 1.0. Khi $C = 2$, hàm Softmax tương đương hoàn toàn với hàm Sigmoid.
14. **Câu 14: Tại sao trong hàm Softmax ta phải trừ đi giá trị $\max(Z)$ (Softmax Stable Trick)?**
    - *Trả lời*: Vì hàm Softmax có số mũ $e^z$. Nếu giá trị $z$ lớn (ví dụ $z = 1000$), $e^{1000}$ sẽ vượt quá giới hạn biểu diễn của số dấu phẩy động 64-bit và gây ra lỗi tràn số tràn số dương (`Overflow` ra `inf` hoặc `NaN`). Trừ đi $\max(Z)$ đảm bảo số mũ lớn nhất luôn bằng $e^0 = 1$, các số khác đều $\le 1$, giúp triệt tiêu hoàn toàn nguy cơ tràn số mà không làm đổi kết quả xác suất.
15. **Câu 15: Hàm mất mát trong Hồi quy Logistic là gì? Tại sao không dùng hàm MSE như Linear Regression?**
    - *Trả lời*: Hàm mất mát là **Binary Cross-Entropy (Log-loss)**. Nếu dùng MSE cho Logistic Regression, do có hàm phi tuyến Sigmoid lồng bên trong, hàm mất mát sẽ trở thành **hàm không lồi (Non-convex)**, chứa rất nhiều điểm cực tiểu cục bộ (local minima), khiến thuật toán Gradient Descent dễ bị mắc kẹt và không tìm được nghiệm tối ưu toàn cục. Ngược lại, Binary Cross-Entropy là hàm lồi (Convex), đảm bảo Gradient Descent luôn hội tụ về nghiệm tối ưu toàn cục duy nhất.
16. **Câu 16: Mã hóa One-hot là gì và tại sao lại cần thiết trong Softmax Regression?**
    - *Trả lời*: Mã hóa One-hot biến một nhãn số nguyên rời rạc (ví dụ số 2 trong 3 lớp) thành vector nhị phân $[0, 0, 1]^T$. Việc này triệt tiêu giả định sai lầm về thứ bậc tự nhiên giữa các nhãn (nếu để số nguyên 0, 1, 2 thì máy tính sẽ ngộ nhận lớp 2 "lớn gấp đôi" lớp 1), đồng thời cho phép tính toán trực tiếp hàm mất mát Categorical Cross-Entropy dưới dạng tích vô hướng giữa vector nhãn và vector xác suất dự đoán.

---

### Nhóm 5: Đánh giá mô hình & Tối ưu hóa (Chung)
17. **Câu 17: Trong bài toán y tế (chẩn đoán ung thư hoặc tim mạch Framingham), ta ưu tiên độ đo Precision hay Recall hơn? Vì sao?**
    - *Trả lời*: Ưu tiên **Recall (Độ nhạy)** hơn rất nhiều. Bởi vì trong y tế, việc chẩn đoán nhầm một người khỏe mạnh thành có bệnh (False Positive) chỉ khiến họ phải đi làm xét nghiệm lại; nhưng nếu bỏ sót một người thực sự mắc bệnh nặng mà kết luận là khỏe mạnh (False Negative) thì người bệnh sẽ mất cơ hội điều trị kịp thời và có thể dẫn đến tử vong.
18. **Câu 18: Khi tập dữ liệu bị mất cân bằng trầm trọng (Imbalanced Data), độ đo Accuracy có đáng tin cậy không? Nên dùng độ đo nào thay thế?**
    - *Trả lời*: Accuracy hoàn toàn **không đáng tin cậy**. Ví dụ tập dữ liệu có 95% mẫu nhãn 0 và 5% mẫu nhãn 1, mô hình đoán toàn bộ là 0 vẫn đạt Accuracy 95% nhưng vô dụng. Ta phải thay thế bằng các độ đo: **Balanced Accuracy, Precision, Recall, F1-Score** hoặc diện tích dưới đường cong **ROC-AUC / PR-AUC**.
19. **Câu 19: Khác biệt giữa Batch Gradient Descent, Stochastic Gradient Descent (SGD) và Mini-batch GD?**
    - *Trả lời*:
      - Batch GD: Dùng toàn bộ $N$ mẫu dữ liệu trong tập train để tính đạo hàm một lần rồi mới cập nhật $\mathbf{W}$. Rất chính xác nhưng cực kỳ chậm khi dữ liệu lớn.
      - Stochastic GD (SGD): Mỗi bước chỉ chọn ngẫu nhiên đúng $1$ mẫu dữ liệu để tính đạo hàm và cập nhật trọng số ngay lập tức. Rất nhanh, giúp thoát khỏi các hố cục bộ, nhưng đường đi đến nghiệm bị dao động (nhiễu).
      - Mini-batch GD: Thỏa hiệp giữa 2 phương pháp trên, mỗi bước lấy một nhóm nhỏ (batch size = 32, 64, 128, 200 mẫu) để cập nhật. Đây là tiêu chuẩn vàng trong thực tế và Deep Learning.
20. **Câu 20: Tốc độ học (Learning Rate - $\eta$) quá lớn hoặc quá nhỏ sẽ dẫn tới hậu quả gì?**
    - *Trả lời*:
      - $\eta$ quá nhỏ: Mỗi bước dịch chuyển của trọng số cực kỳ bé, thuật toán tốn rất nhiều thời gian mới hội tụ hoặc hết số vòng lặp tối đa mà chưa đến đích.
      - $\eta$ quá lớn: Các bước nhảy quá dài, thuật toán có thể nhảy vượt qua điểm tối ưu, dẫn tới hiện tượng hàm mất mát dao động dữ dội hoặc phân kỳ ra vô cùng (`Loss -> inf / NaN`).

---

# PHẦN 5: LỜI KHUYÊN & LỘ TRÌNH ÔN TẬP ĐỂ ĐẠT ĐIỂM CAO

Để tự tin 100% trước bất kỳ buổi bảo vệ thực hành hay thi kết thúc học phần, bạn hãy thực hiện theo đúng 3 bước sau:

1. **Bước 1 — Vẽ lại sơ đồ tư duy (Mindmap)**:
   - Tự lấy một tờ giấy trắng A4, chia làm 4 cột đại diện cho 4 tuần.
   - Viết ra tên thuật toán, dạng bài toán (Hồi quy hay Phân loại), công thức dự đoán cốt lõi và hàm mất mát tương ứng.
2. **Bước 2 — Đọc lướt lại các file notebook của mình**:
   - Mở các file notebook bạn đã làm:
     - `NguyenHaiAnh_24001968_TH1KNN.ipynb` (Week 1)
     - `NguyenHaiAnh_24001968_TH2nb.ipynb` / `TH2bnb.ipynb` (Week 2)
     - `NguyenHaiAnh_24001968_TH3logisticreg.ipynb` (Week 3)
     - `week4/` (Week 4: K-NN multiclass & Softmax MNIST)
   - Với mỗi ô code, hãy chỉ tay vào màn hình và tự giải thích: *"Dòng này đang chuẩn bị dữ liệu gì?", "Hàm này đang tính công thức toán nào trong tài liệu ôn tập?"*.
3. **Bước 3 — Luyện tập 20 câu hỏi vấn đáp**:
   - Nhờ một người bạn hoặc tự đứng trước gương đọc câu hỏi và trả lời to thành tiếng mà không nhìn đáp án. Khi bạn diễn đạt trôi chảy bằng lời của chính mình, kiến thức đó đã thực sự trở thành của bạn!

---
*Chúc Nguyễn Hải Anh ôn tập thật tốt, làm chủ hoàn toàn kiến thức Machine Learning và đạt điểm xuất sắc trong môn học!*
