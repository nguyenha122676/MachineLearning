# GIÁO TRÌNH TOÀN DIỆN: NAÏVE BAYES — ZERO TO HERO
## Từ Trực giác Sơ cấp, Toán học Chuyên sâu đến Làm chủ Kỹ thuật Thực chiến

**Tác giả**: Nguyễn Hải Anh (MSSV: 24001968)  
**Chuyên đề**: Machine Learning — Probabilistic Graphical Models & Classification Algorithms  
**Mục tiêu**: Giúp người học từ người chưa biết gì (Level 0: Zero) trở thành chuyên gia nắm vững bản chất toán học, cài đặt thuật toán từ đầu và làm chủ mô hình trong thực tế (Level 5: Hero).

---

```
                       LỘ TRÌNH CHINH PHỤC "ZERO TO HERO"
  ┌────────────────────────────────────────────────────────────────────────┐
  │ LEVEL 0: THE ZERO       - Trực giác đời sống & Nghịch lý Xác suất Bayes│
  │ LEVEL 1: THE APPRENTICE - Giả thiết "Ngây thơ" & Tính tay bằng giấy bút│
  │ LEVEL 2: THE PRACTITIONER- 3 Biến thể: Gaussian, Multinomial, Bernoulli│
  │ LEVEL 3: THE ENGINEER   - Kỹ thuật số học, Bẫy Underflow & Code Scratch│
  │ LEVEL 4: THE SCIENTIST  - Chứng minh Toán học, Dirichlet Prior & QDA/LDA│
  │ LEVEL 5: THE HERO       - Hybrid Naïve Bayes, Calibration & Streaming  │
  └────────────────────────────────────────────────────────────────────────┘
```

---

## LEVEL 0: THE ZERO — TRỰC GIÁC ĐỜI THƯỜNG & NGHỊCH LÝ BAYES

### 0.1. Bộ não con người học từ quá khứ như thế nào?
Hãy tưởng tượng một buổi sáng bạn bước ra ban công chuẩn bị đi học:
1. Bạn nhìn lên bầu trời: Thấy **mây đen cuồn cuộn** và cảm nhận **gió lạnh thổi mạnh**.
2. Ngay lập tức trong đầu bạn nảy ra suy nghĩ: *"Khả năng cao trời sắp mưa, phải mang theo áo mưa!"*

Quá trình suy nghĩ tưởng như vô thức đó chính là **Suy diễn Bayes (Bayesian Inference)**:
- **Kinh nghiệm quá khứ (Prior)**: Trong cả năm qua ở Hà Nội, số ngày mưa chỉ chiếm khoảng 20%, số ngày không mưa chiếm 80%.
- **Quan sát hiện tại (Evidence / Feature)**: Bầu trời đầy mây đen và gió mạnh.
- **Độ hợp lý (Likelihood)**: Bạn biết rằng *khi trời sắp mưa*, xác suất có mây đen lên tới 90%. Ngược lại, *khi trời tạnh ráo*, xác suất mây đen chỉ khoảng 10%.
- **Kết luận (Posterior)**: Sau khi nhìn thấy mây đen, niềm tin của bạn về việc "trời mưa" tăng vọt từ 20% ban đầu lên trên 70%.

---

### 0.2. Xác suất có điều kiện: Tránh cạm bẫy tư duy lớn nhất
Hai đại lượng sau đây **hoàn toàn khác nhau**, và nhầm lẫn giữa chúng là nguyên nhân gây ra hầu hết các sai lầm trong thống kê và đời sống:
- $P(A \mid B)$: Xác suất xảy ra $A$ **khi biết** $B$ đã xảy ra.
- $P(B \mid A)$: Xác suất xảy ra $B$ **khi biết** $A$ đã xảy ra.

> **Ví dụ kinh điển**:
> - $P(\text{Mây đen} \mid \text{Mưa})$: Nếu trời đang mưa, xác suất có mây đen là gần $100\%$.
> - $P(\text{Mưa} \mid \text{Mây đen})$: Nếu trời có mây đen, xác suất trời mưa chỉ khoảng $40 - 50\%$ (vì rất nhiều ngày trời âm u mây đen nhưng gió thổi mây đi và không mưa).

---

### 0.3. Nghịch lý Xét nghiệm Y tế (The Medical Test Paradox)
Để thấy được sức mạnh kỳ diệu của Định lý Bayes, hãy giải bài toán nổi tiếng sau:

> **Bài toán**:
> Một căn bệnh hiếm gặp có tỷ lệ mắc trong dân số là **0.1%** (tức 1 trong 1000 người bị mắc).
> Có một bộ kít xét nghiệm rất hiện đại:
> - Nếu một người **thực sự bị bệnh**, xét nghiệm cho kết quả **Dương tính với độ chính xác 99%** (Độ nhạy - Sensitivity = 99%).
> - Nếu một người **hoàn toàn khỏe mạnh**, xét nghiệm cho kết quả **Âm tính với độ chính xác 99%** (Độ đặc hiệu - Specificity = 99%, tức chỉ có 1% người khỏe bị Dương tính giả).
>
> Bạn đi khám sức khỏe ngẫu nhiên và nhận được kết quả: **DƯƠNG TÍNH**.  
> **Hỏi: Xác suất bạn thực sự bị bệnh hiểm nghèo này là bao nhiêu phần trăm?**

**Trực giác của 95% mọi người**: Sẽ hoảng loạn và nghĩ rằng mình có **99%** khả năng bị bệnh!  
**Sự thật Toán học theo Định lý Bayes**:
Đặt:
- $B$: Biến cố bị bệnh. $P(B) = 0.001 \Rightarrow P(\bar{B}) = 0.999$ (Khỏe mạnh).
- $D$: Biến cố xét nghiệm Dương tính.
- $P(D \mid B) = 0.99$ (Bị bệnh thì 99% dương tính).
- $P(D \mid \bar{B}) = 0.01$ (Khỏe mạnh nhưng bị dương tính giả 1%).

Theo Định lý Bayes:
$$P(B \mid D) = \frac{P(D \mid B) P(B)}{P(D)} = \frac{P(D \mid B) P(B)}{P(D \mid B) P(B) + P(D \mid \bar{B}) P(\bar{B})}$$
Thay số:
$$P(B \mid D) = \frac{0.99 \times 0.001}{(0.99 \times 0.001) + (0.01 \times 0.999)} = \frac{0.00099}{0.00099 + 0.00999} = \frac{0.00099}{0.01098} \approx \mathbf{9.02\%}!$$

```
                           HÌNH HỌC HÓA 100,000 NGƯỜI DÂN
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ TỔNG DÂN SỐ: 100,000 người                                                  │
 │                                                                             │
 │ 100 người THỰC SỰ BỊ BỆNH:                                                  │
 │   ├─ 99 người xét nghiệm DƯƠNG TÍNH THỰC (True Positive)                   │
 │   └─  1 người xét nghiệm Âm tính giả                                       │
 │                                                                             │
 │ 99,900 người HOÀN TOÀN KHỎE MẠNH:                                           │
 │   ├─ 999 người xét nghiệm DƯƠNG TÍNH GIẢ (False Positive - 1%)              │
 │   └─ 98,901 người xét nghiệm Âm tính thực                                  │
 │                                                                             │
 │ TỔNG SỐ NGƯỜI DƯƠNG TÍNH = 99 (thật) + 999 (giả) = 1098 người               │
 │ XÁC SUẤT BỊ BỆNH THỰC SỰ = 99 / 1098 ≈ 9.02%!                              │
 └─────────────────────────────────────────────────────────────────────────────┘
```

> **Bài học cốt lõi**: **Xác suất tiên nghiệm $P(\text{Bệnh})$ cực kỳ quan trọng!** Vì bệnh quá hiếm, nên trong số những người bị dương tính, số người "khỏe nhưng bị dương tính giả" (999 người) đông gấp 10 lần số người "bị bệnh thật" (99 người).

---

## LEVEL 1: THE APPRENTICE — GIẢ THIẾT "NGÂY THƠ" & TÍNH TAY BẰNG GIẤY BÚT

### 1.1. Bước chuyển dịch từ Bài toán Đơn biến sang Đa biến
Trong bài toán phân loại máy học, đầu vào không chỉ là 1 triệu chứng duy nhất, mà là một vector gồm $d$ đặc trưng:
$$\mathbf{x} = (x_1, x_2, \dots, x_d)$$
Ví dụ phân loại email rác:
- $x_1$: Có từ "khuyến mãi" không?
- $x_2$: Có từ "chuyển tiền" không?
- $x_3$: Có file đính kèm lạ không?
- ...
- $x_{2500}$: Có từ "thân gửi" không?

Công thức Bayes tổng quát:
$$P(y = c_k \mid x_1, x_2, \dots, x_d) = \frac{P(y = c_k) \cdot P(x_1, x_2, \dots, x_d \mid y = c_k)}{P(x_1, x_2, \dots, x_d)}$$

**Bế tắc thực tế**: Làm sao tính được $P(x_1, x_2, \dots, x_d \mid y = c_k)$?
Nếu mỗi từ chỉ có 2 trạng thái (Có/Không), thì tổ hợp của 2500 từ sẽ tạo ra:
$$2^{2500} \approx 10^{752} \text{ khả năng!}$$
*(Trong khi toàn bộ vũ trụ quan sát được chỉ có khoảng $10^{80}$ nguyên tử).* Không có bất kỳ chiếc máy tính nào hay tập dữ liệu nào trên Trái Đất có thể thống kê được xác suất đồng thời này!

---

### 1.2. Giả thiết "Ngây thơ" (Naïve Assumption) cứu rỗi bài toán
Để đập tan sự bùng nổ số chiều, các nhà khoa học đưa ra một giả thiết cực kỳ táo bạo:
> **Giả thiết**: *Khi đã biết nhãn của đối tượng (ví dụ: email này là SPAM), ta xem như sự xuất hiện của từ này hoàn toàn không liên quan gì đến sự xuất hiện của từ kia.*

Toán học hóa:
$$P(x_1, x_2, \dots, x_d \mid y = c_k) = P(x_1 \mid y = c_k) \cdot P(x_2 \mid y = c_k) \dots P(x_d \mid y = c_k) = \prod_{j=1}^d P(x_j \mid y = c_k)$$

- Thay vì phải nhớ $2^{2500}$ tổ hợp, ta chỉ cần nhớ $2500$ con số xác suất riêng lẻ!
- Độ phức tạp tính toán giảm từ **Hàm mũ vô hạn** về **Đường thẳng tuyến tính $\mathcal{O}(d)$**!

---

### 1.3. Thực hành Tính tay từng bước một (Step-by-step Pen & Paper)
Hãy cùng giải bài toán kinh điển phân loại văn bản ẩm thực miền Bắc (B) và Nam (N) từ tài liệu thực hành bằng giấy bút:

**Tập huấn luyện (Training Set):**
- $D_1$: "Hà-Nội phở cháo-lòng Hà-Nội" $\rightarrow$ Lớp **B**
- $D_2$: "Hà-Nội bún-chả phở ô-mai" $\rightarrow$ Lớp **B**
- $D_3$: "Phở bánh-giò ô-mai" $\rightarrow$ Lớp **B**
- $D_4$: "Sài-gòn hủ-tiếu bánh-bò phở" $\rightarrow$ Lớp **N**

**Văn bản cần kiểm thử:**
- $D_6$: "phở hủ-tiếu bánh-bò" $\rightarrow$ **Thuộc lớp B hay N?**

#### Bước 1: Xây dựng Từ điển vựng $V$ (Vocabulary)
Liệt kê toàn bộ các từ xuất hiện trong tập huấn luyện (không trùng lặp):
$$V = \{\text{Hà-Nội (1)}, \text{phở (2)}, \text{cháo-lòng (3)}, \text{bún-chả (4)}, \text{ô-mai (5)}, \text{bánh-giò (6)}, \text{Sài-gòn (7)}, \text{hủ-tiếu (8)}, \text{bánh-bò (9)}\}$$
Kích thước từ điển: $|V| = 9$.

#### Bước 2: Tính Xác suất Tiên nghiệm $P(B)$ và $P(N)$
Có tổng cộng 4 văn bản, trong đó 3 văn bản thuộc lớp B, 1 văn bản thuộc lớp N:
$$P(B) = \frac{3}{4} = 0.75, \quad P(N) = \frac{1}{4} = 0.25$$

#### Bước 3: Đếm số lượng từ trong từng lớp
- **Trong lớp B** ($D_1, D_2, D_3$):
  - Hà-Nội: 2 + 1 + 0 = 3 lần
  - phở: 1 + 1 + 1 = 3 lần
  - cháo-lòng: 1 + 0 + 0 = 1 lần
  - bún-chả: 0 + 1 + 0 = 1 lần
  - ô-mai: 0 + 1 + 1 = 2 lần
  - bánh-giò: 0 + 0 + 1 = 1 lần
  - Sài-gòn: 0, hủ-tiếu: 0, bánh-bò: 0.
  - **Tổng số lượt từ trong lớp B**: $N_B = 3 + 3 + 1 + 1 + 2 + 1 + 0 + 0 + 0 = 11$ từ.
- **Trong lớp N** ($D_4$):
  - Sài-gòn: 1, hủ-tiếu: 1, bánh-bò: 1, phở: 1. Các từ khác: 0.
  - **Tổng số lượt từ trong lớp N**: $N_N = 1 + 1 + 1 + 1 = 4$ từ.

#### Bước 4: Áp dụng làm mịn Laplace ($\alpha = 1$)
Công thức xác suất của từ $w_j$ trong lớp $c$:
$$P(w_j \mid c) = \frac{\text{count}(w_j, c) + 1}{N_c + |V|} = \frac{\text{count}(w_j, c) + 1}{N_c + 9}$$
- Mẫu số lớp B: $11 + 9 = 20$.
- Mẫu số lớp N: $4 + 9 = 13$.

Ta cần tính xác suất của các từ có trong $D_6$ ("phở", "hủ-tiếu", "bánh-bò"):
| Từ khóa trong $D_6$ | $P(\text{từ} \mid B)$ | $P(\text{từ} \mid N)$ |
| :--- | :---: | :---: |
| **phở** | $\frac{3 + 1}{20} = \frac{4}{20}$ | $\frac{1 + 1}{13} = \frac{2}{13}$ |
| **hủ-tiếu** | $\frac{0 + 1}{20} = \frac{1}{20}$ | $\frac{1 + 1}{13} = \frac{2}{13}$ |
| **bánh-bò** | $\frac{0 + 1}{20} = \frac{1}{20}$ | $\frac{1 + 1}{13} = \frac{2}{13}$ |

#### Bước 5: Tính điểm số hậu nghiệm (Posterior Score)
- **Với Lớp B**:
  $$\text{Score}(B) = P(B) \times P(\text{phở} \mid B) \times P(\text{hủ-tiếu} \mid B) \times P(\text{bánh-bò} \mid B)$$
  $$\text{Score}(B) = 0.75 \times \frac{4}{20} \times \frac{1}{20} \times \frac{1}{20} = 0.75 \times 0.2 \times 0.05 \times 0.05 = \mathbf{0.000375} = 3.75 \times 10^{-4}$$
- **Với Lớp N**:
  $$\text{Score}(N) = P(N) \times P(\text{phở} \mid N) \times P(\text{hủ-tiếu} \mid N) \times P(\text{bánh-bò} \mid N)$$
  $$\text{Score}(N) = 0.25 \times \frac{2}{13} \times \frac{2}{13} \times \frac{2}{13} = 0.25 \times \frac{8}{2197} \approx \mathbf{0.000910} = 9.10 \times 10^{-4}$$

#### Bước 6: Chuẩn hóa và Kết luận
Chuẩn hóa về phần trăm xác suất:
$$P(B \mid D_6) = \frac{0.000375}{0.000375 + 0.000910} = \frac{0.000375}{0.001285} \approx \mathbf{29.18\%}$$
$$P(N \mid D_6) = \frac{0.000910}{0.000375 + 0.000910} = \frac{0.000910}{0.001285} \approx \mathbf{70.82\%}$$

$\Rightarrow$ **Kết luận chính xác**: Vì $70.82\% > 29.18\%$, văn bản $D_6$ được phân loại vào **Miền Nam (N)**!

---

## LEVEL 2: THE PRACTITIONER — 3 BIẾN THỂ CHỦ ĐẠO

Mỗi dạng dữ liệu trong thực tế đòi hỏi một phân bố toán học tương ứng:

```
                            BẢN ĐỒ DỮ LIỆU VÀ BIẾN THỂ
                                DỮ LIỆU ĐẦU VÀO
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
   SỐ THỰC LIÊN TỤC             TẦN SUẤT ĐẾM TỪ               NHỊ PHÂN 0 / 1
(Chiều dài, Cân nặng, ...)     (Văn bản, Đơn hàng, ...)     (Có/Không, Bệnh tật, ...)
         │                             │                             │
         ▼                             ▼                             ▼
   GAUSSIAN NB                  MULTINOMIAL NB                 BERNOULLI NB
f(x) = Gauss(μ, σ²)             P(x) ~ Π p_j^(x_j)           P(x) ~ Π p_j^x (1-p_j)^(1-x)
```

### 2.1. Gaussian Naïve Bayes (Liên tục)
- **Đặc trưng**: $x_j \in \mathbb{R}$.
- **Hàm mật độ xác suất**:
  $$\mathcal{N}(x_j; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(x_j - \mu)^2}{2\sigma^2} \right)$$
- **Cách tính $\mu$ và $\sigma$**:
  - Kỳ vọng $\mu$: Trung bình cộng các giá trị của thuộc tính $j$ trong lớp đó.
  - Phương sai $\sigma^2$: Độ biến thiên bình phương quanh giá trị trung bình.
- **Thực hành với Iris**: Khi dùng 4 thuộc tính (đài hoa, cánh hoa), Setosa có độ dài cánh đài trung bình $\mu = 1.48$ cm, trong khi Virginica là $\mu = 5.58$ cm $\rightarrow$ mô hình phân biệt tách bạch với độ chính xác **96.67%**.

### 2.2. Multinomial Naïve Bayes (Đếm tần suất)
- **Đặc trưng**: Vector số đếm $\mathbf{x} \in \{0, 1, 2, \dots\}^d$.
- **Đặc điểm nổi bật**: Rất nhạy bén với tần suất lặp lại. Nếu từ "giảm giá" xuất hiện 10 lần trong email, trọng số nghi ngờ email rác sẽ tăng lũy thừa $p^{10}$.

### 2.3. Bernoulli Naïve Bayes (Nhị phân 0/1)
- **Đặc trưng**: Vector chỉ báo $\mathbf{x} \in \{0, 1\}^d$.
- **Quy tắc phạt vắng mặt**:
  Nếu bệnh nhân **không bị ho** ($x_{\text{ho}} = 0$), thì thừa số $(1 - P(\text{ho} \mid \text{Covid})) = (1 - 0.9) = 0.1$ sẽ làm giảm mạnh khả năng bệnh nhân bị Covid! Đây là điều mà Multinomial NB hoàn toàn không làm được.

---

## LEVEL 3: THE ENGINEER — KỸ THUẬT SỐ HỌC, BẪY UNDERFLOW & TỰ VIẾT CODE

### 3.1. Bẫy tràn số dưới số thực (Numerical Underflow Trap)
Khi lập trình mô hình Naïve Bayes cho bài toán có 1000 thuộc tính:
```python
# CODE NGUY HIỂM (SAI LẦM PHỔ BIẾN):
score = prior
for p in likelihoods:
    score *= p  # Nhân liên tiếp 1000 số có giá trị ~ 0.001
# Kết quả: score = 0.0 (Bị tràn số dưới, máy tính chịu thua!)
```

**Giải pháp của Kỹ sư chuyên nghiệp**: Chuyển toàn bộ sang miền Logarit:
$$\ln(A \times B \times C) = \ln(A) + \ln(B) + \ln(C)$$
$$\text{Log-Score}(c_k) = \ln P(y = c_k) + \sum_{j=1}^d \ln P(x_j \mid y = c_k)$$

```python
# CODE CHUẨN MỰC AN TOÀN SỐ HỌC:
log_score = np.log(prior) + np.sum(np.log(likelihoods))
```

### 3.2. Toàn bộ mã nguồn Gaussian Naïve Bayes Tự Viết (Scratch)
Dưới đây là class `GaussianNBScratch` hoàn chỉnh, chuẩn PEP-8, chạy tương đương Scikit-Learn:

```python
import numpy as np

class GaussianNBScratch:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes = None
        self.means = {}
        self.vars = {}
        self.priors = {}

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        
        for c in self.classes:
            X_c = X[y == c]
            # 1. Tính Prior P(y = c)
            self.priors[c] = X_c.shape[0] / n_samples
            # 2. Tính kỳ vọng mu
            self.means[c] = np.mean(X_c, axis=0)
            # 3. Tính phương sai sigma^2 + var_smoothing chống chia cho 0
            self.vars[c] = np.var(X_c, axis=0) + self.var_smoothing * np.max(np.var(X, axis=0))

    def _calculate_log_likelihood(self, x, c):
        mean = self.means[c]
        var = self.vars[c]
        # ln f(x) = -0.5 * ln(2*pi*var) - ((x - mean)^2) / (2*var)
        log_pdf = -0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2) / (2 * var)
        return np.sum(log_pdf)

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes:
                log_prior = np.log(self.priors[c])
                log_likelihood = self._calculate_log_likelihood(x, c)
                posteriors.append(log_prior + log_likelihood)
            # Chọn lớp có Posterior Log-Score lớn nhất
            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)
```

---

## LEVEL 4: THE SCIENTIST — TOÁN HỌC CHUYÊN SÂU & DẪN XUẤT LÝ THUYẾT

### 4.1. Bản chất suy diễn Bayes của Laplace Smoothing
Tại sao lại cộng $\alpha$ vào tử số và $\alpha |V|$ vào mẫu số?
Trong lý thuyết Bayes:
- Likelihood của từ vựng tuân theo phân bố Đa thức (Multinomial Distribution).
- **Phân bố tiên nghiệm liên hợp (Conjugate Prior)** của phân bố Đa thức là **Phân bố Dirichlet**:
  $$\operatorname{Dir}(\mathbf{p} \mid \boldsymbol{\alpha}) = \frac{1}{\mathrm{B}(\boldsymbol{\alpha})} \prod_{j=1}^{|V|} p_j^{\alpha_j - 1}$$
- Khi ta không có bất kỳ định kiến nào về từ vựng (tiên nghiệm đều Uniform Dirichlet), ta đặt tất cả $\alpha_j = \alpha + 1$.
- Xác suất hậu nghiệm của tham số:
  $$P(\mathbf{p} \mid \mathcal{D}) \propto \text{Likelihood} \times \text{Prior} = \left(\prod_{j=1}^{|V|} p_j^{N_j}\right) \left(\prod_{j=1}^{|V|} p_j^\alpha\right) = \prod_{j=1}^{|V|} p_j^{N_j + \alpha}$$
- Lấy kỳ vọng của phân bố Dirichlet hậu nghiệm này (hoặc cực đại hóa MAP):
  $$\mathbb{E}[p_j] = \frac{N_j + \alpha}{\sum_{k=1}^{|V|} (N_k + \alpha)} = \frac{N_j + \alpha}{N + \alpha |V|}$$
$\Rightarrow$ **Làm mịn Laplace chính là suy diễn Bayes thuần khiết (Pure Bayesian Inference)!**

---

### 4.2. Mối liên hệ hình học: Naïve Bayes, LDA và QDA
Xét bài toán phân loại 2 lớp Gaussian. Ranh giới quyết định (Decision Boundary) là tập hợp các điểm mà xác suất 2 lớp bằng nhau:
$$\ln \frac{P(y = 1 \mid \mathbf{x})}{P(y = 0 \mid \mathbf{x})} = 0 \iff \ln \frac{P(y = 1)}{P(y = 0)} + \ln \frac{f(\mathbf{x} \mid y = 1)}{f(\mathbf{x} \mid y = 0)} = 0$$

Khai triển hàm mật độ Gauss đa biến:
$$-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_1)^T \mathbf{\Sigma}_1^{-1} (\mathbf{x} - \boldsymbol{\mu}_1) + \frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_0)^T \mathbf{\Sigma}_0^{-1} (\mathbf{x} - \boldsymbol{\mu}_0) + \frac{1}{2} \ln \frac{|\mathbf{\Sigma}_0|}{|\mathbf{\Sigma}_1|} + \ln \frac{P(y=1)}{P(y=0)} = 0$$

```
                         HỆ THỐNG MÔ HÌNH PHÂN BIỆT GAUSS
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ 1. QDA (Quadratic Discriminant Analysis):                                   │
 │    - Σ_1 ≠ Σ_0 (Ma trận đầy đủ, tự do)                                      │
 │    - Ranh giới: BẬC HAI PHI TUYẾN TÍNH (Parabol / Elip / Hyperbol)          │
 │                                                                             │
 │ 2. GAUSSIAN NAÏVE BAYES (General):                                          │
 │    - Σ_1 ≠ Σ_0 nhưng bắt buộc phải là ĐƯỜNG CHÉO (Diagonal: Cov(X_i,X_j)=0) │
 │    - Ranh giới: BẬC HAI TRỤC SONG SONG (Axis-aligned Quadratic)             │
 │                                                                             │
 │ 3. LDA (Linear Discriminant Analysis):                                      │
 │    - Σ_1 = Σ_0 = Σ (Dùng chung ma trận hiệp phương sai đầy đủ)              │
 │    - Thành phần bậc hai x^T Σ^-1 x tự triệt tiêu lẫn nhau!                 │
 │    - Ranh giới: ĐƯỜNG THẲNG TUYẾN TÍNH (Hyperplane)                         │
 │                                                                             │
 │ 4. LINEAR NAÏVE BAYES:                                                      │
 │    - Σ_1 = Σ_0 và là ma trận ĐƯỜNG CHÉO.                                    │
 │    - Ranh giới: SIÊU PHẲNG TUYẾN TÍNH ĐƠN GIẢN                              │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

### 4.3. Định lý Asymptotic Sample Complexity (Ng & Jordan, NIPS 2001)
Tại sao Naïve Bayes lại được khuyên dùng khi dữ liệu ít, còn Logistic Regression dùng khi dữ liệu nhiều?
Giáo sư Andrew Ng (Stanford) và Michael Jordan (UC Berkeley) đã chứng minh:
- Naïve Bayes (Generative) đạt đến sai số tiệm cận thấp nhất chỉ cần:
  $$N = \mathcal{O}(\log d) \text{ mẫu dữ liệu}$$
- Logistic Regression (Discriminative) cần tới:
  $$N = \mathcal{O}(d) \text{ mẫu dữ liệu}$$
- Khi $d = 10,000$ từ vựng, Naïve Bayes chỉ cần vài chục đến vài trăm văn bản là đã học rất tốt, trong khi Logistic Regression cần hàng chục nghìn văn bản mới không bị overfit!

---

## LEVEL 5: THE HERO — HYBRID NAÏVE BAYES, CALIBRATION & STREAMING

### 5.1. Mô hình Hỗn hợp Đa biến (Hybrid Naïve Bayes Architecture)
Trong các hệ thống phân tích tài chính hoặc bệnh viện, mỗi hồ sơ bệnh nhân là một cấu trúc hỗn hợp:
- $X_{\text{tuổi}}, X_{\text{huyết áp}}$: Số thực liên tục $\rightarrow$ Dùng **Gaussian PDF**.
- $X_{\text{hút thuốc}}, X_{\text{tiểu đường}}$: Nhị phân $\rightarrow$ Dùng **Bernoulli PMF**.
- $X_{\text{số lần nhập viện}}$: Biến đếm không âm $\rightarrow$ Dùng **Poisson PMF**: $P(k) = \frac{\lambda^k e^{-\lambda}}{k!}$.
- $X_{\text{nhóm máu}}$: Danh mục 4 trạng thái (A, B, AB, O) $\rightarrow$ Dùng **Categorical PMF**.

**Hàm mục tiêu Hỗn hợp Hợp nhất (Unified Log-Likelihood)**:
$$\hat{y} = \arg\max_{c_k} \left[ \ln P(y = c_k) + \sum_{j \in \text{Gauss}} \ln \mathcal{N}(x_j) + \sum_{j \in \text{Bern}} \ln \mathcal{B}(x_j) + \sum_{j \in \text{Pois}} \ln \mathcal{P}(x_j) + \sum_{j \in \text{Cat}} \ln \theta_j \right]$$
Toàn bộ hệ thống chạy song song trên CPU, tốc độ suy luận hàng triệu bản ghi mỗi giây.

---

### 5.2. Hiệu chuẩn xác suất (Probability Calibration)
Một nhược điểm cố hữu của Naïve Bayes: **Xác suất dự đoán bị phân cực quá mức (Poorly Calibrated)**.
- Khi Naïve Bayes dự đoán $P(y = 1 \mid \mathbf{x}) = 0.999$, trên thực tế tần suất đúng của nó có thể chỉ đạt khoảng $85\%$. Nguyên nhân là việc nhân liên tiếp hàng chục giả thiết độc lập đã "khuếch đại" niềm tin lên quá cao.
- **Giải pháp Hero**: Dùng **Platt Scaling (Logistic Calibration)** hoặc **Isotonic Regression** bọc bên ngoài Naïve Bayes:
```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB

base_gnb = GaussianNB()
# Hiệu chuẩn xác suất bằng phương pháp Sigmoid (Platt Scaling) qua 5-Fold CV
calibrated_gnb = CalibratedClassifierCV(estimator=base_gnb, method='sigmoid', cv=5)
calibrated_gnb.fit(X_train, y_train)

# Xác suất sau hiệu chuẩn phản ánh chính xác tỷ lệ rủi ro thực tế!
reliable_probabilities = calibrated_gnb.predict_proba(X_test)
```

---

### 5.3. Học tăng dần trực tuyến (Online Incremental Learning with `partial_fit`)
Với hệ thống lớn hàng triệu đơn hàng mỗi ngày, việc huấn luyện lại từ đầu mô hình là bất khả thi. Naïve Bayes hỗ trợ cập nhật động trong thời gian thực:

```python
from sklearn.naive_bayes import MultinomialNB
import numpy as np

clf = MultinomialNB()
classes = np.array([0, 1])

# Giả lập luồng dữ liệu truyền vào theo từng đợt (Streaming batches)
for batch_X, batch_y in streaming_data_generator():
    # Cập nhật tham số mô hình ngay lập tức mà không quên kiến thức cũ
    clf.partial_fit(batch_X, batch_y, classes=classes)
```

---

## TỔNG KẾT BẢN ĐỒ TƯ DUY (NAÏVE BAYES CHEAT SHEET)

| Bài toán | Kiểu dữ liệu | Mô hình khuyên dùng | Tham số chính | Mẹo thực chiến |
| :--- | :--- | :--- | :--- | :--- |
| **Y tế, Cảm biến, Iris** | Số thực liên tục | `GaussianNB` | `var_smoothing=1e-9` | Chuẩn hóa dữ liệu về phân bố chuẩn (PowerTransformer) nếu bị lệch (skewed). |
| **Phân loại văn bản, Spam** | Tần suất đếm từ | `MultinomialNB` | `alpha=1.0` (Laplace) | Dùng TF-IDF hoặc CountVectorizer, loại bỏ Stopwords. |
| **Triệu chứng, Chữ số đen trắng** | Nhị phân 0/1 | `BernoulliNB` | `alpha=1.0`, `binarize=0.0` | Thích hợp cho văn bản ngắn (tweet, tin nhắn SMS). |
| **Khảo sát, Phân khúc khách hàng** | Danh mục rời rạc | `CategoricalNB` | `alpha=1.0` | Sử dụng OrdinalEncoder hoặc One-Hot + MultinomialNB. |
| **Dữ liệu lớn thời gian thực** | Bất kỳ | `partial_fit()` | Batch stream | Cực kỳ tiết kiệm RAM, đáp ứng tức thời. |
