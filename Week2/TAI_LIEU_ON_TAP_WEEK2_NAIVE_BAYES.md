# GIÁO TRÌNH ÔN TẬP TOÀN DIỆN TUẦN 2 & 2B (WEEK 2)
## MÔ HÌNH PHÂN LOẠI XÁC SUẤT NAÏVE BAYES (GAUSSIAN, MULTINOMIAL, BERNOULLI)

> **Sinh viên**: Nguyễn Hải Anh (MSSV: 24001968)  
> **Học phần**: Machine Learning (Học máy)  
> **Chủ đề**: Mô hình phân loại dựa trên xác suất (Probabilistic Generative Classifier)  
> **Thư mục liên quan**: `d:\MachineLearning\Week2` và `d:\MachineLearning\week2b`  
> **Các bài tập thực hành**: Phân loại hoa Iris, Ung thư vú Wisconsin, Lọc thư rác Spam Email (ex6DataPrepared), Phân loại đơn hàng (`multinomial_nb_orders.csv`), Triệu chứng bệnh lý (`bernoulli_nb_symptoms.csv`).

---

## MỤC LỤC
1. [Bản chất bài toán Phân loại (Classification) & Triết lý Xác suất](#1-bản-chất-bài-toán-phân-loại-classification--triết-lý-xác-suất)
2. [Định lý Bayes & Suy diễn thống kê (Bayesian Inference)](#2-định-lý-bayes--suy-diễn-thống-kê-bayesian-inference)
   - 2.1. Phân biệt xác suất có điều kiện $P(A \mid B)$ và $P(B \mid A)$
   - 2.2. Bốn thành phần cốt lõi: Prior, Likelihood, Evidence, Posterior
   - 2.3. Nguyên lý Ước lượng Hậu nghiệm Cực đại (MAP - Maximum A Posteriori)
3. [Giả thiết "Ngây thơ" (Conditional Independence Assumption)](#3-giả-thiết-ngây-thơ-conditional-independence-assumption)
   - Tại sao lại "Ngây thơ"? Ưu thế giảm số lượng tham số từ hàm mũ về tuyến tính
4. [Ba biến thể kinh điển của Naïve Bayes](#4-ba-biến-thể-kinh-điển-của-naïve-bayes)
   - 4.1. Gaussian Naïve Bayes (GNB): Dành cho thuộc tính liên tục
   - 4.2. Multinomial Naïve Bayes (MNB): Dành cho số đếm tần suất (Bag-of-Words)
   - 4.3. Bernoulli Naïve Bayes (BNB): Dành cho biến nhị phân có/không
5. [Hai kỹ thuật số học "sống còn" trong cài đặt](#5-hai-kỹ-thuật-số-học-sống-còn-trong-cài-đặt)
   - 5.1. Bẫy Underflow và Kỹ thuật Log-trick
   - 5.2. Hiện tượng Zero-Probability & Kỹ thuật Làm trơn Laplace (Laplace Smoothing)
6. [Giải phẫu mã nguồn từng dòng (Line-by-Line Code Breakdown)](#6-giải-phẫu-mã-nguồn-từng-dòng-line-by-line-code-breakdown)
   - Tự viết Gaussian NB bằng NumPy
   - Tự viết Multinomial NB bằng NumPy
   - Sử dụng thư viện `scikit-learn` chuẩn xác
7. [Chi tiết các bài tập thực hành Tuần 2 & 2b](#7-chi-tiết-các-bài-tập-thực-hành-tuần-2--2b)
8. [Bộ 10 câu hỏi vấn đáp bảo vệ bài tập Tuần 2 kèm đáp án chuẩn](#8-bộ-10-câu-hỏi-vấn-đáp-bảo-vệ-bài-tập-tuần-2-kèm-đáp-án-chuẩn)

---

## 1. BẢN CHẤT BÀI TOÁN PHÂN LOẠI (CLASSIFICATION) & TRIẾT LÝ XÁC SUẤT

Trong Tuần 1, bạn đã làm quen với bài toán Hồi quy (dự đoán số thực liên tục như giá nhà, điểm GPA).  
Từ Tuần 2, chúng ta chính thức bước sang **Bài toán Phân loại (Classification)**:
- Đầu ra $y$ là một **nhãn rời rạc** thuộc tập hợp hữu hạn $C$ lớp: $y \in \{c_1, c_2, \dots, c_C\}$.
  - Nhị phân ($C=2$): Email là Spam ($1$) hay Bình thường ($0$); Khối u Lành tính ($0$) hay Ác tính ($1$).
  - Đa lớp ($C > 2$): Phân loại hoa Iris vào 3 loài (Setosa, Versicolor, Virginica); Nhận diện loại tin tức thuộc 20 chủ đề.

### Hai trường phái phân loại trong Machine Learning:
1. **Mô hình Phân biệt (Discriminative Models)**: Cố gắng vẽ ra một ranh giới (Decision Boundary) ngăn cách trực tiếp giữa các lớp (ví dụ: Logistic Regression, SVM, Cây quyết định).
2. **Mô hình Sinh (Generative Models)**: Không vội vẽ ranh giới, mà tìm cách mô hình hóa phân phối xác suất của từng lớp riêng biệt: *"Dữ liệu của người bị ung thư thường trông như thế nào? Dữ liệu của người khỏe mạnh trông như thế nào?"*. Sau đó dùng Định lý Bayes để tính xác suất một người rơi vào lớp nào.  
👉 **Naïve Bayes chính là vị vua tiêu biểu của trường phái Mô hình Sinh!**

---

## 2. ĐỊNH LÝ BAYES & SUY DIỄN THỐNG KÊ (BAYESIAN INFERENCE)

### 2.1. Phân biệt xác suất có điều kiện: Tránh cạm bẫy tư duy lớn nhất
Hai đại lượng sau đây **hoàn toàn khác biệt**:
- $P(A \mid B)$: Xác suất xảy ra $A$ **khi đã biết** $B$ xảy ra.
- $P(B \mid A)$: Xác suất xảy ra $B$ **khi đã biết** $A$ xảy ra.

> **Ví dụ kinh điển**:
> - $P(\text{Mây đen} \mid \text{Mưa})$: Nếu trời đang mưa, xác suất có mây đen là gần $100\%$.
> - $P(\text{Mưa} \mid \text{Mây đen})$: Nếu nhìn thấy mây đen, xác suất trời mưa chỉ tầm $40 - 50\%$ (vì có rất nhiều ngày âm u mây đen nhưng gió thổi đi không hề mưa).

Trong Machine Learning:
- Bác sĩ biết: $P(\text{Triệu chứng} \mid \text{Bệnh})$ (dễ dàng thống kê từ các ca bệnh trong sách y khoa).
- Nhưng bệnh nhân cần biết: $P(\text{Bệnh} \mid \text{Triệu chứng})$ ("Tôi đang có triệu chứng này, vậy xác suất tôi bị bệnh là bao nhiêu?").
- **Định lý Bayes chính là cây cầu toán học giúp đảo ngược điều kiện đó!**

---

### 2.2. Bốn thành phần cốt lõi của Định lý Bayes

$$P(y = c \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid y = c) \cdot P(y = c)}{P(\mathbf{x})}$$

```
                CẤU TRÚC ĐỊNH LÝ BAYES TRONG HỌC MÁY
                
                      Likelihood (Độ hợp lý)       Prior (Tiên nghiệm)
                        ┌──────────────┐         ┌──────────────┐
                        │ P(x | y = c) │    ×    │   P(y = c)   │
   Posterior            └──────────────┘         └──────────────┘
  (Hậu nghiệm)      = ─────────────────────────────────────────────
┌──────────────┐                    ┌──────────────┐
│ P(y = c | x) │                    │     P(x)     │
└──────────────┘                    └──────────────┘
                                  Evidence (Bằng chứng)
```

1. **$P(y = c)$ (Prior Probability - Xác suất Tiên nghiệm)**:
   - Niềm tin ban đầu về việc lớp $c$ xảy ra khi chưa quan sát bất kỳ thuộc tính nào.
   - Ước lượng đơn giản từ tập train: $P(y = c) = \frac{\text{Số mẫu lớp } c}{\text{Tổng số mẫu}}$.
2. **$P(\mathbf{x} \mid y = c)$ (Likelihood - Độ hợp lý)**:
   - Nếu một mẫu thực sự thuộc lớp $c$, xác suất mẫu đó có vector đặc trưng $\mathbf{x}$ là bao nhiêu?
3. **$P(\mathbf{x})$ (Evidence - Bằng chứng)**:
   - Xác suất xuất hiện vector đặc trưng $\mathbf{x}$ trên toàn bộ quần thể.  
     Theo công thức xác suất toàn phần: $P(\mathbf{x}) = \sum_{k=1}^C P(\mathbf{x} \mid y = k) P(y = k)$.
   - **Lưu ý quan trọng**: Với một mẫu $\mathbf{x}$ đang xét, $P(\mathbf{x})$ là một hằng số giống hệt nhau đối với mọi lớp $c$.
4. **$P(y = c \mid \mathbf{x})$ (Posterior Probability - Xác suất Hậu nghiệm)**:
   - Mục tiêu cuối cùng! Xác suất mẫu $\mathbf{x}$ thuộc về lớp $c$ **sau khi** ta đã nhìn thấy các đặc trưng của nó.

---

### 2.3. Nguyên lý Ước lượng Hậu nghiệm Cực đại (MAP - Maximum A Posteriori)

Khi cần dự đoán nhãn cho mẫu mới $\mathbf{x}$, ta tính xác suất hậu nghiệm của nó với tất cả các lớp $c \in \{1, 2, \dots, C\}$ và chọn lớp có **xác suất lớn nhất**:
$$\hat{y} = \arg\max_{c} P(y = c \mid \mathbf{x}) = \arg\max_{c} \frac{P(\mathbf{x} \mid y = c) P(y = c)}{P(\mathbf{x})}$$

Vì mẫu số $P(\mathbf{x}) > 0$ là giống nhau cho mọi lớp $c$, ta có thể **bỏ qua hoàn toàn mẫu số**:
$$\hat{y} = \arg\max_{c} \Big[ P(\mathbf{x} \mid y = c) \cdot P(y = c) \Big]$$

---

## 3. GIẢ THIẾT "NGÂY THƠ" (CONDITIONAL INDEPENDENCE ASSUMPTION)

### 3.1. Nghịch lý số chiều & Tại sao phải "Ngây thơ"?
Vector đặc trưng $\mathbf{x}$ gồm $d$ thuộc tính: $\mathbf{x} = [x_1, x_2, \dots, x_d]^T$.  
Để tính xác suất đồng thời:
$$P(\mathbf{x} \mid y = c) = P(x_1, x_2, \dots, x_d \mid y = c)$$
Nếu mỗi thuộc tính chỉ đơn giản nhận 2 giá trị ($0$ hoặc $1$), thì tổ hợp của $d$ thuộc tính sẽ có $2^d$ trạng thái khả dĩ.  
- Nếu văn bản có $d = 1000$ từ, máy tính phải ước lượng $2^{1000}$ tham số xác suất (nhiều hơn số hạt nguyên tử trong vũ trụ!). Không một bộ dữ liệu nào trên đời đủ lớn để chứa hết các trường hợp này.

### 3.2. Cứu tinh: Giả thiết độc lập có điều kiện
Naïve Bayes giải quyết vấn đề nan giải trên bằng một giả định cực kỳ táo bạo:  
> **"Tất cả các đặc trưng $x_1, x_2, \dots, x_d$ hoàn toàn độc lập với nhau khi đã biết nhãn $y$."**

Khi các biến độc lập, xác suất của một tích bằng tích các xác suất:
$$P(x_1, x_2, \dots, x_d \mid y = c) = P(x_1 \mid y = c) \times P(x_2 \mid y = c) \times \dots \times P(x_d \mid y = c) = \prod_{j=1}^d P(x_j \mid y = c)$$

Nhờ giả thiết này:
- Số lượng tham số cần ước lượng giảm từ hàm mũ $\mathcal{O}(C \cdot 2^d)$ xuống còn tuyến tính $\mathcal{O}(C \cdot d)$!
- Ta chỉ cần tính riêng rẽ xác suất của từng thuộc tính một trên mỗi lớp.
- Mặc dù ngoài đời các từ hay triệu chứng thường phụ thuộc nhau (ví dụ từ "học" rất hay đi kèm từ "máy"), nhưng Naïve Bayes vẫn hoạt động cực kỳ xuất sắc vì mục đích phân loại chỉ là tìm ra lớp nào có điểm số cao hơn, chứ không cần con số xác suất tuyệt đối phải hoàn hảo 100%.

---

## 4. BA BIẾN THỂ KINH ĐIỂN CỦA NAÏVE BAYES

Tùy thuộc vào bản chất vật lý của các đặc trưng đầu vào $X$, Naïve Bayes được chia thành 3 dòng chính:

### 4.1. Gaussian Naïve Bayes (GNB)
- **Áp dụng khi**: Đặc trưng là **biến số thực liên tục** ($x_j \in \mathbb{R}$), ví dụ: Chiều dài cánh hoa (cm), Nồng độ cồn, Huyết áp, Độ tuổi.
- **Giả định**: Trong mỗi lớp $c$, đặc trưng $x_j$ tuân theo **Phân phối chuẩn Gauss (Gaussian Distribution)** với kỳ vọng $\mu_{cj}$ và phương sai $\sigma_{cj}^2$:
  $$x_j \mid y = c \sim \mathcal{N}(\mu_{cj}, \sigma_{cj}^2)$$
- **Hàm mật độ xác suất (PDF)**:
  $$P(x_j \mid y = c) = \frac{1}{\sqrt{2\pi \sigma_{cj}^2}} \exp\left( -\frac{(x_j - \mu_{cj})^2}{2\sigma_{cj}^2} \right)$$
- **Cách học tham số từ tập Train**:
  - Với mỗi lớp $c$ và mỗi đặc trưng $j$:
    $$\mu_{cj} = \frac{1}{N_c} \sum_{i \in \text{lớp } c} x_{ij} \quad (\text{Trung bình cộng mẫu})$$
    $$\sigma_{cj}^2 = \frac{1}{N_c} \sum_{i \in \text{lớp } c} (x_{ij} - \mu_{cj})^2 + \epsilon \quad (\text{Phương sai mẫu, cộng } \epsilon \text{ để tránh chia 0})$$

---

### 4.2. Multinomial Naïve Bayes (MNB)
- **Áp dụng khi**: Đặc trưng là **số đếm tần suất** ($x_j \in \{0, 1, 2, 3, \dots\}$), biểu thị số lần một từ xuất hiện trong văn bản (Mô hình Túi từ - Bag of Words).
- **Ví dụ**: Email có 3 lần xuất hiện từ "khuyến mãi", 5 lần từ "tiền mặt", 0 lần từ "học phí".
- **Công thức hợp lý**:
  $$P(\mathbf{x} \mid y = c) \propto \prod_{j=1}^d (p_{cj})^{x_j}$$
  Trong đó $p_{cj}$ là xác suất từ thứ $j$ xuất hiện trong một văn bản thuộc lớp $c$:
  $$p_{cj} = \frac{N_{cj} + \alpha}{N_c + \alpha \cdot d}$$
  - $N_{cj}$: Tổng số lần từ $j$ xuất hiện trong tất cả văn bản của lớp $c$.
  - $N_c = \sum_{j=1}^d N_{cj}$: Tổng số từ của toàn bộ lớp $c$.
  - $d$: Tổng số từ vựng trong từ điển.
  - $\alpha$: Hệ số làm trơn Laplace ($\alpha = 1$).

---

### 4.3. Bernoulli Naïve Bayes (BNB)
- **Áp dụng khi**: Đặc trưng là **biến nhị phân chỉ có giá trị 0 hoặc 1** ($x_j \in \{0, 1\}$).
- **Ví dụ**:
  - Trong văn bản: Từ $j$ có xuất hiện ($1$) hay không xuất hiện ($0$), bất kể xuất hiện bao nhiêu lần.
  - Trong y tế: Bệnh nhân có bị sốt ($1$) hay không ($0$), có bị ho ($1$) hay không ($0$).
- **Công thức hợp lý**:
  $$P(\mathbf{x} \mid y = c) = \prod_{j=1}^d \Big[ p_{cj}^{x_j} (1 - p_{cj})^{1 - x_j} \Big]$$
  - Nếu từ $j$ xuất hiện ($x_j = 1$): Nhân thêm xác suất $p_{cj}$.
  - Nếu từ $j$ KHÔNG xuất hiện ($x_j = 0$): Nhân thêm xác suất $(1 - p_{cj})$.
- **SỰ KHÁC BIỆT CHẾT NGƯỜI GIỮA MULTINOMIAL VÀ BERNOULLI**:
  - Multinomial NB **chỉ quan tâm** đến những từ xuất hiện trong văn bản ($x_j > 0$). Từ nào không có mặt ($x_j = 0$) thì số mũ bằng 0 $\implies p_{cj}^0 = 1$ (bỏ qua).
  - Bernoulli NB **phạt cả sự vắng mặt** của từ! Nếu một email Spam mà lại vắng mặt các từ thường gặp của thư rác, Bernoulli NB sẽ nhân thêm xác suất $(1 - p_{cj})$ để hạ điểm lớp Spam xuống.

---

## 5. HAI KỸ THUẬT SỐ HỌC "SỐNG CÒN" TRONG CÀI ĐẶT

### 5.1. Bẫy Underflow và Kỹ thuật Log-trick
- **Nguy cơ**: Mỗi xác suất $P(x_j \mid y)$ là một số nhỏ hơn 1 (ví dụ $0.005$). Khi nhân 500 số nhỏ hơn 1 lại với nhau:
  $$\prod_{j=1}^{500} 0.005 \approx 10^{-1151} \implies \text{Máy tính coi là } 0.0 \text{ (Tràn số dưới - Underflow)!}$$
  Khi tất cả các lớp đều bị bằng `0.0`, mô hình hoàn toàn bất lực và không thể so sánh được!
- **Giải pháp Log-trick**:
  Áp dụng hàm logarit tự nhiên ($\ln$) vào cả hai vế. Vì hàm logarit đơn điệu tăng nên $\arg\max f(x) \equiv \arg\max \ln f(x)$.  
  Phép nhân biến thành phép cộng:
  $$\ln \left[ P(y=c) \prod_{j=1}^d P(x_j \mid y=c) \right] = \ln P(y=c) + \sum_{j=1}^d \ln P(x_j \mid y=c)$$
  Máy tính chỉ việc cộng các số âm (như $-3.2 + (-5.1) + \dots$), không bao giờ lo bị tràn số!

---

### 5.2. Hiện tượng Zero-Probability & Kỹ thuật Làm trơn Laplace (Laplace Smoothing)
- **Nguy cơ**: Giả sử trong tập train, từ "bitcoin" chưa từng xuất hiện trong email thông thường ($y = \text{Ham}$). Khi đó $P(\text{"bitcoin"} \mid \text{Ham}) = 0$.
  Hôm nay, một người bạn gửi email công việc có nhắc: *"Tớ vừa đọc tin về bitcoin"*.  
  Tích xác suất sẽ bị:
  $$P(\text{email} \mid \text{Ham}) = P(\text{"tớ"} \mid \text{Ham}) \times \dots \times \mathbf{0} \times \dots = 0$$
  Chỉ vì một từ duy nhất chưa từng thấy, cơ hội của lớp Ham bị biến thành $0$ tuyệt đối, ép email đó bị xếp nhầm vào thư rác!
- **Giải pháp Laplace Smoothing (Add-one smoothing)**:
  Coi như mọi từ trong từ điển đều đã từng xuất hiện ít nhất $\alpha$ lần ($\alpha = 1$):
  $$p_{cj} = \frac{N_{cj} + \alpha}{N_c + \alpha \cdot d}$$
  Nhờ có tử số $+1$, xác suất sẽ không bao giờ bằng $0$. Nếu từ chưa từng xuất hiện ($N_{cj} = 0$), nó vẫn nhận được một xác suất nhỏ $\frac{1}{N_c + d}$.

---

## 6. GIẢI PHẪU MÃ NGUỒN TỪNG DÒNG (LINE-BY-LINE CODE BREAKDOWN)

### A. Tự cài đặt Gaussian Naïve Bayes bằng NumPy từ đầu
```python
import numpy as np

class CustomGaussianNB:
    def fit(self, X, y):
        # X có shape (N, d), y có shape (N,)
        self.classes = np.unique(y)  # Lấy danh sách các lớp: [0, 1, 2]
        n_classes = len(self.classes)
        n_features = X.shape[1]
        
        # Khởi tạo mảng chứa trung bình, phương sai và prior
        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)
        
        # Tính toán tham số cho từng lớp
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]  # Lọc các dòng dữ liệu thuộc về lớp c
            self.means[idx, :] = np.mean(X_c, axis=0)       # Trung bình từng cột
            # Cộng thêm hằng số 1e-9 vào phương sai để chống chia cho 0
            self.variances[idx, :] = np.var(X_c, axis=0) + 1e-9
            self.priors[idx] = X_c.shape[0] / float(X.shape[0]) # Tỷ lệ mẫu lớp c

    def _calculate_log_likelihood(self, class_idx, x):
        # Tính log mật độ xác suất Gauss cho 1 mẫu x (d chiều)
        mean = self.means[class_idx]
        var = self.variances[class_idx]
        
        # Công thức log Gauss: -0.5 * ln(2*pi*var) - ((x - mean)^2) / (2 * var)
        log_pdf = -0.5 * np.log(2 * np.pi * var) - ((x - mean)**2) / (2 * var)
        # Tổng log của d thuộc tính độc lập
        return np.sum(log_pdf)

    def predict(self, X):
        y_pred = []
        for x in X:
            posteriors = []
            for idx, c in enumerate(self.classes):
                # Tính: log(Prior) + sum(log(Likelihood))
                log_prior = np.log(self.priors[idx])
                log_likelihood = self._calculate_log_likelihood(idx, x)
                posteriors.append(log_prior + log_likelihood)
            # Chọn lớp có posterior lớn nhất
            y_pred.append(self.classes[np.argmax(posteriors)])
        return np.array(y_pred)
```

---

### B. Sử dụng thư viện `scikit-learn` chuẩn xác
```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Với dữ liệu liên tục (Iris, Breast Cancer):
gnb = GaussianNB(var_smoothing=1e-9)
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)

# 2. Với dữ liệu tần suất từ văn bản (Spam email):
mnb = MultinomialNB(alpha=1.0)  # alpha=1.0 chính là Laplace smoothing
mnb.fit(X_train_counts, y_train)
y_pred_mnb = mnb.predict(X_test_counts)

# 3. Với dữ liệu nhị phân (Triệu chứng bệnh 0/1):
bnb = BernoulliNB(alpha=1.0)
bnb.fit(X_train_binary, y_train)
y_pred_bnb = bnb.predict(X_test_binary)

# Đánh giá kết quả
print("Accuracy:", accuracy_score(y_test, y_pred_gnb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_gnb))
```

---

## 7. CHI TIẾT CÁC BÀI TẬP THỰC HÀNH TUẦN 2 & 2B

### 1. Phân loại hoa Iris (`iris.csv`)
- **Dữ liệu**: 150 mẫu hoa chia đều 3 lớp (Setosa, Versicolor, Virginica), 4 thuộc tính chiều dài/chiều rộng đài hoa và cánh hoa.
- **Mô hình**: Gaussian Naïve Bayes.
- **Kết quả**: Độ chính xác rất cao ($> 95\%$). Lớp Setosa phân tách độc lập hoàn toàn, trong khi Versicolor và Virginica có một chút giao thoa nhẹ ở biên.

### 2. Phân loại ung thư vú Wisconsin (`breast-cancer-wisconsin.data`)
- **Dữ liệu**: 699 mẫu bệnh phẩm sinh thiết với 9 chỉ số tế bào học (độ dày khối, độ đồng nhất kích thước tế bào...). Nhãn: 2 (Lành tính) và 4 (Ác tính).
- **Mô hình**: Gaussian NB.
- **Ý nghĩa y tế**: Chú ý kiểm tra chỉ số Recall của lớp Ác tính để đảm bảo không bỏ sót ca ung thư nào.

### 3. Phân loại thư rác Spam Email (`ex6DataPrepared.zip` / `bai6_spam_ex6_nb.py`)
- **Dữ liệu**: Hàng ngàn email đã được trích xuất thành vector từ vựng (Word IDs & Frequencies).
- **Mô hình**: Multinomial Naïve Bayes.
- **Kết quả**: Chứng minh sức mạnh vượt trội của MNB trên bài toán lọc Spam — vừa huấn luyện cực nhanh (chỉ mất vài giây), vừa đạt độ chính xác trên $98\%$.

### 4. Phân loại triệu chứng bệnh nhân (`bernoulli_nb_symptoms.csv`)
- **Dữ liệu**: Bảng dữ liệu nhị phân 0/1 ghi nhận sự có mặt/vắng mặt của các triệu chứng bệnh.
- **Mô hình**: Bernoulli Naïve Bayes.

---

## 8. BỘ 10 CÂU HỎI VẤN ĐÁP BẢO VỆ BÀI TẬP TUẦN 2 KÈM ĐÁP ÁN CHUẨN

1. **Câu 1: Tại sao thuật toán này lại có tên là "Naïve" (Ngây thơ)?**  
   *Đáp*: Vì nó giả định các thuộc tính hoàn toàn độc lập với nhau khi biết nhãn lớp ($P(\mathbf{x} \mid y) = \prod P(x_j \mid y)$). Giả định này gần như không bao giờ đúng trong thực tế, nhưng nó giúp việc tính toán đơn giản đi hàng triệu lần.

2. **Câu 2: Naïve Bayes là Mô hình Sinh (Generative) hay Mô hình Phân biệt (Discriminative)? Khác nhau ở điểm nào?**  
   *Đáp*: Là Mô hình Sinh (Generative Model). Khác biệt ở chỗ: Mô hình Sinh học phân phối xác suất chung $P(\mathbf{x}, y) = P(\mathbf{x} \mid y) P(y)$ của từng lớp rồi dùng Định lý Bayes để suy ra nhãn; trong khi Mô hình Phân biệt (như Logistic Regression) học trực tiếp ranh giới quyết định $P(y \mid \mathbf{x})$.

3. **Câu 3: Làm thế nào để chọn giữa Gaussian NB, Multinomial NB và Bernoulli NB?**  
   *Đáp*: Dựa vào bản chất của đặc trưng đầu vào $X$:
   - Nếu $X$ là số thực liên tục $\implies$ Chọn **Gaussian NB**.
   - Nếu $X$ là số nguyên đếm tần suất lặp lại (Bag-of-Words) $\implies$ Chọn **Multinomial NB**.
   - Nếu $X$ là cờ nhị phân 0/1 (xuất hiện hay không xuất hiện) $\implies$ Chọn **Bernoulli NB**.

4. **Câu 4: Bẫy Tràn số dưới (Underflow) là gì? Khắc phục bằng cách nào?**  
   *Đáp*: Khi nhân hàng trăm xác suất nhỏ hơn 1 lại với nhau, kết quả sẽ tiến về $0$ vượt quá độ chính xác số thực của máy tính. Khắc phục bằng cách lấy logarit tự nhiên ($\ln$), biến phép nhân xác suất thành phép cộng các log-xác suất âm.

5. **Câu 5: Hiện tượng "Zero Probability" xảy ra khi nào và tại sao nó lại nguy hiểm?**  
   *Đáp*: Xảy ra khi một giá trị thuộc tính chưa từng xuất hiện trong tập train của một lớp nào đó, khiến $P(x_j \mid y) = 0$. Vì các xác suất được nhân với nhau, một số 0 này sẽ xóa sạch tích xác suất của toàn bộ lớp đó thành 0, bất chấp các thuộc tính khác có khớp đến đâu.

6. **Câu 6: Làm trơn Laplace (Laplace Smoothing) hoạt động như thế nào?**  
   *Đáp*: Bằng cách cộng thêm một lượng ảo $\alpha$ (thường là 1) vào tử số và $\alpha \times d$ vào mẫu số: $p_{cj} = \frac{N_{cj} + 1}{N_c + d}$. Điều này đảm bảo xác suất không bao giờ bị triệt tiêu về $0$.

7. **Câu 7: Điểm khác nhau giữa Multinomial NB và Bernoulli NB khi xử lý văn bản là gì?**  
   *Đáp*: Multinomial NB tính đến tần số xuất hiện của từ (từ xuất hiện 5 lần sẽ được nhân 5 lần) và bỏ qua các từ không có mặt. Bernoulli NB chỉ quan tâm từ có xuất hiện hay không (0 hoặc 1) và phạt cả sự vắng mặt của từ bằng cách nhân thêm $(1 - p_{cj})$.

8. **Câu 8: Trong Gaussian Naïve Bayes, tham số nào được học từ tập huấn luyện?**  
   *Đáp*: Với mỗi lớp $c$ và mỗi đặc trưng $j$, mô hình học 2 tham số: Giá trị trung bình mẫu $\mu_{cj}$ và Phương sai mẫu $\sigma_{cj}^2$. Ngoài ra còn học xác suất tiên nghiệm của từng lớp $P(y = c)$.

9. **Câu 9: Naïve Bayes có cần chuẩn hóa thang đo dữ liệu (Feature Scaling như Min-Max hay Z-Score) không?**  
   *Đáp*: Hoàn toàn **KHÔNG CẦN**. Bởi vì Naïve Bayes tính toán xác suất độc lập trên từng thuộc tính một, không có phép tính khoảng cách Euclid giữa các thuộc tính khác nhau như K-NN.

10. **Câu 10: Ưu điểm lớn nhất của Naïve Bayes so với các mô hình phức tạp (như Mạng nơ-ron hay SVM) là gì?**  
    *Đáp*: Tốc độ huấn luyện và dự đoán siêu nhanh (chỉ cần 1 lần quét qua dữ liệu để đếm tần suất hoặc tính trung bình), cần rất ít bộ nhớ, hoạt động tốt ngay cả với tập dữ liệu nhỏ hoặc số chiều cực lớn (như phân loại văn bản), và ít bị Overfitting.
