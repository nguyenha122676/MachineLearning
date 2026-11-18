# TÀI LIỆU LÝ THUYẾT ĐẦY ĐỦ VÀ CHUYÊN SÂU: HỌ THUẬT TOÁN NAÏVE BAYES
## Dành cho Học phần Machine Learning & Thi Vấn đáp Chuyên đề Phân loại

**Tác giả**: Nguyễn Hải Anh (MSSV: 24001968)  
**Tài liệu tham chiếu**: Giáo trình Machine Learning, tài liệu `thuchanh_bai1_NaiveBayes.pdf` và dẫn xuất toán học `HybridNaiveBayes_Derivation.pdf`.

---

## MỤC LỤC
1. [Nền tảng Xác suất & Thống kê suy diễn](#1-nền-tảng-xác-suất--thống-kê-suy-diễn)
   - 1.1. Không gian mẫu, Biến ngẫu nhiên và Xác suất có điều kiện
   - 1.2. Định lý Xác suất toàn phần và Định lý Bayes
   - 1.3. Hai trường phái tham số: MLE (Maximum Likelihood) vs MAP (Maximum A Posteriori)
2. [Bộ phân loại Naïve Bayes Tổng quát](#2-bộ-phân-loại-naïve-bayes-tổng-quát)
   - 2.1. Bộ phân loại Bayes Tối ưu (Optimal Bayes Classifier)
   - 2.2. Giả thiết độc lập có điều kiện ("Naïve Assumption")
   - 2.3. Quy tắc phân loại MAP trong Naïve Bayes
   - 2.4. Biến đổi Log-Likelihood chống tràn số dưới (Numerical Underflow)
   - 2.5. Bản chất Generative Model vs Discriminative Model (So sánh với Logistic Regression)
3. [Gaussian Naïve Bayes (Dữ liệu liên tục)](#3-gaussian-naïve-bayes-dữ-liệu-liên-tục)
   - 3.1. Giả thiết phân bố chuẩn 1 chiều
   - 3.2. Dẫn xuất toán học: Tại sao mật độ xác suất (PDF) thay thế được xác suất thực sự?
   - 3.3. Ước lượng tham số $\mu, \sigma^2$ bằng MLE
   - 3.4. Ma trận hiệp phương sai đường chéo và so sánh với QDA / LDA
   - 3.5. Kỹ thuật ổn định phương sai (Variance Smoothing $\epsilon$)
4. [Multinomial Naïve Bayes (Dữ liệu đếm tần suất / Văn bản)](#4-multinomial-naïve-bayes-dữ-liệu-đếm-tần-suất--văn-bản)
   - 4.1. Phân bố Đa thức (Multinomial Distribution)
   - 4.2. Biểu diễn văn bản (Bag of Words, Term Frequency)
   - 4.3. Bài toán xác suất bằng 0 ("Zero Probability Problem")
   - 4.4. Kỹ thuật làm mịn Laplace / Lidstone ($\alpha$-smoothing)
   - 4.5. Bản chất Bayes: Ước lượng MAP với phân bố tiên nghiệm Dirichlet (Dirichlet Conjugate Prior)
5. [Bernoulli Naïve Bayes (Dữ liệu nhị phân 0/1)](#5-bernoulli-naïve-bayes-dữ-liệu-nhị-phân-01)
   - 5.1. Phân bố Bernoulli
   - 5.2. Công thức Likelihood cho vector nhị phân
   - 5.3. Vai trò của đặc trưng vắng mặt $(1 - p_j)$ — Sự khác biệt cốt lõi với Multinomial NB
   - 5.4. Ước lượng tham số có làm mịn
6. [Categorical Naïve Bayes (Dữ liệu danh mục nhiều trạng thái)](#6-categorical-naïve-bayes-dữ-liệu-danh-mục-nhiều-trạng-thái)
   - 6.1. Phân bố Categorical
   - 6.2. So sánh One-Hot Encoding + Multinomial NB vs Ordinal Encoding + Categorical NB
7. [Hybrid Naïve Bayes (Mô hình Hỗn hợp đa biến)](#7-hybrid-naïve-bayes-mô-hình-hỗn-hợp-đa-biến)
   - 7.1. Động lực thực tế của mô hình lai
   - 7.2. Dẫn xuất toán học tổng quát từ tài liệu viện dẫn (`HybridNaiveBayes_Derivation.pdf`)
   - 7.3. Thuật toán phân loại trên tập đặc trưng hỗn hợp
8. [So sánh toàn diện & Hướng dẫn lựa chọn mô hình](#8-so-sánh-toàn-diện--hướng-dẫn-lựa-chọn-mô-hình)
   - 8.1. Bảng so sánh 4 biến thể Naïve Bayes
   - 8.2. So sánh Naïve Bayes với K-Nearest Neighbors và Logistic Regression
   - 8.3. Sơ đồ khối ra quyết định chọn mô hình (Decision Flowchart)
9. [Ưu điểm, Nhược điểm & Ứng dụng thực tế](#9-ưu-điểm-nhược-điểm--ứng-dụng-thực-tế)
10. [Tuyển tập 10 câu hỏi lý thuyết chuyên sâu thi Vấn đáp](#10-tuyển-tập-10-câu-hỏi-lý-thuyết-chuyên-sâu-thi-vấn-đáp)

---

## 1. Nền tảng Xác suất & Thống kê suy diễn

### 1.1. Không gian mẫu, Biến ngẫu nhiên và Xác suất có điều kiện
Xét không gian xác suất $(\Omega, \mathcal{F}, P)$, với:
- $\Omega$ là không gian mẫu (tập hợp tất cả các kết cục có thể xảy ra).
- $\mathcal{F}$ là $\sigma$-đại số các biến cố.
- $P: \mathcal{F} \rightarrow [0, 1]$ là hàm độ đo xác suất thỏa mãn 3 tiên đề Kolmogorov:
  1. $P(A) \ge 0, \quad \forall A \in \mathcal{F}$
  2. $P(\Omega) = 1$
  3. Tính cộng tính đếm được: Nếu $A_1, A_2, \dots$ đôi một xung khắc ($A_i \cap A_j = \emptyset, \forall i \neq j$), thì $P(\bigcup_{i=1}^\infty A_i) = \sum_{i=1}^\infty P(A_i)$.

**Định nghĩa Xác suất có điều kiện (Conditional Probability):**
Cho hai biến cố $A$ và $B$ với $P(B) > 0$. Xác suất xảy ra $A$ khi biết $B$ đã xảy ra được định nghĩa:
$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

Từ đó suy ra Quy tắc nhân xác suất (Chain Rule):
$$P(A \cap B) = P(A \mid B) P(B) = P(B \mid A) P(A)$$
Tổng quát cho $d$ biến ngẫu nhiên $X_1, X_2, \dots, X_d$:
$$P(X_1, X_2, \dots, X_d) = P(X_1) \prod_{j=2}^d P(X_j \mid X_1, \dots, X_{j-1})$$

---

### 1.2. Định lý Xác suất toàn phần và Định lý Bayes

**Định lý Xác suất toàn phần (Law of Total Probability):**
Nếu hệ biến cố $\{C_1, C_2, \dots, C_K\}$ lập thành một phân hoạch của $\Omega$ (tức là $C_i \cap C_j = \emptyset, \forall i \ne j$ và $\bigcup_{k=1}^K C_k = \Omega$), thì với mọi biến cố $A \in \mathcal{F}$:
$$P(A) = \sum_{k=1}^K P(A \cap C_k) = \sum_{k=1}^K P(A \mid C_k) P(C_k)$$

**Định lý Bayes (Bayes' Theorem):**
Từ định nghĩa xác suất có điều kiện và định lý xác suất toàn phần:
$$P(C_k \mid A) = \frac{P(A \mid C_k) P(C_k)}{P(A)} = \frac{P(A \mid C_k) P(C_k)}{\sum_{j=1}^K P(A \mid C_j) P(C_j)}$$

Trong bối cảnh học máy (Machine Learning):
- **$C_k$ (hay $y = c_k$)**: Lớp đối tượng cần phân loại (Class label).
- **$A$ (hay $\mathbf{x} = [x_1, \dots, x_d]$)**: Vector đặc trưng đầu vào quan sát được (Observed feature vector).
- **$P(y = c_k)$**: **Prior Probability (Xác suất tiên nghiệm)** — Niềm tin ban đầu về xác suất xảy ra của lớp $c_k$ trước khi quan sát dữ liệu $\mathbf{x}$.
- **$P(\mathbf{x} \mid y = c_k)$**: **Likelihood (Độ hợp lý)** — Xác suất quan sát thấy đặc trưng $\mathbf{x}$ nếu mẫu thuộc về lớp $c_k$.
- **$P(\mathbf{x}) = \sum_j P(\mathbf{x} \mid y = c_j) P(y = c_j)$**: **Evidence (Bằng chứng / Hằng số chuẩn hóa)** — Tổng xác suất biên duyên của dữ liệu $\mathbf{x}$ trên toàn bộ các lớp.
- **$P(y = c_k \mid \mathbf{x})$**: **Posterior Probability (Xác suất hậu nghiệm)** — Niềm tin sau khi đã quan sát dữ liệu $\mathbf{x}$, quyết định mẫu thuộc về lớp nào.

$$\text{Posterior} = \frac{\text{Likelihood} \times \text{Prior}}{\text{Evidence}}$$

---

### 1.3. Hai trường phái tham số: MLE vs MAP

Khi xây dựng mô hình xác suất phụ thuộc tham số $\theta$:
1. **MLE (Maximum Likelihood Estimation - Ước lượng hợp lý cực đại)**:
   Coi $\theta$ là một hằng số cố định chưa biết. Mục tiêu là tìm giá trị $\theta$ làm cho dữ liệu huấn luyện quan sát được $\mathcal{D} = \{\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(N)}\}$ có xác suất xảy ra cao nhất:
   $$\hat{\theta}_{\text{MLE}} = \arg\max_\theta P(\mathcal{D} \mid \theta) = \arg\max_\theta \sum_{i=1}^N \ln P(\mathbf{x}^{(i)} \mid \theta)$$
2. **MAP (Maximum A Posteriori - Ước lượng tiên nghiệm cực đại)**:
   Coi $\theta$ là một biến ngẫu nhiên tuân theo phân bố tiên nghiệm $P(\theta)$. Mục tiêu là tối đa hóa xác suất hậu nghiệm của tham số:
   $$\hat{\theta}_{\text{MAP}} = \arg\max_\theta P(\theta \mid \mathcal{D}) = \arg\max_\theta \left[ \ln P(\mathcal{D} \mid \theta) + \ln P(\theta) \right]$$
   - Khi chọn phân bố tiên nghiệm đều (Uniform prior: $P(\theta) = \text{const}$), thì $\hat{\theta}_{\text{MAP}} \equiv \hat{\theta}_{\text{MLE}}$.
   - **Kỹ thuật làm mịn Laplace chính là một dạng ước lượng MAP với phân bố tiên nghiệm Dirichlet!**

---

## 2. Bộ phân loại Naïve Bayes Tổng quát

### 2.1. Bộ phân loại Bayes Tối ưu (Optimal Bayes Classifier)
Quy tắc quyết định Bayes tối ưu là quy tắc giảm thiểu sai số phân loại kỳ vọng (Bayes Risk). Khi ma trận tổn thất là hàm 0-1 (0-1 Loss: phạt 1 nếu đoán sai, 0 nếu đoán đúng), quyết định tối ưu là chọn lớp có xác suất hậu nghiệm lớn nhất (nguyên lý MAP):
$$\hat{y} = \arg\max_{c_k \in \mathcal{C}} P(y = c_k \mid \mathbf{x}) = \arg\max_{c_k} \frac{P(\mathbf{x} \mid y = c_k) P(y = c_k)}{P(\mathbf{x})}$$
Vì mẫu số $P(\mathbf{x}) > 0$ giống nhau cho mọi lớp $c_k$, bài toán tối ưu tương đương với:
$$\hat{y} = \arg\max_{c_k} P(y = c_k) P(\mathbf{x} \mid y = c_k)$$

**Khó khăn thực tế ("Lời nguyền số chiều" - Curse of Dimensionality):**
Để tính $P(\mathbf{x} \mid y = c_k) = P(x_1, x_2, \dots, x_d \mid y = c_k)$:
- Nếu $d$ thuộc tính nhị phân $\{0, 1\}$, có $2^d - 1$ tham số tự do cần ước lượng cho mỗi lớp. Với $d = 50$, $2^{50} \approx 1.12 \times 10^{15}$ tham số — không có tập dữ liệu nào trên thế giới đủ lớn để ước lượng phân bố này!

---

### 2.2. Giả thiết độc lập có điều kiện ("Naïve Assumption")
Để giải quyết bài toán trên, thuật toán đưa ra **giả thiết độc lập có điều kiện**:
> *Khi đã biết trước nhãn phân lớp $y = c_k$, các đặc trưng $x_1, x_2, \dots, x_d$ hoàn toàn độc lập với nhau.*

Toán học hóa giả định này:
$$P(x_1, x_2, \dots, x_d \mid y = c_k) = \prod_{j=1}^d P(x_j \mid y = c_k)$$

- Nhờ giả định này, số lượng tham số cần ước lượng giảm từ hàm mũ $\mathcal{O}(K \cdot 2^d)$ xuống hàm tuyến tính $\mathcal{O}(K \cdot d)$!
- Đây là lý do thuật toán có tên là **"Naïve" (Ngây thơ)**: Trong thực tế, các đặc trưng hiếm khi độc lập hoàn toàn (ví dụ: sốt thường đi kèm nhức đầu, chiều dài cánh hoa tương quan với chiều rộng cánh hoa). Tuy nhiên, Naïve Bayes vẫn hoạt động xuất sắc bất ngờ trong thực nghiệm.

---

### 2.3. Quy tắc phân loại MAP trong Naïve Bayes
Kết hợp giả định độc lập vào công thức Bayes, hàm mục tiêu phân loại là:
$$\hat{y} = \arg\max_{c_k \in \{c_1, \dots, c_K\}} \left[ P(y = c_k) \prod_{j=1}^d P(x_j \mid y = c_k) \right]$$

Xác suất tiên nghiệm $P(y = c_k)$ được ước lượng trực tiếp bằng tần suất xuất hiện của lớp $c_k$ trong tập dữ liệu huấn luyện:
$$P(y = c_k) = \frac{N_k}{N}$$
với $N_k = \sum_{i=1}^N \mathbb{I}(y^{(i)} = c_k)$ là số lượng mẫu thuộc lớp $c_k$, và $N$ là tổng số mẫu.

---

### 2.4. Biến đổi Log-Likelihood chống tràn số dưới (Numerical Underflow)

Trong không gian nhiều chiều ($d$ lớn, ví dụ phân loại văn bản với $d = 2500$ từ vựng):
- Mỗi xác suất thành phần $P(x_j \mid y = c_k) \in (0, 1)$ là một số thực rất nhỏ (ví dụ $10^{-3} \sim 10^{-4}$).
- Tích của hàng nghìn số nhỏ sẽ dẫn đến:
  $$\prod_{j=1}^d P(x_j \mid y = c_k) \approx 0 \quad (\text{dưới ngưỡng biểu diễn } \text{float64} \approx 2.2 \times 10^{-308})$$
  Máy tính sẽ làm tròn về `0.0`, khiến mọi lớp đều có điểm số 0 và không thể so sánh `argmax`!

**Giải pháp**: Lấy logarit tự nhiên ($\ln$) hai vế:
Vì hàm $\ln(u)$ là hàm đơn điệu tăng ngặt trên $(0, +\infty)$, nên $\arg\max_u f(u) \equiv \arg\max_u \ln f(u)$:
$$\hat{y} = \arg\max_{c_k} \left[ \ln P(y = c_k) + \sum_{j=1}^d \ln P(x_j \mid y = c_k) \right]$$
- Biến đổi phép nhân thành phép cộng các số âm lớn.
- Đảm bảo độ chính xác số học tuyệt đối và tăng tốc độ tính toán CPU.

---

### 2.5. Generative Model vs Discriminative Model

Naïve Bayes thuộc nhóm **Generative Model (Mô hình sinh)**:
- **Generative (Naïve Bayes, HMM, GMM)**: Mô hình hóa phân bố đồng thời $P(\mathbf{x}, y) = P(y) P(\mathbf{x} \mid y)$. Mô hình cố gắng tìm hiểu cách dữ liệu của từng lớp được sinh ra như thế nào, sau đó dùng định lý Bayes để suy diễn $P(y \mid \mathbf{x})$.
- **Discriminative (Logistic Regression, SVM, MLP)**: Mô hình hóa trực tiếp ranh giới quyết định hoặc xác suất hậu nghiệm $P(y \mid \mathbf{x})$ mà không quan tâm dữ liệu $\mathbf{x}$ được sinh ra như thế nào.

**So sánh kinh điển giữa Naïve Bayes và Logistic Regression (Andrew Ng & Michael Jordan, NIPS 2001):**
1. Cả hai mô hình đều tạo ra ranh giới quyết định tuyến tính (Linear Decision Boundary) trong trường hợp dữ liệu rời rạc.
2. Naïve Bayes đạt đến cận sai số tiệm cận (asymptotic error) nhanh hơn rất nhiều: chỉ cần lượng dữ liệu $\mathcal{O}(\log d)$, trong khi Logistic Regression cần $\mathcal{O}(d)$ mẫu.
3. Khi dữ liệu ít, Naïve Bayes thường vượt trội hơn Logistic Regression vì ít bị overfit (variance thấp). Khi dữ liệu cực lớn, Logistic Regression sẽ đạt độ chính xác cao hơn vì không bị giới hạn bởi giả thiết độc lập.

---

## 3. Gaussian Naïve Bayes (Dữ liệu liên tục)

### 3.1. Giả thiết phân bố chuẩn 1 chiều
Khi đặc trưng $x_j \in \mathbb{R}$ là số thực liên tục, ta giả định với mỗi lớp $y = c_k$, biến ngẫu nhiên $X_j$ tuân theo phân bố chuẩn Gauss:
$$X_j \mid (y = c_k) \sim \mathcal{N}(\mu_{k, j}, \sigma_{k, j}^2)$$
Hàm mật độ xác suất (Probability Density Function - PDF):
$$f(x_j \mid \mu_{k, j}, \sigma_{k, j}^2) = \frac{1}{\sqrt{2\pi \sigma_{k, j}^2}} \exp\left( -\frac{(x_j - \mu_{k, j})^2}{2\sigma_{k, j}^2} \right)$$

---

### 3.2. Dẫn xuất toán học: Tại sao PDF thay thế được xác suất thực sự?
*(Trích từ tài liệu chuyên sâu `HybridNaiveBayes_Derivation.pdf`)*

Nhiều người học thường băn khoăn: *Giá trị mật độ xác suất $f(x)$ có thể lớn hơn 1, trong khi xác suất $P$ bắt buộc phải nằm trong $[0, 1]$. Vậy tại sao có thể thay trực tiếp $f(x_j)$ vào công thức Bayes vốn áp dụng cho xác suất?*

**Chứng minh toán học:**
Với một biến ngẫu nhiên liên tục, xác suất tại đúng một điểm duy nhất là bằng 0: $P(X = z) = 0$.
Tuy nhiên, ta xét biến cố quan sát được giá trị trong một lân cận vô cùng bé kích thước $\epsilon > 0$ quanh điểm $z_j$:
$$P\left(x_j \in \left[z_j - \frac{\epsilon}{2}, z_j + \frac{\epsilon}{2}\right] \,\middle|\, y = c_k\right) = \int_{z_j - \epsilon/2}^{z_j + \epsilon/2} \mathcal{N}(x; \mu_{k, j}, \sigma_{k, j}^2) \, dx$$
Theo Định lý giá trị trung bình tích phân (Mean Value Theorem for Integrals), khi $\epsilon \rightarrow 0$:
$$P(x_j \approx z_j \mid y = c_k) = \epsilon \cdot \mathcal{N}(z_j; \mu_{k, j}, \sigma_{k, j}^2)$$

Thay biểu thức này vào công thức Bayes cho toàn bộ $d$ đặc trưng:
$$P(y = c_k \mid \mathbf{z}) = \frac{P(y = c_k) \prod_{j=1}^d \left[ \epsilon \cdot \mathcal{N}(z_j; \mu_{k, j}, \sigma_{k, j}^2) \right]}{\sum_{\ell=1}^K P(y = c_\ell) \prod_{j=1}^d \left[ \epsilon \cdot \mathcal{N}(z_j; \mu_{\ell, j}, \sigma_{\ell, j}^2) \right]}$$
Đưa thừa số chung $\epsilon^d$ ra ngoài cả tử số và mẫu số:
$$P(y = c_k \mid \mathbf{z}) = \frac{\epsilon^d \cdot \left[ P(y = c_k) \prod_{j=1}^d \mathcal{N}(z_j; \mu_{k, j}, \sigma_{k, j}^2) \right]}{\epsilon^d \cdot \left[ \sum_{\ell=1}^K P(y = c_\ell) \prod_{j=1}^d \mathcal{N}(z_j; \mu_{\ell, j}, \sigma_{\ell, j}^2) \right]}$$
**Thừa số $\epsilon^d$ bị triệt tiêu hoàn toàn!**
Kết quả:
$$P(y = c_k \mid \mathbf{z}) = \frac{P(y = c_k) \prod_{j=1}^d \mathcal{N}(z_j; \mu_{k, j}, \sigma_{k, j}^2)}{\sum_{\ell=1}^K P(y = c_\ell) \prod_{j=1}^d \mathcal{N}(z_j; \mu_{\ell, j}, \sigma_{\ell, j}^2)}$$
$\Rightarrow$ Việc sử dụng hàm mật độ Gauss thay thế cho xác suất là **hoàn toàn chính xác và chặt chẽ về mặt toán học**.

---

### 3.3. Ước lượng tham số $\mu, \sigma^2$ bằng MLE
Hàm log-likelihood cho thuộc tính $j$ của lớp $c_k$:
$$\ell(\mu_{k, j}, \sigma_{k, j}^2) = -\frac{N_k}{2} \ln(2\pi) - \frac{N_k}{2} \ln(\sigma_{k, j}^2) - \frac{1}{2\sigma_{k, j}^2} \sum_{i: y^{(i)} = c_k} (x_{i, j} - \mu_{k, j})^2$$
Lấy đạo hàm riêng theo $\mu_{k, j}$ và cho bằng 0:
$$\frac{\partial \ell}{\partial \mu_{k, j}} = \frac{1}{\sigma_{k, j}^2} \sum_{i: y^{(i)} = c_k} (x_{i, j} - \mu_{k, j}) = 0 \implies \hat{\mu}_{k, j} = \frac{1}{N_k} \sum_{i: y^{(i)} = c_k} x_{i, j}$$
Lấy đạo hàm riêng theo $\sigma_{k, j}^2$ và cho bằng 0:
$$\frac{\partial \ell}{\partial \sigma_{k, j}^2} = -\frac{N_k}{2\sigma_{k, j}^2} + \frac{1}{2(\sigma_{k, j}^2)^2} \sum_{i: y^{(i)} = c_k} (x_{i, j} - \hat{\mu}_{k, j})^2 = 0 \implies \hat{\sigma}_{k, j}^2 = \frac{1}{N_k} \sum_{i: y^{(i)} = c_k} (x_{i, j} - \hat{\mu}_{k, j})^2$$

---

### 3.4. Ma trận hiệp phương sai đường chéo và so sánh với QDA / LDA
Trong phân tích phân biệt đa biến (Multivariate Discriminant Analysis):
- **QDA (Quadratic Discriminant Analysis)**: Giả định mỗi lớp có một ma trận hiệp phương sai đầy đủ $\mathbf{\Sigma}_k \in \mathbb{R}^{d \times d}$. Ranh giới quyết định là mặt bậc hai (quadratic). Số tham số hiệp phương sai cần ước lượng: $K \cdot \frac{d(d+1)}{2}$.
- **LDA (Linear Discriminant Analysis)**: Giả định tất cả các lớp dùng chung một ma trận hiệp phương sai $\mathbf{\Sigma}_1 = \dots = \mathbf{\Sigma}_K = \mathbf{\Sigma}$. Ranh giới quyết định là siêu phẳng tuyến tính.
- **Gaussian Naïve Bayes**: Chính là **QDA nhưng bị ràng buộc ma trận hiệp phương sai phải có dạng đường chéo (Diagonal Covariance Matrix)**:
  $$\mathbf{\Sigma}_k = \operatorname{diag}(\sigma_{k, 1}^2, \sigma_{k, 2}^2, \dots, \sigma_{k, d}^2)$$
  Các phần tử ngoài đường chéo $\operatorname{Cov}(X_a, X_b) = 0$ thể hiện tính độc lập giữa các thuộc tính. Số tham số chỉ còn $K \cdot d$!

---

### 3.5. Kỹ thuật ổn định phương sai (Variance Smoothing)
Nếu trong tập huấn luyện, tất cả các mẫu của một lớp có cùng một giá trị đặc trưng (ví dụ: mẫu thực nghiệm cảm biến đứng yên), phương sai mẫu sẽ bằng 0: $\sigma_{k, j}^2 = 0$.
Khi đó, hàm mật độ $\frac{1}{\sqrt{2\pi \cdot 0}}$ sẽ gây lỗi chia cho 0 (`ZeroDivisionError`).
**Giải pháp trong Scikit-Learn (`var_smoothing`)**:
Cộng thêm một lượng nhỏ vào phương sai:
$$\sigma_{k, j}^2 \leftarrow \sigma_{k, j}^2 + \epsilon \cdot \max_{k, j}(\sigma_{k, j}^2)$$
với mặc định $\epsilon = 10^{-9}$ (`var_smoothing=1e-9`).

---

## 4. Multinomial Naïve Bayes (Dữ liệu đếm / Văn bản)

### 4.1. Phân bố Đa thức (Multinomial Distribution)
Mở rộng của phân bố Nhị thức cho bài toán có $d$ kết cục rời rạc.
Giả sử một văn bản có tổng độ dài $n = \sum_{j=1}^d x_j$ từ, trong đó từ thứ $j$ trong từ điển vựng xuất hiện $x_j$ lần. Xác suất sinh ra văn bản đó khi biết thuộc lớp $c_k$:
$$P(\mathbf{x} \mid y = c_k) = \frac{(\sum_{j=1}^d x_j)!}{\prod_{j=1}^d x_j!} \prod_{j=1}^d p_{k, j}^{x_j}$$
Trong đó:
- $p_{k, j} = P(w_j \mid y = c_k)$ là xác suất từ $w_j$ được chọn khi sinh văn bản thuộc lớp $c_k$, thỏa mãn $\sum_{j=1}^d p_{k, j} = 1$.
- Đại lượng $\frac{(\sum x_j)!}{\prod x_j!}$ là hệ số tổ hợp đa thức, độc lập với lớp $c_k$ khi xét trên cùng một văn bản kiểm thử $\mathbf{x}$, nên bị triệt tiêu khi so sánh $\arg\max$.

Do đó:
$$P(\mathbf{x} \mid y = c_k) \propto \prod_{j=1}^d p_{k, j}^{x_j}$$
Chuyển sang logarit:
$$\ln P(\mathbf{x} \mid y = c_k) = \sum_{j=1}^d x_j \ln p_{k, j}$$

---

### 4.2. Vấn đề xác suất bằng 0 ("Zero Probability Problem")
Nếu ước lượng xác suất $p_{k, j}$ bằng tần suất xuất hiện thông thường (MLE):
$$p_{k, j}^{\text{MLE}} = \frac{N_{k, j}}{N_k}$$
với $N_{k, j}$ là số lần từ $w_j$ xuất hiện trong lớp $c_k$, và $N_k = \sum_{j=1}^d N_{k, j}$ là tổng số từ của lớp $c_k$.

**Nguy cơ**: Nếu văn bản kiểm thử có một từ chưa từng xuất hiện trong lớp $c_k$ ở tập Train ($N_{k, j} = 0$), thì:
$$p_{k, j} = 0 \implies \prod_{j=1}^d p_{k, j}^{x_j} = 0 \quad (\text{hoặc } \ln p_{k, j} = -\infty)$$
Toàn bộ tích xác suất của lớp $c_k$ bị triệt tiêu về 0, mặc dù hàng trăm từ còn lại trong văn bản đều mang đặc trưng rất mạnh của lớp $c_k$!

---

### 4.3. Kỹ thuật làm mịn Laplace / Lidstone ($\alpha$-smoothing)
Để khắc phục, nhà toán học Pierre-Simon Laplace đã đề xuất nguyên tắc: *Giả sử mỗi từ trong từ điển vựng đều đã xuất hiện thêm $\alpha$ lần trước khi quan sát dữ liệu*:
$$\hat{p}_{k, j} = \frac{N_{k, j} + \alpha}{N_k + \alpha \cdot |V|}$$
- $\alpha > 0$: Hệ số làm mịn (Smoothing parameter).
  - Khi $\alpha = 1$: Gọi là **Laplace Smoothing** (hoặc Add-1 smoothing).
  - Khi $0 < \alpha < 1$: Gọi là **Lidstone Smoothing**.
  - Khi $\alpha = 0$: Trở về ước lượng MLE không làm mịn.
- $|V|$: Kích thước từ điển vựng (tổng số đặc trưng rời rạc).
- **Tính chuẩn hóa được bảo toàn**:
  $$\sum_{j=1}^{|V|} \hat{p}_{k, j} = \frac{\sum_{j=1}^{|V|} (N_{k, j} + \alpha)}{N_k + \alpha \cdot |V|} = \frac{N_k + \alpha \cdot |V|}{N_k + \alpha \cdot |V|} = 1$$

---

### 4.4. Bản chất Bayes: Ước lượng MAP với Tiên nghiệm Dirichlet
Dưới góc nhìn suy diễn Bayes chặt chẽ:
- Tham số $\mathbf{p}_k = (p_{k, 1}, \dots, p_{k, |V|})$ tuân theo **phân bố tiên nghiệm liên hợp Dirichlet (Dirichlet Prior Distribution)**:
  $$\operatorname{Dir}(\mathbf{p}_k \mid \boldsymbol{\alpha}) \propto \prod_{j=1}^{|V|} p_{k, j}^{\alpha_j - 1}$$
- Khi gán tiên nghiệm đều (Symmetric prior: $\alpha_1 = \dots = \alpha_{|V|} = \alpha + 1$):
  $$\text{Posterior} \propto \text{Likelihood} \times \text{Prior} \propto \left(\prod_{j=1}^{|V|} p_{k, j}^{N_{k, j}}\right) \left(\prod_{j=1}^{|V|} p_{k, j}^\alpha\right) = \prod_{j=1}^{|V|} p_{k, j}^{N_{k, j} + \alpha}$$
- Cực đại hóa xác suất hậu nghiệm (MAP) của phân bố Dirichlet này cho ra đúng nghiệm:
  $$\hat{p}_{k, j}^{\text{MAP}} = \frac{N_{k, j} + \alpha}{N_k + \alpha \cdot |V|}$$
$\Rightarrow$ Làm mịn Laplace không phải là một mẹo chắp vá (heuristic) ngẫu nhiên, mà là **kết quả toán học của ước lượng MAP với phân bố tiên nghiệm Dirichlet!**

---

## 5. Bernoulli Naïve Bayes (Dữ liệu nhị phân 0/1)

### 5.1. Phân bố Bernoulli
Dành cho vector đặc trưng nhị phân $\mathbf{x} = (x_1, \dots, x_d) \in \{0, 1\}^d$, trong đó:
$$x_j = \begin{cases} 1 & \text{nếu đặc trưng } j \text{ xuất hiện (True / Có)} \\ 0 & \text{nếu đặc trưng } j \text{ vắng mặt (False / Không)} \end{cases}$$

Đặt $p_{k, j} = P(x_j = 1 \mid y = c_k)$ là xác suất đặc trưng $j$ xuất hiện ở lớp $c_k$.
Hàm khối xác suất của một biến Bernoulli:
$$P(x_j \mid y = c_k) = p_{k, j}^{x_j} (1 - p_{k, j})^{1 - x_j}$$
Theo giả thiết độc lập có điều kiện:
$$P(\mathbf{x} \mid y = c_k) = \prod_{j=1}^d \left[ p_{k, j}^{x_j} (1 - p_{k, j})^{1 - x_j} \right]$$
Dạng Log-Likelihood:
$$\ln P(\mathbf{x} \mid y = c_k) = \sum_{j=1}^d \left[ x_j \ln p_{k, j} + (1 - x_j) \ln(1 - p_{k, j}) \right]$$

---

### 5.2. Sự khác biệt cốt lõi giữa Bernoulli NB và Multinomial NB

| Tiêu chí | Multinomial Naïve Bayes | Bernoulli Naïve Bayes |
| :--- | :--- | :--- |
| **Không gian vector $\mathbf{x}$** | Đếm tần suất: $\mathbf{x} \in \mathbb{N}_0^d$ | Chỉ báo nhị phân: $\mathbf{x} \in \{0, 1\}^d$ |
| **Số lần lặp lại của từ** | Tính toán trực tiếp ($x_j$ làm số mũ: $p_j^{x_j}$) | Bị triệt tiêu về 1 (từ xuất hiện 1 lần hay 100 lần đều là $x_j = 1$) |
| **Thuộc tính vắng mặt ($x_j = 0$)** | **Bị bỏ qua**: $p_j^0 = 1$, không ảnh hưởng tích | **Tham gia tính toán**: Thừa số $(1 - p_j)$ phạt trực tiếp lớp có từ đó! |
| **Độ dài văn bản** | Khả năng phân loại tốt trên văn bản dài | Nhạy cảm với độ dài (văn bản dài dễ bị phạt nặng) |

---

### 5.3. Ước lượng tham số có làm mịn Laplace
Gọi:
- $D_k$: Tổng số mẫu (văn bản / bệnh nhân) thuộc lớp $c_k$.
- $D_{k, j}$: Số mẫu trong lớp $c_k$ có đặc trưng $j$ xuất hiện ($x_j = 1$).
Ước lượng có làm mịn Laplace ($\alpha$):
$$\hat{p}_{k, j} = \frac{D_{k, j} + \alpha}{D_k + 2\alpha}$$
*(Chú ý mẫu số là $D_k + 2\alpha$ vì biến Bernoulli chỉ có đúng 2 trạng thái: $\{0, 1\}$).*

---

## 6. Categorical Naïve Bayes (Dữ liệu danh mục)

### 6.1. Phân bố Categorical
Khi thuộc tính $x_j$ là biến danh mục có $C_j$ trạng thái rời rạc (ví dụ: ngày trong tuần $C_j = 7$, thành phố $C_j = 20$):
$$P(x_j = c \mid y = c_k) = \theta_{k, j, c} = \frac{N_{k, j, c} + \alpha}{N_k + \alpha \cdot C_j}$$
- $N_{k, j, c}$: Số lần thuộc tính $j$ nhận giá trị $c$ trong lớp $c_k$.
- $C_j$: Số lượng giá trị khác nhau mà thuộc tính $j$ có thể nhận.

### 6.2. Mối quan hệ tương đương giữa One-Hot Multinomial NB và Categorical NB
Trong Bài tập 4 thực hành, khi áp dụng One-Hot Encoding lên 8 biến danh mục, ta thu được 74 cột nhị phân.
- Khi đưa vào `MultinomialNB`, mỗi mẫu có tổng số từ $n = \sum x_j = 8$ (vì mỗi biến danh mục có đúng một vị trí bằng 1).
- Khi đưa vào `CategoricalNB` với Ordinal Encoding, mô hình tính xác suất độc lập trên từng trường danh mục.
$\Rightarrow$ **Kết quả thực nghiệm cho thấy Accuracy của hai phương pháp hoàn toàn trùng khớp (50.33%)** vì bản chất toán học của xác suất hợp lý có điều kiện trên phân hoạch rời rạc là hoàn toàn tương đương!

---

## 7. Hybrid Naïve Bayes (Mô hình Hỗn hợp đa biến)
*(Dẫn xuất trực tiếp từ tài liệu đính kèm `HybridNaiveBayes_Derivation.pdf`)*

### 7.1. Động lực thực tế
Trong thực tế (ví dụ: dữ liệu y tế, thương mại điện tử, tín dụng ngân hàng), một mẫu dữ liệu thường chứa đồng thời:
1. **Các biến liên tục** ($\mathbf{x}_{\text{cont}}$): Tuổi, huyết áp, thu nhập, số dư tài khoản.
2. **Các biến nhị phân** ($\mathbf{x}_{\text{bin}}$): Có hút thuốc không (0/1), có tiền sử bệnh tim không (0/1).
3. **Các biến đếm** ($\mathbf{x}_{\text{count}}$): Số lần giao dịch trong tháng, số lần nhập viện.
4. **Các biến danh mục** ($\mathbf{x}_{\text{cat}}$): Nghề nghiệp, loại thẻ tín dụng, nhóm máu.

Hầu hết các thư viện chuẩn (như Scikit-Learn) tách rời các lớp `GaussianNB`, `MultinomialNB`, `BernoulliNB`. Mô hình **Hybrid Naïve Bayes** giải quyết trọn vẹn bài toán này bằng cách hợp nhất toán học.

---

### 7.2. Dẫn xuất toán học tổng quát
Giả sử vector đặc trưng quan sát $\mathbf{z} \in \mathbb{R}^d$ được phân hoạch thành các nhóm thuộc tính:
- $\mathcal{J}_{\text{cont}}$: Tập chỉ số các biến liên tục tuân theo hàm mật độ xác suất $f_{k, j}(z_j)$ (ví dụ: Gauss).
- $\mathcal{J}_{\text{bin}}$: Tập chỉ số các biến nhị phân tuân theo khối xác suất Bernoulli $p_{k, j}$.
- $\mathcal{J}_{\text{disc}}$: Tập chỉ số các biến rời rạc/đếm tuân theo khối xác suất Đa thức hoặc Poisson $P_{k, j}(z_j)$.

Theo giả thiết Naïve Bayes độc lập có điều kiện, độ hợp lý đồng thời là tích của tất cả các thành phần:
$$P(\mathbf{z} \mid y = c_k) = \left( \prod_{j \in \mathcal{J}_{\text{cont}}} \epsilon \cdot f_{k, j}(z_j) \right) \left( \prod_{j \in \mathcal{J}_{\text{bin}}} p_{k, j}^{z_j} (1 - p_{k, j})^{1 - z_j} \right) \left( \prod_{j \in \mathcal{J}_{\text{disc}}} P_{k, j}(z_j) \right)$$

Đưa vào công thức Bayes và triệt tiêu $\epsilon^{|\mathcal{J}_{\text{cont}}|}$ ở cả tử số và mẫu số (như đã chứng minh ở mục 3.2), ta có xác suất hậu nghiệm chính xác:
$$P(y = c_k \mid \mathbf{z}) = \frac{P(y = c_k) \cdot \prod_{j \in \mathcal{J}_{\text{cont}}} f_{k, j}(z_j) \cdot \prod_{j \in \mathcal{J}_{\text{bin}}} P_{\text{bin}}(z_j \mid c_k) \cdot \prod_{j \in \mathcal{J}_{\text{disc}}} P_{\text{disc}}(z_j \mid c_k)}{\sum_{\ell=1}^K \left[ P(y = c_\ell) \cdot \prod_{j \in \mathcal{J}_{\text{cont}}} f_{\ell, j}(z_j) \cdot \prod_{j \in \mathcal{J}_{\text{bin}}} P_{\text{bin}}(z_j \mid c_\ell) \cdot \prod_{j \in \mathcal{J}_{\text{disc}}} P_{\text{disc}}(z_j \mid c_\ell) \right]}$$

Chuyển sang hàm Log-Likelihood:
$$\hat{y} = \arg\max_{c_k} \left[ \ln P(y = c_k) + \sum_{j \in \mathcal{J}_{\text{cont}}} \ln f_{k, j}(z_j) + \sum_{j \in \mathcal{J}_{\text{bin}}} \ln P_{\text{bin}}(z_j \mid c_k) + \sum_{j \in \mathcal{J}_{\text{disc}}} \ln P_{\text{disc}}(z_j \mid c_k) \right]$$

---

## 8. So sánh toàn diện & Hướng dẫn lựa chọn mô hình

### 8.1. Bảng so sánh 4 biến thể Naïve Bayes

| Đặc trưng so sánh | Gaussian NB | Multinomial NB | Bernoulli NB | Categorical NB |
| :--- | :--- | :--- | :--- | :--- |
| **Miền giá trị đầu vào** | $x_j \in \mathbb{R}$ (Số thực) | $x_j \in \{0, 1, 2, \dots\}$ (Số đếm) | $x_j \in \{0, 1\}$ (Nhị phân) | $x_j \in \{0, 1, \dots, C_j - 1\}$ |
| **Phân bố thành phần** | Chuẩn $\mathcal{N}(\mu, \sigma^2)$ | Đa thức (Multinomial) | Bernoulli | Categorical / Multinoulli |
| **Tham số mỗi lớp** | $2d$ tham số ($\mu_j, \sigma_j^2$) | $d$ tham số ($p_j$) | $d$ tham số ($p_j$) | $\sum_{j=1}^d C_j$ tham số |
| **Cơ chế làm mịn** | Cộng $\epsilon$ vào phương sai | Laplace $\frac{N_{kj} + \alpha}{N_k + \alpha \|V\|}$ | Laplace $\frac{D_{kj} + \alpha}{D_k + 2\alpha}$ | Laplace $\frac{N_{kjc} + \alpha}{N_k + \alpha C_j}$ |
| **Bài toán điển hình** | Hoa Iris, Ung thư vú Wisconsin | Phân loại văn bản dài, Đơn hàng | Triệu chứng bệnh học, Nhận diện chữ số đen trắng | Dự đoán nghề nghiệp, Phân khúc khách hàng |

---

### 8.2. So sánh Naïve Bayes với K-NN và Logistic Regression

| Tiêu chí | Naïve Bayes | K-Nearest Neighbors (K-NN) | Logistic Regression |
| :--- | :--- | :--- | :--- |
| **Loại mô hình** | Parametric, Generative | Non-parametric, Lazy learning | Parametric, Discriminative |
| **Độ phức tạp Training** | $\mathcal{O}(N \cdot d)$ (Rất nhanh, 1 lần quét) | $\mathcal{O}(1)$ (Không học, chỉ lưu) | $\mathcal{O}(N \cdot d \cdot \text{iterations})$ (Gradient Descent) |
| **Độ phức tạp Inference** | $\mathcal{O}(K \cdot d)$ (Cực kỳ nhanh) | $\mathcal{O}(N \cdot d)$ (Rất chậm khi $N$ lớn) | $\mathcal{O}(K \cdot d)$ (Nhanh) |
| **Khả năng diễn giải** | Cao (qua xác suất điều kiện) | Trung bình (qua láng giềng) | Cao (qua trọng số $\mathbf{w}$) |
| **Bộ nhớ sử dụng** | Nhỏ (chỉ lưu tham số) | Lớn (phải lưu toàn bộ dữ liệu) | Nhỏ (chỉ lưu vector trọng số) |
| **Xử lý số chiều cao** | Rất tốt (kháng overfit tốt) | Rất tệ (bị Curse of Dimensionality) | Tốt (cần Regularization L1/L2) |

---

### 8.3. Sơ đồ khối lựa chọn mô hình Naïve Bayes

```
Dữ liệu đầu vào của bài toán là gì?
│
├── Các biến là số thực liên tục (đo lường, chỉ số sinh học, cảm biến)?
│   └── Chọn: GAUSSIAN NAÏVE BAYES (GaussianNB)
│
├── Dữ liệu là văn bản hoặc tần suất đếm số lần xuất hiện?
│   └── Chọn: MULTINOMIAL NAÏVE BAYES (MultinomialNB)
│
├── Dữ liệu nhị phân 0/1 (có/không, đúng/sai, có triệu chứng hay không)?
│   └── Chọn: BERNOULLI NAÏVE BAYES (BernoulliNB)
│
├── Dữ liệu gồm nhiều biến danh mục nhiều trạng thái (city, device, segment)?
│   └── Chọn: CATEGORICAL NAÏVE BAYES hoặc MULTINOMIAL NAÏVE BAYES (sau One-Hot)
│
└── Dữ liệu hỗn hợp cả số thực, nhị phân và danh mục?
    └── Chọn: HYBRID NAÏVE BAYES (Tổng hợp Log-Likelihood)
```

---

## 9. Ưu điểm, Nhược điểm & Ứng dụng thực tế

### 9.1. Ưu điểm vượt trội
1. **Tốc độ huấn luyện và dự đoán siêu việt**: Chỉ cần một lượt quét qua dữ liệu ($O(Nd)$) để tính toán tần suất và phương sai. Nhanh gấp hàng chục đến hàng trăm lần so với SVM hay Neural Networks.
2. **Hoạt động xuất sắc trên tập dữ liệu nhỏ**: Nhờ giả định độc lập, mô hình có phương sai (variance) rất thấp, tránh hoàn toàn hiện tượng quá khớp (overfitting) khi số lượng mẫu hạn chế.
3. **Hiệu quả cao trong không gian số chiều lớn (High-Dimensional Space)**: Đặc biệt trong xử lý ngôn ngữ tự nhiên (NLP) với hàng chục nghìn từ vựng.
4. **Hỗ trợ học trực tuyến (Online Learning / Incremental Learning)**: Khi có dữ liệu mới, chỉ cần cập nhật số đếm và trung bình mà không cần huấn luyện lại toàn bộ mô hình từ đầu (phương thức `partial_fit` trong Scikit-Learn).
5. **Dễ diễn giải (Explainability)**: Có thể trích xuất trực tiếp xác suất điều kiện để giải thích tại sao một mẫu lại bị xếp vào lớp đó.

### 9.2. Nhược điểm & Hạn chế
1. **Giả thiết độc lập phi thực tế**: Bỏ qua hoàn toàn mối tương quan giữa các đặc trưng (ví dụ: trong văn bản, cụm từ "Hà" và "Nội" thường đi liền nhau).
2. **Độ tin cậy của xác suất tuyệt đối không cao (Poor Calibration)**: Mặc dù thứ tự xếp hạng `argmax` rất chính xác (giúp Accuracy cao), nhưng giá trị xác suất $P(y \mid \mathbf{x})$ thường bị đẩy quá sát về 0 hoặc 1 (overconfident) do việc nhân dồn nhiều xác suất thành phần.
3. **Nhạy cảm với dữ liệu mất cân bằng (Imbalanced Data)**: Lớp chiếm đa số sẽ có tiên nghiệm $P(y)$ lấn át các lớp thiểu số.

---

## 10. Tuyển tập 10 câu hỏi lý thuyết chuyên sâu thi Vấn đáp

### Câu 1: Tại sao Naïve Bayes vi phạm giả định độc lập nghiêm trọng nhưng trong thực nghiệm vẫn phân loại rất tốt?
**Trả lời:**
- Phân loại bằng quy tắc MAP là một bài toán **tối ưu rời rạc (0-1 Loss)**, tức là chỉ cần xác định $\arg\max_k P(y=c_k \mid \mathbf{x})$.
- Giả thiết độc lập có thể làm sai lệch giá trị xác suất tuyệt đối, nhưng **rất hiếm khi làm đảo lộn thứ tự xếp hạng tương đối giữa các lớp**. Nếu lớp đúng có xác suất thực là 0.6 và lớp sai là 0.4, Naïve Bayes có thể ước lượng thành 0.95 và 0.05. Giá trị số học bị lệch nhưng nhãn dự đoán vẫn chính xác 100%!
- Hơn nữa, theo phân tích Bias-Variance, Naïve Bayes đánh đổi một lượng Bias nhỏ (do giả định độc lập) để giảm triệt để Variance của mô hình, giúp nó bền vững trước nhiễu.

---

### Câu 2: Trình bày chi tiết bản chất của Laplace Smoothing và chứng minh tại sao mẫu số phải là $N_k + \alpha \cdot |V|$?
**Trả lời:**
- Laplace smoothing giải quyết hiện tượng xác suất bằng 0 khi một từ chưa từng xuất hiện trong lớp huấn luyện.
- Mẫu số phải cộng thêm $\alpha \cdot |V|$ để đảm bảo điều kiện tiên đề của xác suất: tổng xác suất trên toàn bộ không gian mẫu phải bằng 1:
  $$\sum_{j=1}^{|V|} \hat{p}_{k, j} = \sum_{j=1}^{|V|} \frac{N_{k, j} + \alpha}{N_k + \alpha |V|} = \frac{\sum N_{k, j} + \sum \alpha}{N_k + \alpha |V|} = \frac{N_k + \alpha |V|}{N_k + \alpha |V|} = 1$$
- Nếu không cộng $\alpha \cdot |V|$ ở mẫu, tổng xác suất sẽ vượt quá 1 và vi phạm tiên đề xác suất Kolmogorov.

---

### Câu 3: Điểm khác biệt mấu chốt giữa Multinomial Naïve Bayes và Bernoulli Naïve Bayes là gì? Cho ví dụ minh họa khi nào dùng mô hình nào?
**Trả lời:**
- **Multinomial NB**: Dựa trên tần suất từ (Count/TF). Chỉ nhân xác suất của các từ xuất hiện trong văn bản kiểm thử ($x_j > 0$). Phù hợp cho văn bản dài, bài báo, tài liệu khoa học vì tần suất lặp lại của từ chứa đựng chủ đề.
- **Bernoulli NB**: Dựa trên sự hiện diện nhị phân (0/1). Nhân cả xác suất của từ xuất hiện ($p_j$) và từ vắng mặt ($1 - p_j$). Phù hợp cho văn bản ngắn (SMS, Tweet), triệu chứng bệnh học (bệnh nhân có hay không có triệu chứng), nhận dạng chữ viết tay (pixel đen/trắng).
- *Ví dụ*: Trong phân loại thư rác, nếu email chỉ có 3 từ "khuyến mãi", "giảm giá", "tiền", Bernoulli NB rất mạnh. Nhưng trong phân loại bài báo thể thao vs kinh tế dài 1000 từ, việc từ "bóng đá" lặp lại 20 lần giúp Multinomial NB phân loại chính xác vượt trội.

---

### Câu 4: Trong Gaussian Naïve Bayes, làm thế nào để chứng minh rằng ta có thể sử dụng hàm mật độ xác suất (PDF) thay thế cho xác suất có điều kiện?
**Trả lời:**
- Ta xét xác suất biến ngẫu nhiên liên tục rơi vào lân cận $\epsilon$ cực nhỏ quanh điểm quan sát $z_j$: $P(x_j \in [z_j - \epsilon/2, z_j + \epsilon/2] \mid y) \approx \epsilon \cdot \mathcal{N}(z_j)$.
- Khi đưa vào công thức Bayes, ở cả tử số và mẫu số đều xuất hiện thừa số chung $\epsilon^d$ (với $d$ là số chiều không gian đặc trưng).
- Thừa số $\epsilon^d$ bị triệt tiêu hoàn toàn, để lại công thức chỉ chứa tích các hàm mật độ xác suất Gauss. Do đó việc dùng PDF là hoàn toàn chặt chẽ về toán học.

---

### Câu 5: Ma trận hiệp phương sai của Gaussian Naïve Bayes có dạng gì và nó liên hệ thế nào với QDA (Quadratic Discriminant Analysis)?
**Trả lời:**
- Ma trận hiệp phương sai $\mathbf{\Sigma}_k$ của Gaussian Naïve Bayes có dạng **đường chéo (Diagonal Matrix)**:
  $$\mathbf{\Sigma}_k = \operatorname{diag}(\sigma_{k, 1}^2, \sigma_{k, 2}^2, \dots, \sigma_{k, d}^2)$$
  với tất cả các phần tử ngoài đường chéo bằng 0 ($\operatorname{Cov}(X_i, X_j) = 0, \forall i \neq j$).
- Nó chính là mô hình QDA trong trường hợp đặc biệt khi áp đặt ràng buộc các thuộc tính độc lập có điều kiện. Nhờ đó, số tham số hiệp phương sai cần ước lượng giảm từ $\mathcal{O}(K \cdot d^2)$ xuống $\mathcal{O}(K \cdot d)$, giúp mô hình không bị suy biến khi số chiều lớn hơn số lượng mẫu ($d > N$).

---

### Câu 6: Thế nào là hiện tượng Numerical Underflow trong Naïve Bayes và cách khắc phục chuẩn mực trong lập trình?
**Trả lời:**
- Khi số chiều $d$ lớn, tích của hàng trăm xác suất thành phần nhỏ $p_j \in (0, 1)$ sẽ tạo ra một số cực nhỏ vượt quá giới hạn biểu diễn của kiểu số thực dấu phẩy động 64-bit (`float64` dưới $2.2 \times 10^{-308}$), dẫn đến kết quả bị gán bằng 0.
- Khắc phục bằng cách lấy logarit tự nhiên: chuyển tích xác suất thành tổng log-xác suất:
  $$\ln P(y = c_k \mid \mathbf{x}) \propto \ln P(y = c_k) + \sum_{j=1}^d \ln P(x_j \mid y = c_k)$$
- Phép toán cộng trong miền logarit hoàn toàn ổn định và bảo toàn chính xác vị trí cực đại $\arg\max$.

---

### Câu 7: Hãy giải thích tại sao trong bài toán chẩn đoán ung thư vú Wisconsin, ta phải đánh giá kỹ độ đo Recall thay vì chỉ nhìn vào Accuracy?
**Trả lời:**
- Độ chính xác Accuracy có thể bị đánh lừa trong bài toán dữ liệu y tế mất cân bằng.
- Hơn thế nữa, chi phí sai lầm (Cost of Error) giữa 2 loại lỗi là hoàn toàn bất đối xứng:
  - **False Negative (Âm tính giả)**: Bệnh nhân có khối u ác tính nhưng mô hình dự đoán là lành tính. Bệnh nhân không được điều trị kịp thời, tế bào ung thư di căn và có thể dẫn đến tử vong.
  - **False Positive (Dương tính giả)**: Bệnh nhân lành tính bị dự đoán nhầm là ác tính. Hậu quả là bệnh nhân phải làm thêm xét nghiệm sinh thiết xác nhận, gây tốn kém và lo lắng nhưng không nguy hiểm tính mạng.
- **Recall = $\frac{TP}{TP + FN}$** đo lường tỷ lệ ca ung thư thực tế được phát hiện. Recall càng cao thì số ca bị bỏ sót (FN) càng thấp, đảm bảo an toàn tối đa cho tính mạng người bệnh.

---

### Câu 8: Mô hình Hybrid Naïve Bayes hoạt động như thế nào khi dữ liệu chứa đồng thời cả biến liên tục, biến nhị phân và biến danh mục?
**Trả lời:**
- Nhờ giả thiết độc lập có điều kiện, xác suất đồng thời của vector đặc trưng được phân rã thành tích của các phân bố xác suất độc lập trên từng nhóm thuộc tính.
- Khi chuyển sang không gian Log-Likelihood, hàm mục tiêu trở thành tổng tuyến tính của:
  1. $\ln P(y=c_k)$: Log xác suất tiên nghiệm.
  2. $\sum \ln \mathcal{N}(x_{\text{cont}}; \mu, \sigma^2)$: Log mật độ Gauss cho biến liên tục.
  3. $\sum [x_{\text{bin}} \ln p + (1 - x_{\text{bin}}) \ln (1 - p)]$: Log xác suất Bernoulli cho biến nhị phân.
  4. $\sum \ln \theta_{\text{cat}}$: Log xác suất Categorical cho biến danh mục.
- Lớp nào có tổng giá trị Log-Likelihood hỗn hợp này lớn nhất sẽ là lớp được dự đoán.

---

### Câu 9: Naïve Bayes xử lý các giá trị khuyết thiếu (Missing Values) khi dự đoán như thế nào?
**Trả lời:**
- Naïve Bayes có cơ chế xử lý giá trị thiếu tự nhiên và thanh lịch bậc nhất trong Machine Learning:
- Nếu tại mẫu kiểm thử $\mathbf{x}^*$, thuộc tính thứ $j$ bị khuyết thiếu (NaN), ta chỉ việc **bỏ qua thuộc tính $j$ đó khỏi tích xác suất**:
  $$P(\mathbf{x}^* \mid y = c_k) \propto \prod_{i \neq j} P(x_i^* \mid y = c_k)$$
- Điều này tương đương với việc lấy tích phân biên duyên (marginalize out) đặc trưng bị thiếu trên toàn bộ không gian của nó: $\int P(x_1, \dots, x_d \mid y) dx_j = \prod_{i \ne j} P(x_i \mid y)$. Mô hình vẫn phân loại bình thường dựa trên các đặc trưng còn lại mà không cần phải loại bỏ mẫu hay suy đoán phức tạp.

---

### Câu 10: Naïve Bayes là mô hình tuyến tính (Linear Classifier) hay phi tuyến tính (Non-linear Classifier)?
**Trả lời:**
- **Với Multinomial NB và Bernoulli NB**: Là **mô hình tuyến tính**. Ranh giới quyết định giữa hai lớp $c_1$ và $c_2$ là phương trình:
  $$\ln \frac{P(y=c_1 \mid \mathbf{x})}{P(y=c_2 \mid \mathbf{x})} = \mathbf{w}^T \mathbf{x} + b = 0$$
  Đây là một siêu phẳng tuyến tính trong không gian đặc trưng.
- **Với Gaussian NB**:
  - Nếu phương sai của các lớp bằng nhau ($\sigma_{1, j}^2 = \sigma_{2, j}^2, \forall j$): Thành phần bậc hai $(x_j - \mu)^2$ triệt tiêu lẫn nhau, ranh giới quyết định là **tuyến tính** (tương tự LDA).
  - Nếu phương sai của các lớp khác nhau ($\sigma_{1, j}^2 \neq \sigma_{2, j}^2$): Thành phần bậc hai $\frac{x_j^2}{\sigma_{k, j}^2}$ không triệt tiêu, ranh giới quyết định là **mặt bậc hai phi tuyến tính** (parabol, elip hoặc hyperbol trong không gian đa chiều, tương tự QDA).
