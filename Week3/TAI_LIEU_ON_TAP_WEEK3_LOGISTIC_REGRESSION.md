# GIÁO TRÌNH ÔN TẬP TOÀN DIỆN TUẦN 3 (WEEK 3)
## K-NN PHÂN LOẠI NHỊ PHÂN & HỒI QUY LOGISTIC (BINARY LOGISTIC REGRESSION)

> **Sinh viên**: Nguyễn Hải Anh (MSSV: 24001968)  
> **Học phần**: Machine Learning (Học máy)  
> **Chủ đề**: Phân loại nhị phân (Binary Classification) & Ranh giới quyết định tuyến tính  
> **Thư mục liên quan**: `d:\MachineLearning\Week3`  
> **Các bài tập thực hành**: Đỗ/trượt thi (Ví dụ A.1), Nguy cơ đái tháo đường NTM (Ví dụ A.2), Tự cài SGD Logistic (Ví dụ B.1), Trực quan hóa 2D (Ví dụ B.2), Tuyển sinh cao học (`Admission_Predict.csv`), Khách hàng ngân hàng (`banking.csv`), Dự đoán bệnh tim mạch Framingham 10 năm (`framingham.csv`).

---

## MỤC LỤC
1. [Bản chất bài toán Phân loại Nhị phân (Binary Classification)](#1-bản-chất-bài-toán-phân-loại-nhị-phân-binary-classification)
2. [Phần A: Thuật toán K-NN cho Phân loại Nhị phân](#2-phần-a-thuật-toán-k-nn-cho-phân-loại-nhị-phân)
   - 2.1. Khác biệt cốt lõi giữa K-NN Hồi quy và K-NN Phân loại
   - 2.2. Cơ chế Biểu quyết đa số (Majority Voting)
   - 2.3. Quy tắc chọn tham số $K$ lẻ để tránh hòa điểm
3. [Phần B: Thuật toán Hồi quy Logistic (Logistic Regression)](#3-phần-b-thuật-toán-hồi-quy-logistic-logistic-regression)
   - 3.1. Nghịch lý tên gọi: Tại sao tên "Hồi quy" mà lại dùng để "Phân loại"?
   - 3.2. Tại sao Hồi quy tuyến tính thất bại khi phân loại 0/1?
   - 3.3. Tỷ số cược (Odds) & Hàm Log-odds (Logit)
   - 3.4. Hàm kích hoạt Sigmoid: Phép màu nén số thực về xác suất $[0, 1]$
   - 3.5. Mặt siêu phẳng ranh giới quyết định (Decision Boundary)
4. [Toán học Tối ưu hóa: Hàm mất mát Cross-Entropy & Gradient Descent](#4-toán-học-tối-ưu-hóa-hàm-mất-mát-cross-entropy--gradient-descent)
   - 4.1. Tại sao KHÔNG dùng hàm MSE cho Logistic Regression? (Bẫy hàm phi lồi)
   - 4.2. Xây dựng hàm mất mát Binary Cross-Entropy (Log Loss) từ nguyên lý MLE
   - 4.3. Đạo hàm tuyệt đẹp của hàm mất mát
   - 4.4. Giải thuật leo dốc giảm dần ngẫu nhiên (Stochastic Gradient Descent - SGD)
5. [Các độ đo đánh giá mô hình phân loại (Evaluation Metrics)](#5-các-độ-đo-đánh-giá-mô-hình-phân-loại-evaluation-metrics)
   - Ma trận nhầm lẫn (Confusion Matrix): TP, TN, FP, FN
   - Accuracy và cạm bẫy dữ liệu mất cân bằng (Imbalanced Data)
   - Precision, Recall, F1-Score: Đánh đổi đánh cược trong y tế & kinh doanh
   - Đường cong ROC và chỉ số AUC
6. [Giải phẫu mã nguồn từng dòng (Line-by-Line Code Breakdown)](#6-giải-phẫu-mã-nguồn-từng-dòng-line-by-line-code-breakdown)
   - Tự viết SGD Logistic Regression bằng NumPy
   - Kỹ thuật vẽ đường ranh giới quyết định 2D bằng `contourf`
   - Sử dụng thư viện `scikit-learn` chuẩn xác
7. [Chi tiết các bài tập thực hành Tuần 3](#7-chi-tiết-các-bài-tập-thực-hành-tuần-3)
8. [Bộ 10 câu hỏi vấn đáp bảo vệ bài tập Tuần 3 kèm đáp án chuẩn](#8-bộ-10-câu-hỏi-vấn-đáp-bảo-vệ-bài-tập-tuần-3-kèm-đáp-án-chuẩn)

---

## 1. BẢN CHẤT BÀI TOÁN PHÂN LOẠI NHỊ PHÂN (BINARY CLASSIFICATION)

Trong bài toán phân loại nhị phân, biến mục tiêu $y$ chỉ nhận một trong hai giá trị duy nhất:
$$y \in \{0, 1\}$$
- $y = 1$: Thường quy ước là lớp **Dương tính (Positive)** — sự kiện ta quan tâm (ví dụ: Mắc bệnh, Trúng tuyển, Rời bỏ dịch vụ, Gian lận thẻ tín dụng).
- $y = 0$: Quy ước là lớp **Âm tính (Negative)** — trạng thái bình thường (Khỏe mạnh, Trượt tuyển, Khách hàng trung thành, Giao dịch hợp lệ).

Mục tiêu của mô hình là tìm ra quy luật từ vector đặc trưng $\mathbf{x} = [x_1, x_2, \dots, x_d]^T$ để dự đoán xác suất mẫu đó thuộc về lớp 1:
$$P(y = 1 \mid \mathbf{x})$$
Sau đó so sánh với một ngưỡng quyết định (mặc định là $0.5$):
- Nếu $P(y = 1 \mid \mathbf{x}) \ge 0.5 \implies$ Dự đoán $\hat{y} = 1$.
- Nếu $P(y = 1 \mid \mathbf{x}) < 0.5 \implies$ Dự đoán $\hat{y} = 0$.

---

## 2. PHẦN A: THUẬT TOÁN K-NN CHO PHÂN LOẠI NHỊ PHÂN

### 2.1. Khác biệt cốt lõi giữa K-NN Hồi quy và K-NN Phân loại
Các bước đầu tiên của K-NN luôn giống nhau:
1. Tính khoảng cách Euclid $d(\mathbf{x}_{\text{test}}, \mathbf{x}_i)$ tới tất cả các điểm huấn luyện.
2. Sắp xếp khoảng cách tăng dần và lấy ra $K$ láng giềng gần nhất.

**SỰ KHÁC BIỆT NẰM Ở BƯỚC RA QUYẾT ĐỊNH CUỐI CÙNG**:
- **K-NN Hồi quy (Week 1)**: Lấy **trung bình cộng** nhãn của $K$ láng giềng: $\hat{y} = \frac{1}{K}\sum y_i$.
- **K-NN Phân loại (Week 3)**: Lấy **biểu quyết đa số (Majority Voting)** — nhãn nào xuất hiện nhiều nhất trong $K$ láng giềng thì chọn nhãn đó!

### 2.2. Cơ chế Biểu quyết đa số (Majority Voting)
Giả sử ta xét một bệnh nhân mới với $K = 5$:
- Trong 5 láng giềng gần nhất, có **3 người bị bệnh ($y = 1$)** và **2 người khỏe mạnh ($y = 0$)**.
- Ta đếm số phiếu: Phiếu cho lớp 1 là 3, phiếu cho lớp 0 là 2.
- Vì $3 > 2$, mô hình kết luận dự đoán bệnh nhân mới là: $\hat{y} = 1$.
- Xác suất ước lượng: $P(y = 1 \mid \mathbf{x}) = \frac{3}{5} = 0.6$ ($60\%$).

### 2.3. Quy tắc chọn tham số $K$ lẻ trong phân loại nhị phân
Trong bài toán nhị phân, **luôn luôn chọn $K$ là số lẻ** ($K = 3, 5, 7, 9\dots$):
- Nếu chọn $K$ chẵn (ví dụ $K = 4$), có thể xảy ra tình huống hòa điểm: 2 phiếu cho lớp 0 và 2 phiếu cho lớp 1! Lúc đó máy tính không biết phải quyết định ra sao và phải tung đồng xu ngẫu nhiên.
- Chọn $K$ lẻ triệt tiêu hoàn toàn khả năng hòa phiếu trong phân loại 2 lớp.

---

## 3. PHẦN B: THUẬT TOÁN HỒI QUY LOGISTIC (LOGISTIC REGRESSION)

### 3.1. Nghịch lý tên gọi: Tại sao tên "Hồi quy" mà lại dùng để "Phân loại"?
Đây là câu hỏi mà hầu như sinh viên nào đi thi vấn đáp cũng bị hỏi!
- **Trả lời**: Về bản chất toán học, mô hình vẫn thực hiện một phép **Hồi quy (Regression)** trên một biến số thực liên tục gọi là **Log-odds (logit)**:
  $$z = \ln \left( \frac{p}{1 - p} \right) = \mathbf{w}^T \bar{\mathbf{x}} = w_0 + w_1 x_1 + \dots + w_d x_d$$
- Nhưng sau khi tính được $z$, ta không giữ nguyên con số đó mà đưa nó qua hàm Sigmoid để nén về xác suất $[0, 1]$, rồi áp dụng ngưỡng so sánh để phân loại vào nhãn $0$ hoặc $1$.
- Do mục đích cuối cùng của người dùng là gán nhãn rời rạc, nên nó là một thuật toán **Phân loại (Classification)**.

---

### 3.2. Tại sao Hồi quy tuyến tính thất bại khi dùng để phân loại 0/1?
Nhiều người nghĩ: *"Tại sao không dùng luôn Hồi quy tuyến tính $\hat{y} = \mathbf{w}^T \bar{\mathbf{x}}$, nếu $\hat{y} \ge 0.5$ thì đoán là 1, ngược lại đoán là 0?"*

Có 2 lý do chí mạng khiến Hồi quy tuyến tính hoàn toàn thất bại:
1. **Giá trị vượt ra ngoài khoảng xác suất**: Đường thẳng tuyến tính không có giới hạn, nó có thể dự đoán $\hat{y} = -2.5$ hoặc $\hat{y} = +8.0$. Nói rằng *"Xác suất bệnh nhân bị ung thư là $-250\%$"* là một điều vô lý về mặt toán học.
2. **Cực kỳ nhạy cảm với các điểm ngoại lai (Outliers)**:  
   Giả sử ta phân loại sinh viên Đỗ ($1$) hay Trượt ($0$) dựa trên số giờ ôn thi. Hầu hết sinh viên ôn từ 2 đến 10 giờ. Bỗng nhiên có 1 sinh viên "mọt sách" ôn tới 100 giờ và đỗ. Điểm dữ liệu này nằm rất xa về bên phải. Đường thẳng hồi quy tuyến tính sẽ bị điểm này kéo lệch hẳn góc nghiêng, làm dịch chuyển ngưỡng cắt $0.5$ và khiến hàng loạt sinh viên ôn 6, 7 giờ trước đó vốn đỗ bỗng bị dự đoán nhầm thành trượt!

```
     y
 1.0 ┤      *   *   *   *   *                          * (Outlier: 100 giờ)
     │     /                                         /
     │    /                                         / (Đường thẳng bị kéo lệch)
 0.5 ┤---/-----------------------------------------/--- Ngưỡng 0.5 bị đẩy dịch sang phải!
     │  /                                         /
 0.0 ┤ *   *   *   *                             /
     └──────────────────────────────────────────┴────────> Giờ học
      0   2   4   6   8   10                    100
```

---

### 3.3. Tỷ số cược (Odds) & Hàm Log-odds (Logit)
Để xây dựng mô hình đúng đắn, các nhà toán học đã đi từ xác suất sang tỷ số cược:
- Gọi $p = P(y = 1 \mid \mathbf{x})$ là xác suất biến cố xảy ra ($0 \le p \le 1$).
- **Odds (Tỷ số cược)**: Tỷ lệ giữa xác suất xảy ra và xác suất không xảy ra:
  $$\text{Odds} = \frac{p}{1 - p}$$
  - Khi $p = 0.5 \implies \text{Odds} = \frac{0.5}{0.5} = 1$ (cơ hội ngang nhau 50-50).
  - Khi $p = 0.8 \implies \text{Odds} = \frac{0.8}{0.2} = 4$ (khả năng xảy ra cao gấp 4 lần không xảy ra).
  - Dải giá trị của Odds là: $(0, +\infty)$.
- **Logit (Log-odds)**: Lấy logarit tự nhiên của Odds:
  $$\text{logit}(p) = \ln(\text{Odds}) = \ln \left( \frac{p}{1 - p} \right)$$
  - Dải giá trị của logit mở rộng từ $(-\infty, +\infty)$!
  - Lúc này, ta hoàn toàn có thể đặt hàm logit bằng một tổ hợp tuyến tính của các biến đầu vào:
    $$\ln \left( \frac{p}{1 - p} \right) = \mathbf{w}^T \bar{\mathbf{x}}$$

---

### 3.4. Hàm kích hoạt Sigmoid: Phép màu nén số thực về xác suất $[0, 1]$
Bây giờ, ta giải phương trình trên để tìm ngược lại $p$:
$$\ln \left( \frac{p}{1 - p} \right) = z \iff \frac{p}{1 - p} = e^z \iff p = e^z(1 - p) \iff p(1 + e^z) = e^z$$
$$p = \frac{e^z}{1 + e^z} = \frac{1}{1 + e^{-z}}$$

> **Định nghĩa Hàm Sigmoid (Logistic Function)**:
> $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
> với $z = \mathbf{w}^T \bar{\mathbf{x}} = w_0 + w_1 x_1 + \dots + w_d x_d$.

```
                HÌNH HỌC HÀM SIGMOID: σ(z) = 1 / (1 + e^(-z))
                
         1.0 ┤                         . - - ' ' ' 
             │                     . '
             │                  . '
         0.5 ┤ - - - - - - - - ┼ - - - - - - - - - - Ngưỡng quyết định 0.5 (z = 0)
             │              .  │
             │           . '   │
         0.0 ┤ . . . - '       │
             └─────────────────┴───────────────────> z = w^T * x
                              z = 0
             [Vùng dự đoán: y = 0] [Vùng dự đoán: y = 1]
```

#### Các đặc tính toán học tuyệt mỹ của Sigmoid:
1. **Nén hoàn hảo**: Với bất kỳ giá trị $z$ nào từ $-\infty$ đến $+\infty$, $\sigma(z)$ luôn nằm trọn vẹn trong khoảng $(0, 1)$.
2. **Điểm đối xứng**: $\sigma(-z) = 1 - \sigma(z)$. Tại $z = 0$, $\sigma(0) = 0.5$.
3. **Đạo hàm siêu ngắn gọn**:
   $$\sigma'(z) = \frac{d}{dz} \left( \frac{1}{1 + e^{-z}} \right) = \frac{e^{-z}}{(1 + e^{-z})^2} = \sigma(z) \cdot (1 - \sigma(z))$$
   *(Tính chất này giúp giải thuật Gradient Descent tính toán cực nhanh mà không cần lặp lại phép chia phức tạp).*

---

### 3.5. Mặt siêu phẳng ranh giới quyết định (Decision Boundary)
Quy tắc phân loại:
$$\hat{y} = 1 \iff P(y = 1 \mid \mathbf{x}) \ge 0.5 \iff \sigma(\mathbf{w}^T \bar{\mathbf{x}}) \ge 0.5 \iff \mathbf{w}^T \bar{\mathbf{x}} \ge 0$$
Do đó, phương trình:
$$\mathbf{w}^T \bar{\mathbf{x}} = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_d x_d = 0$$
chính là phương trình của **Mặt phẳng siêu ranh giới (Decision Boundary)**:
- Nó chia không gian đặc trưng làm 2 nửa.
- Phía bên này mặt phẳng: $\mathbf{w}^T \bar{\mathbf{x}} > 0 \implies \hat{y} = 1$.
- Phía bên kia mặt phẳng: $\mathbf{w}^T \bar{\mathbf{x}} < 0 \implies \hat{y} = 0$.

---

## 4. TOÁN HỌC TỐI ƯU HÓA: HÀM MẤT MÁT CROSS-ENTROPY & GRADIENT DESCENT

### 4.1. Tại sao KHÔNG dùng hàm MSE cho Logistic Regression?
Nếu ta lấy hàm sai số bình phương MSE làm hàm mất mát:
$$J(\mathbf{w}) = \frac{1}{2N} \sum_{i=1}^N (y_i - \sigma(\mathbf{w}^T \bar{\mathbf{x}}_i))^2$$
Vì bên trong có hàm phi tuyến $\sigma(z)$, khi lấy đạo hàm, hàm mất mát này trở thành **Hàm không lồi (Non-convex Function)**:
- Nó chứa hàng trăm "cái hố" cực tiểu cục bộ (Local Minima) và các vùng yên ngựa (Saddle Points).
- Thuật toán Gradient Descent sẽ dễ dàng bị rơi vào một hố rác cục bộ và mắc kẹt ở đó, không bao giờ tìm được bộ trọng số tối ưu thực sự!

```
      HÀM MSE: PHI LỒI (NON-CONVEX)          HÀM CROSS-ENTROPY: LỒI (CONVEX)
             (Rất nguy hiểm)                             (Tuyệt đối an toàn)
             
       J(w)                                       J(w)
        ^                                          ^
        │   /\      /\                             │  \                /
        │  /  \    /  \                            │   \              /
        │ /    \--/    \                           │    \            /
        │/  (Hố cục bộ) \                          │     \__________/ (Cực tiểu toàn cục)
        └─────────────────> w                      └──────────────────> w
```

---

### 4.2. Xây dựng hàm mất mát Binary Cross-Entropy (Log-loss) từ nguyên lý MLE
Nguyên lý Ước lượng hợp lý cực đại (Maximum Likelihood Estimation - MLE) phát biểu:  
*"Hãy tìm bộ trọng số $\mathbf{w}$ sao cho xác suất xảy ra các sự kiện thực tế trong tập dữ liệu là lớn nhất."*

Với 1 mẫu dữ liệu $(\mathbf{x}_i, y_i)$, xác suất nhận đúng nhãn thực tế là:
$$P(y_i \mid \mathbf{x}_i) = (\hat{y}_i)^{y_i} (1 - \hat{y}_i)^{1 - y_i}$$
- Nếu $y_i = 1$: $P = \hat{y}_i^1 (1 - \hat{y}_i)^0 = \hat{y}_i$.
- Nếu $y_i = 0$: $P = \hat{y}_i^0 (1 - \hat{y}_i)^1 = 1 - \hat{y}_i$.

Hàm hợp lý toàn thể cho $N$ mẫu độc lập:
$$L(\mathbf{w}) = \prod_{i=1}^N (\hat{y}_i)^{y_i} (1 - \hat{y}_i)^{1 - y_i}$$

Để tối đa hóa tích này, ta lấy logarit tự nhiên và đổi dấu để biến bài toán Cực đại hóa thành bài toán Cực tiểu hóa hàm mất mát (Negative Log-Likelihood):

> **Hàm mất mát Binary Cross-Entropy (Log-loss)**:
> $$J(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^N \Big[ y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i) \Big]$$

**Ý nghĩa hình phạt**:
- Nếu thực tế $y_i = 1$: Mất mát là $-\ln(\hat{y}_i)$. Nếu mô hình đoán $\hat{y}_i = 1$, mất mát $= 0$. Nếu mô hình đoán $\hat{y}_i \to 0$, mất mát vọt lên $+\infty$!
- Nếu thực tế $y_i = 0$: Mất mát là $-\ln(1 - \hat{y}_i)$. Nếu mô hình đoán $\hat{y}_i \to 1$, mất mát vọt lên $+\infty$!
- **Đặc biệt**: Hàm Cross-Entropy là **Hàm lồi (Convex Function)**, chỉ có duy nhất một điểm cực tiểu toàn cục!

---

### 4.3. Đạo hàm tuyệt đẹp của hàm mất mát
Áp dụng quy tắc đạo hàm hàm hợp (Chain rule):
$$\frac{\partial J}{\partial \mathbf{w}} = \frac{\partial J}{\partial \hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial z_i} \cdot \frac{\partial z_i}{\partial \mathbf{w}}$$
1. $\frac{\partial J}{\partial \hat{y}_i} = -\frac{y_i}{\hat{y}_i} + \frac{1 - y_i}{1 - \hat{y}_i} = \frac{\hat{y}_i - y_i}{\hat{y}_i (1 - \hat{y}_i)}$
2. $\frac{\partial \hat{y}_i}{\partial z_i} = \sigma'(z_i) = \hat{y}_i (1 - \hat{y}_i)$
3. $\frac{\partial z_i}{\partial \mathbf{w}} = \bar{\mathbf{x}}_i$

Nhân 3 biểu thức lại, cụm mẫu số triệt tiêu hoàn toàn:
$$\nabla_\mathbf{w} J(\mathbf{w}) = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i) \bar{\mathbf{x}}_i$$

---

### 4.4. Giải thuật leo dốc giảm dần ngẫu nhiên (SGD)
Vì hàm mất mát này không thể giải phương trình đạo hàm bằng 0 theo cách giải tích đóng như Hồi quy tuyến tính, ta dùng thuật toán lặp **Gradient Descent**:
Quy tắc cập nhật trọng số tại mỗi bước với mẫu dữ liệu $(\mathbf{x}_i, y_i)$:
$$\mathbf{w}_{\text{new}} = \mathbf{w}_{\text{old}} - \eta \cdot \nabla_\mathbf{w} J = \mathbf{w}_{\text{old}} + \eta (y_i - \hat{y}_i) \bar{\mathbf{x}}_i$$
- $\eta > 0$: Tốc độ học (Learning Rate).
- $(y_i - \hat{y}_i)$: Sai số giữa nhãn thực và xác suất dự đoán.
  - Nếu dự đoán đúng ($y_i = 1, \hat{y}_i \approx 1$), sai số $\approx 0$, không cần chỉnh trọng số.
  - Nếu dự đoán sai nghiêm trọng, sai số lớn, trọng số sẽ bị đẩy mạnh theo hướng sửa sai.

---

## 5. CÁC ĐỘ ĐO ĐÁNH GIÁ MÔ HÌNH PHÂN LOẠI (EVALUATION METRICS)

### 5.1. Ma trận nhầm lẫn (Confusion Matrix)

```
                            MA TRẬN NHẦM LẪN (CONFUSION MATRIX)
                                    Dự đoán: 0         Dự đoán: 1
                     Thực tế: 0  ┌───────────────┬───────────────┐
                                 │ TN (True Neg) │ FP (False Pos)│  (Sai lầm loại I)
                                 ├───────────────┼───────────────┤
                     Thực tế: 1  │ FN (False Neg)│ TP (True Pos) │  (Sai lầm loại II - Tử huyệt y tế)
                                 └───────────────┴───────────────┘
```
- **TP (True Positive)**: Thực tế có bệnh, mô hình đoán có bệnh (ĐÚNG).
- **TN (True Negative)**: Thực tế khỏe mạnh, mô hình đoán khỏe mạnh (ĐÚNG).
- **FP (False Positive - Dương tính giả)**: Thực tế khỏe mạnh, nhưng máy báo có bệnh (Báo động giả).
- **FN (False Negative - Âm tính giả)**: Thực tế đang mắc bệnh nặng, nhưng máy báo khỏe mạnh (BỎ SÓT NGUY HIỂM!).

---

### 5.2. Accuracy và cạm bẫy dữ liệu mất cân bằng (Imbalanced Data)
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
> **Cạm bẫy**: Giả sử trong tập dữ liệu khám sàng lọc ung thư gồm 10,000 người, chỉ có 50 người thực sự bị bệnh (chiếm 0.5%), còn 9,950 người hoàn toàn khỏe mạnh.  
> Một lập trình viên lười biếng viết mô hình: `def predict(x): return 0` (luôn luôn đoán khỏe mạnh).  
> Mô hình này đạt **Accuracy lên tới 99.5%**! Nhưng trên thực tế, mô hình này vứt đi vì nó bỏ sót $100\%$ những người bị bệnh ung thư!  
> $\implies$ **Trong dữ liệu mất cân bằng, không bao giờ được tin vào Accuracy!**

---

### 5.3. Precision, Recall và F1-Score: Sự đánh đổi kinh điển

#### 1. Precision (Độ chuẩn xác / Độ tin cậy của lời cảnh báo)
$$\text{Precision} = \frac{TP}{TP + FP}$$
- **Ý nghĩa**: Trong số những người mà máy **bảo là có bệnh**, thực sự có bao nhiêu người bị bệnh?
- **Khi nào cần Precision cao?**: Khi hậu quả của việc "Báo động giả" (False Positive) là rất đắt đỏ.
  - Ví dụ: Lọc email Spam. Ta thà để sót một bức thư rác vào hộp thư chính (False Negative) còn hơn là ném nhầm một hợp đồng làm ăn triệu USD của đối tác vào thùng rác Spam (False Positive)!

#### 2. Recall (Độ nhạy / Khả năng truy quét)
$$\text{Recall} = \frac{TP}{TP + FN}$$
- **Ý nghĩa**: Trong số tất cả những người **thực sự bị bệnh ngoài đời**, mô hình đã bắt được bao nhiêu người?
- **Khi nào cần Recall cao?**: Khi hậu quả của việc "Bỏ sót" (False Negative) là thảm họa chết người!
  - Ví dụ: Chẩn đoán ung thư, dự đoán nhồi máu cơ tim, phát hiện bom mìn sân bay. Thà kiểm tra nhầm một người khỏe mạnh (FP) còn hơn bỏ lọt một người mang bom lên máy bay (FN)!

#### 3. F1-Score (Điểm điều hòa)
$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
- Cân bằng hài hòa giữa Precision và Recall. Nếu một trong hai chỉ số quá tệ, $F_1$ sẽ bị kéo tụt xuống ngay.

---

### 5.4. Đường cong ROC và chỉ số AUC
- **ROC (Receiver Operating Characteristic)**: Đồ thị biểu diễn mối quan hệ giữa **TPR (True Positive Rate = Recall)** và **FPR (False Positive Rate = $\frac{FP}{FP + TN}$)** khi ta thay đổi ngưỡng cắt xác suất từ $1.0$ dần về $0.0$.
- **AUC (Area Under Curve)**: Diện tích nằm dưới đường cong ROC (từ 0.0 đến 1.0).
  - $\text{AUC} = 1.0$: Mô hình hoàn hảo tuyệt đối.
  - $\text{AUC} = 0.5$: Mô hình đoán bừa như tung đồng xu may rủi.
  - $\text{AUC} > 0.8$: Mô hình xuất sắc.

---

## 6. GIẢI PHẪU MÃ NGUỒN TỪNG DÒNG (LINE-BY-LINE CODE BREAKDOWN)

### A. Tự viết Stochastic Gradient Descent cho Hồi quy Logistic bằng NumPy
```python
import numpy as np

# 1. Hàm Sigmoid
def sigmoid(z):
    # Nén giá trị z về khoảng (0, 1)
    return 1.0 / (1.0 + np.exp(-z))

# 2. Thuật toán huấn luyện SGD
def logistic_regression_sgd(X, y, eta=0.05, max_epochs=100):
    # X có shape (N, d), y có shape (N,)
    N, d = X.shape
    
    # Kỹ thuật Bias trick: Thêm cột số 1 vào đầu ma trận X -> (N, d + 1)
    Xbar = np.concatenate((np.ones((N, 1)), X), axis=1)
    
    # Khởi tạo vector trọng số w ngẫu nhiên hoặc bằng 0 có kích thước (d + 1, 1)
    w = np.zeros((d + 1, 1))
    y = y.reshape(N, 1)  # Đảm bảo y có kích thước (N, 1)
    
    for epoch in range(max_epochs):
        # Trộn ngẫu nhiên thứ tự các mẫu (Stochastic)
        permutation = np.random.permutation(N)
        for i in permutation:
            xi = Xbar[i, :].reshape(d + 1, 1)  # Lấy mẫu thứ i dạng cột (d + 1, 1)
            yi = y[i, 0]                       # Nhãn thực tế (0 hoặc 1)
            
            # Tính xác suất dự đoán y_hat = sigmoid(w^T * xi)
            zi = np.dot(w.T, xi)
            y_hat = sigmoid(zi)
            
            # Cập nhật trọng số theo đạo hàm: w = w + eta * (y - y_hat) * x
            w = w + eta * (yi - y_hat) * xi
            
    return w

# 3. Hàm dự đoán xác suất và nhãn
def predict(w, X_new, threshold=0.5):
    N_new = X_new.shape[0]
    Xbar_new = np.concatenate((np.ones((N_new, 1)), X_new), axis=1)
    probabilities = sigmoid(np.dot(Xbar_new, w))
    # Gán nhãn 1 nếu xác suất >= threshold, ngược lại gán 0
    predictions = (probabilities >= threshold).astype(int)
    return predictions, probabilities
```

---

### B. Kỹ thuật vẽ đường ranh giới quyết định (Decision Boundary) 2D
```python
import matplotlib.pyplot as plt

def plot_decision_boundary(X, y, w):
    # X gồm 2 đặc trưng (N, 2)
    # Ranh giới quyết định là phương trình: w0 + w1*x1 + w2*x2 = 0
    # => x2 = -(w0 + w1*x1) / w2
    
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', label='Class 0')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label='Class 1')
    
    # Tạo dải giá trị cho trục hoành x1
    x1_values = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    # Tính đường thẳng x2 tương ứng
    x2_values = -(w[0] + w[1] * x1_values) / w[2]
    
    plt.plot(x1_values, x2_values, color='green', linewidth=2, label='Decision Boundary')
    plt.legend()
    plt.show()
```

---

### C. Sử dụng `scikit-learn` chuẩn mực
```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# BƯỚC 1: BẮT BUỘC CHUẨN HÓA DỮ LIỆU ĐỂ GRADIENT DESCENT HỘI TỤ
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# BƯỚC 2: KHỞI TẠO VÀ HUẤN LUYỆN MÔ HÌNH
# solver='liblinear' hoặc 'lbfgs'
clf = LogisticRegression(solver='liblinear', random_state=42)
clf.fit(X_train_scaled, y_train)

# BƯỚC 3: DỰ ĐOÁN NHÃN VÀ XÁC SUẤT
y_pred = clf.predict(X_test_scaled)
y_prob = clf.predict_proba(X_test_scaled)[:, 1] # Cột 1 là xác suất lớp 1

# BƯỚC 4: BÁO CÁO KẾT QUẢ ĐẦY ĐỦ
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
```

---

## 7. CHI TIẾT CÁC BÀI TẬP THỰC HÀNH TUẦN 3

### 1. Phân loại sinh viên Đỗ / Trượt (Ví dụ A.1)
- Áp dụng K-NN phân loại nhị phân trên bộ dữ liệu nhỏ mẫu để hiểu cách đếm số phiếu `highest_rank`.

### 2. Dự đoán nguy cơ đái tháo đường NTM (Ví dụ A.2)
- Biến đổi nhãn liên tục bề dày màng mạch NTM thành nhãn nhị phân dựa trên ngưỡng y tế (ví dụ: dày $> 1.0 \text{mm} \implies 1$, ngược lại $\implies 0$).

### 3. Tự viết SGD và Trực quan hóa ranh giới 2D (Ví dụ B.1 & B.2)
- Sinh 2 cụm điểm Gauss ngẫu nhiên trên mặt phẳng 2D, chạy SGD từ đầu bằng NumPy và vẽ đường thẳng phân chia màu xanh ngăn cách 2 đám mây điểm.

### 4. Dự đoán trúng tuyển cao học (`Admission_Predict.csv` - Bài tập 1)
- Cột mục tiêu `Chance of Admit` là số thực từ 0 đến 1. Ta chuyển thành bài toán nhị phân bằng cách đặt ngưỡng: `Chance of Admit >= 0.75` là Đỗ ($1$), ngược lại là Trượt ($0$).

### 5. Phân loại khách hàng gửi tiết kiệm (`banking.csv` - Bài tập 2)
- Dữ liệu Marketing ngân hàng với hơn 40,000 khách hàng, rất nhiều thuộc tính phân loại (nghề nghiệp, học vấn, tình trạng hôn nhân...). Áp dụng One-Hot Encoding cho các biến rời rạc trước khi huấn luyện Logistic Regression.

### 6. Dự đoán nguy cơ bệnh tim mạch Framingham 10 năm (`framingham.csv` - Bài tập 3)
- Bài toán y tế thực tế kinh điển: Dự đoán nguy cơ mắc bệnh tim mạch vành trong 10 năm tới dựa trên tuổi tác, huyết áp, hút thuốc, cholesterol...
- **Điểm mấu chốt**: Dữ liệu có nhiều giá trị bị khuyết (`NaN`) $\implies$ Phải xử lý bằng `dropna()` hoặc điền trung vị (`fillna(median)`), và dữ liệu bị mất cân bằng lớp trầm trọng $\implies$ Phải tập trung tối ưu chỉ số **Recall** và **ROC-AUC**.

---

## 8. BỘ 10 CÂU HỎI VẤN ĐÁP BẢO VỆ BÀI TẬP TUẦN 3 KÈM ĐÁP ÁN CHUẨN

1. **Câu 1: Tại sao mô hình có tên là "Hồi quy Logistic" nhưng lại dùng cho bài toán Phân loại?**  
   *Đáp*: Vì mô hình thực hiện hồi quy tuyến tính trên đại lượng log-odds liên tục $z = \mathbf{w}^T \bar{\mathbf{x}}$, sau đó ánh xạ giá trị này qua hàm Sigmoid để thu được xác suất $[0, 1]$ và áp dụng ngưỡng phân chia (ví dụ $0.5$) để phân loại mẫu vào lớp $0$ hoặc $1$.

2. **Câu 2: Công thức hàm Sigmoid là gì và đạo hàm của nó có gì đặc biệt?**  
   *Đáp*: $\sigma(z) = \frac{1}{1 + e^{-z}}$. Đạo hàm của nó có dạng cực kỳ ngắn gọn và đẹp: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.

3. **Câu 3: Tại sao không dùng hàm mất mát MSE cho Logistic Regression?**  
   *Đáp*: Khi kết hợp hàm Sigmoid phi tuyến với hàm mất mát MSE, hàm mất mát sẽ trở thành phi lồi (Non-convex), chứa nhiều điểm cực tiểu cục bộ khiến giải thuật Gradient Descent dễ bị mắc kẹt.

4. **Câu 4: Hàm mất mát chuẩn trong Hồi quy Logistic là gì?**  
   *Đáp*: Là hàm **Binary Cross-Entropy (Log-loss)**: $J(\mathbf{w}) = -\frac{1}{N}\sum [y_i \ln \hat{y}_i + (1 - y_i)\ln(1 - \hat{y}_i)]$. Đây là hàm lồi (Convex), đảm bảo luôn tìm được cực tiểu toàn cục duy nhất.

5. **Câu 5: Ranh giới quyết định (Decision Boundary) của Hồi quy Logistic có dạng hình học gì?**  
   *Đáp*: Là một siêu phẳng tuyến tính có phương trình $\mathbf{w}^T \bar{\mathbf{x}} = 0$ (trong không gian 2D nó là một đường thẳng, trong 3D là một mặt phẳng).

6. **Câu 6: Phân biệt sự khác nhau giữa Precision và Recall qua ví dụ thực tế.**  
   *Đáp*: 
   - Precision: "Máy báo có bệnh thì tin được bao nhiêu %?" Quan trọng trong lọc Spam, tránh cảnh báo nhầm.
   - Recall: "Có bao nhiêu % bệnh nhân thực tế bị máy bắt được?" Quan trọng trong chẩn đoán y tế, tránh bỏ lọt bệnh nhân nặng.

7. **Câu 7: Khi tập dữ liệu bị mất cân bằng lớp trầm trọng (Imbalanced Data), độ đo nào bị vô hiệu hóa? Vì sao?**  
   *Đáp*: Độ đo Accuracy bị vô hiệu hóa. Vì mô hình chỉ cần dự đoán tất cả các mẫu vào lớp chiếm đa số thì Accuracy đã đạt con số rất cao (ví dụ 99%), nhưng hoàn toàn không bắt được mẫu nào của lớp thiểu số quan trọng.

8. **Câu 8: Khác biệt giữa Batch Gradient Descent và Stochastic Gradient Descent (SGD) là gì?**  
   *Đáp*: Batch GD dùng toàn bộ $N$ mẫu dữ liệu trong tập train để tính đạo hàm một lần rồi mới cập nhật trọng số (chính xác nhưng rất chậm khi $N$ lớn). SGD mỗi bước chỉ bốc ngẫu nhiên đúng $1$ mẫu dữ liệu để cập nhật trọng số ngay lập tức (chạy siêu nhanh, thoát khỏi điểm cực tiểu bẫy, nhưng đường đi bị dao động nhiễu).

9. **Câu 9: Ngưỡng quyết định (Decision Threshold) có bắt buộc phải cố định là 0.5 không? Khi nào nên đổi?**  
   *Đáp*: Không bắt buộc. Trong y tế, nếu muốn tăng Recall (bắt triệt để người mắc bệnh hiểm nghèo), ta có thể hạ ngưỡng xuống $0.3$ hoặc $0.2$. Trong phát hiện gian lận giao dịch khắt khe, ta có thể nâng ngưỡng lên $0.7$ để tránh làm phiền khách hàng.

10. **Câu 10: Tại sao trong K-NN phân loại nhị phân ta luôn luôn chọn $K$ là số lẻ?**  
    *Đáp*: Để tránh hiện tượng hòa phiếu (Tie-breaking) giữa 2 lớp khi đếm số phiếu bầu từ $K$ láng giềng gần nhất.
