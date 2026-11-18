# GIÁO TRÌNH ÔN TẬP TOÀN DIỆN TUẦN 4 (WEEK 4)
## K-NN PHÂN LOẠI NHIỀU LỚP & HỒI QUY SOFTMAX (MULTINOMIAL LOGISTIC REGRESSION) & BÀI TOÁN NHẬN DẠNG CHỮ SỐ VIẾT TAY MNIST

> **Sinh viên**: Nguyễn Hải Anh (MSSV: 24001968)  
> **Học phần**: Machine Learning (Học máy)  
> **Chủ đề**: Phân loại đa lớp (Multiclass Classification), Softmax Regression & Nhận dạng hình ảnh  
> **Thư mục liên quan**: `d:\MachineLearning\week4`  
> **Các bài tập thực hành**: Phân loại 3 cụm Gauss 2D (Ví dụ A.1/B.1), Phân loại hoa Iris 4 chiều + PCA (Ví dụ A.2/B.2), Phân loại văn bản 20 Newsgroups (Ví dụ B.3), Phân loại 7 loại kính (`glass.csv` - Ví dụ B.4), Nhận dạng chữ số viết tay MNIST 784 chiều (Ví dụ 4).

---

## MỤC LỤC
1. [Bản chất bài toán Phân loại Nhiều lớp (Multiclass Classification)](#1-bản-chất-bài-toán-phân-loại-nhiều-lớp-multiclass-classification)
2. [Phần A: Thuật toán K-NN cho Phân loại Nhiều lớp](#2-phần-a-thuật-toán-k-nn-cho-phân-loại-nhiều-lớp)
   - 2.1. Khái quát hóa thuật toán lên $C$ lớp ($C \ge 3$)
   - 2.2. Xây dựng phương thức tìm nhãn có tần suất cao nhất (`highest_rank`)
   - 2.3. Ví dụ dữ liệu 3 cụm Gauss 2D & Trực quan hóa
   - 2.4. Phân loại hoa Iris & Kỹ thuật giảm chiều PCA (Principal Component Analysis)
3. [Phần B: Thuật toán Hồi quy Softmax (Multinomial Logistic Regression)](#3-phần-b-thuật-toán-hồi-quy-softmax-multinomial-logistic-regression)
   - 3.1. Từ Hàm Sigmoid (2 lớp) đến Hàm Softmax (nhiều lớp)
   - 3.2. Công thức toán học hàm Softmax
   - 3.3. Kỹ thuật ổn định số học (Softmax Stable Trick - Chống tràn số `exp`)
   - 3.4. Mã hóa One-hot (One-hot Encoding) & Biểu diễn ma trận thưa
   - 3.5. Hàm mất mát Categorical Cross-Entropy
   - 3.6. Đạo hàm và Quy tắc cập nhật trọng số Stochastic Gradient Descent (SGD)
4. [Dự án thực tế lớn: Nhận dạng chữ số viết tay MNIST](#4-dự-án-thực-tế-lớn-nhận-dạng-chữ-số-viết-tay-mnist)
   - 4.1. Bản chất dữ liệu ảnh $28 \times 28$ & Không gian đặc trưng 784 chiều
   - 4.2. Cấu trúc file nhị phân IDX và cách giải nén file `.gz` bằng byte
   - 4.3. Huấn luyện mô hình Softmax tự xây dựng vs `scikit-learn`
5. [Giải phẫu mã nguồn từng dòng (Line-by-Line Code Breakdown)](#5-giải-phẫu-mã-nguồn-từng-dòng-line-by-line-code-breakdown)
   - Code `convert_labels` ma trận thưa
   - Code `softmax_stable`
   - Vòng lặp huấn luyện SGD Softmax
   - Kỹ thuật vẽ vùng phân chia đa lớp bằng `contourf`
6. [Chi tiết các bài tập thực hành Tuần 4](#6-chi-tiết-các-bài-tập-thực-hành-tuần-4)
7. [Bộ 10 câu hỏi vấn đáp bảo vệ bài tập Tuần 4 kèm đáp án chuẩn](#7-bộ-10-câu-hỏi-vấn-đáp-bảo-vệ-bài-tập-tuần-4-kèm-đáp-án-chuẩn)

---

## 1. BẢN CHẤT BÀI TOÁN PHÂN LOẠI NHIỀU LỚP (MULTICLASS CLASSIFICATION)

Trong Tuần 3, chúng ta đã nghiên cứu bài toán phân loại nhị phân ($C = 2$ lớp: $0$ hoặc $1$).  
Tuy nhiên, thế giới thực phần lớn là các bài toán có từ 3 nhóm trở lên ($C \ge 3$):
- Phân loại hoa Iris vào $3$ loài: Setosa ($0$), Versicolor ($1$), Virginica ($2$).
- Phân loại $7$ loại kính theo thành phần hóa học (`glass.csv`): Kính cửa sổ nổi, kính xe hơi, bóng đèn, hộp đựng...
- Phân loại $10$ chữ số viết tay từ ảnh chụp: $0, 1, 2, \dots, 9$ (bộ dữ liệu MNIST).
- Phân loại các bài báo vào $20$ chủ đề tin tức khác nhau (bộ dữ liệu 20 Newsgroups).

> **Mục tiêu của Mô hình Phân loại Đa lớp**:  
> Với mỗi vector đặc trưng đầu vào $\mathbf{x} \in \mathbb{R}^d$, mô hình phải tính ra một vector phân phối xác suất gồm $C$ phần tử:
> $$\mathbf{a} = [a_0, a_1, \dots, a_{C-1}]^T, \quad \text{với } a_c = P(y = c \mid \mathbf{x})$$
> thỏa mãn 2 điều kiện tiên quyết:
> 1. $a_c > 0, \quad \forall c \in \{0, 1, \dots, C-1\}$ (Tất cả các xác suất đều dương).
> 2. $\sum_{c=0}^{C-1} a_c = 1.0$ (Tổng xác suất của $C$ lớp phải bằng đúng $100\%$).
>
> Nhãn dự đoán cuối cùng là lớp có xác suất cao nhất:
> $$\hat{y} = \arg\max_{c} a_c$$

---

## 2. PHẦN A: THUẬT TOÁN K-NN CHO PHÂN LOẠI NHIỀU LỚP

### 2.1. Khái quát hóa thuật toán lên $C$ lớp
Nguyên lý của K-NN khi mở rộng lên $C$ lớp ($C \ge 3$) vẫn giữ trọn vẹn bản chất tự nhiên:
1. Cho một điểm mới $\mathbf{x}_{\text{unseen}}$ (trong tập Validation hoặc Test).
2. Tính khoảng cách Euclid $d$ từ $\mathbf{x}_{\text{unseen}}$ tới tất cả các điểm trong tập huấn luyện:
   $$d(\mathbf{x}_{\text{unseen}}, \mathbf{x}_i) = \|\mathbf{x}_{\text{unseen}} - \mathbf{x}_i\|_2$$
3. Sắp xếp mảng khoảng cách tăng dần bằng `np.argsort`.
4. Chọn ra $K$ chỉ số có khoảng cách nhỏ nhất, trích xuất danh sách $K$ nhãn tương ứng: $\{y_{n_1}, y_{n_2}, \dots, y_{n_K}\}$.
5. **Dự đoán nhãn**: Nhãn nào xuất hiện với **tần suất nhiều nhất** trong danh sách đó sẽ được chọn làm nhãn dự đoán $\hat{y}$.

---

### 2.2. Xây dựng phương thức tìm nhãn có tần suất cao nhất (`highest_rank`)
Trong tài liệu thực hành PDF của môn học, đoạn code sau được dùng để tìm phần tử xuất hiện nhiều nhất trong một danh sách:

```python
def highest_rank(arr):
    # arr là một list các nhãn của K láng giềng, ví dụ: [1, 2, 2, 0, 2]
    count_num = {}
    for i in arr:
        count_num[i] = arr.count(i)
    # max với key=count_num.get trả về key có value (số lần đếm) lớn nhất
    return max(count_num, key=count_num.get)
```
- **Ví dụ**: Nếu `arr = [0, 2, 1, 2, 2]`:
  - `count_num = {0: 1, 2: 3, 1: 1}`.
  - Nhãn `2` có tần suất cao nhất ($3$ lần), do đó hàm trả về `2`.

---

### 2.3. Ví dụ A.1: Dữ liệu nhân tạo 3 cụm Gauss 2D & Trực quan hóa
- **Dữ liệu**: Tạo ra $N = 1500$ điểm trên mặt phẳng 2D ($d = 2$), chia đều cho $C = 3$ lớp (mỗi lớp 500 điểm).
- **Tâm của 3 cụm**: $C_0 = (2, 2)$, $C_1 = (8, 3)$, $C_2 = (3, 6)$.
- **Ma trận hiệp phương sai chung**: $\Sigma = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$ (các điểm phân bố hình tròn đều xung quanh tâm).
- **Phân chia dữ liệu**: Mỗi lớp lấy 400 điểm đầu làm tập Train (tổng 1200 điểm), 100 điểm sau làm tập Validation (tổng 300 điểm).
- **Thực nghiệm K-NN**: Đặt $K = 20$. K-NN phân loại chính xác hầu như toàn bộ tập Validation vì 3 cụm này tách biệt khá rõ ràng trên mặt phẳng.

---

### 2.4. Ví dụ A.2: Phân loại hoa Iris & Kỹ thuật giảm chiều PCA
- **Bộ dữ liệu Iris**: Gồm 150 mẫu hoa, 3 lớp ($C = 3$), mỗi lớp 50 mẫu.
- **Thách thức**: Mỗi bông hoa có tới 4 đặc trưng ($d = 4$: Sepal length, Sepal width, Petal length, Petal width). Mắt người chỉ nhìn được tối đa không gian 2D hoặc 3D, không thể vẽ trực tiếp 4 chiều lên màn hình phẳng!
- **Giải pháp: Kỹ thuật Phân tích Thành phần Chính (PCA - Principal Component Analysis)**:
  - PCA chiếu dữ liệu từ không gian 4 chiều xuống mặt phẳng 2 chiều sao cho **bảo toàn tối đa phương sai (lượng thông tin)** của dữ liệu ban đầu.
  ```python
  from sklearn.decomposition import PCA as sklearnPCA
  # Chiếu từ 4 chiều xuống 2 chiều
  pca = sklearnPCA(n_components=2)
  transformed = pca.fit_transform(X_norm)
  ```
  Sau khi giảm xuống 2D, ta dùng hàm `plt.scatter` để vẽ các loài hoa với 3 màu sắc khác nhau (Đỏ: Setosa, Xanh lá: Versicolor, Xanh dương: Virginica).

---

## 3. PHẦN B: THUẬT TOÁN HỒI QUY SOFTMAX (MULTINOMIAL LOGISTIC REGRESSION)

### 3.1. Từ Hàm Sigmoid (2 lớp) đến Hàm Softmax (nhiều lớp)
- Trong bài toán 2 lớp (Tuần 3), ta chỉ cần 1 vector trọng số $\mathbf{w}$ để tính 1 điểm số $z = \mathbf{w}^T \bar{\mathbf{x}}$, rồi đưa qua hàm Sigmoid:
  $$P(y = 1 \mid \mathbf{x}) = \frac{1}{1 + e^{-z}} = \frac{e^z}{e^z + 1}$$
  Khi đó xác suất lớp 0 tự động bằng $1 - P(y=1)$.
- Nhưng khi có $C$ lớp ($C \ge 3$), mỗi lớp sẽ có một bộ trọng số riêng: $\mathbf{w}_0, \mathbf{w}_1, \dots, \mathbf{w}_{C-1}$.  
  Ta ghép $C$ vector này thành **Ma trận trọng số toàn thể $\mathbf{W} \in \mathbb{R}^{(d+1) \times C}$**:
  $$\mathbf{W} = [\mathbf{w}_0, \mathbf{w}_1, \dots, \mathbf{w}_{C-1}]$$
  Với mỗi mẫu $\mathbf{x}$, ta tính được một vector gồm $C$ điểm số thô (Logits):
  $$\mathbf{z} = \mathbf{W}^T \bar{\mathbf{x}} = [z_0, z_1, \dots, z_{C-1}]^T, \quad \text{với } z_c = \mathbf{w}_c^T \bar{\mathbf{x}}$$
  Các điểm số $z_c$ này có thể âm, dương, rất lớn hoặc rất bé. Làm thế nào để chuyển đổi chúng thành xác suất hợp lệ?

---

### 3.2. Công thức toán học hàm Softmax

> **Định nghĩa Hàm Softmax**:
> Với mỗi điểm số $z_c$ ($c = 0, 1, \dots, C-1$), xác suất để mẫu $\mathbf{x}$ thuộc về lớp $c$ là:
> $$a_c = \text{Softmax}(z_c) = \frac{e^{z_c}}{\sum_{k=0}^{C-1} e^{z_k}}$$

```
                          CƠ CHẾ HOẠT ĐỘNG CỦA SOFTMAX
                          
Điểm số thô (Logits z)         Hàm mũ e^z                 Xác suất a_c (Tổng = 1.0)
     ┌───────────┐           ┌───────────┐                  ┌───────────┐
z0 = │   2.0     │    exp    │   7.39    │    Chia cho      │   0.17    │ (17%)
z1 = │   1.0     │  ======>  │   2.72    │  Tổng = 42.92    │   0.06    │ (6%)
z2 = │   3.5     │           │  33.12    │  ===========>    │   0.77    │ (77% -> Dự đoán lớp 2!)
     └───────────┘           └───────────┘                  └───────────┘
```

**Tại sao Softmax lại hoàn hảo?**
1. Hàm mũ $e^{z_c}$ luôn luôn dương ($> 0$), bất chấp $z_c$ là số âm bao nhiêu chăng nữa.
2. Mẫu số là tổng của tất cả các tử số, do đó tổng các $a_c$ luôn luôn bằng đúng $1.0$ ($100\%$).
3. Softmax có tính chất "kéo dãn khoảng cách": Phần tử có điểm số $z$ lớn nhất sẽ được số mũ $e^z$ khuếch đại mạnh mẽ, chiếm phần lớn tỷ trọng xác suất (gần giống như toán tử `max`, nhưng mượt mà hơn nên gọi là `soft-max`).

---

### 3.3. Kỹ thuật ổn định số học (Softmax Stable Trick - BẮT BUỘC PHẢI BIẾT)

#### Vấn đề Tràn số dương (Floating-Point Overflow):
Trong máy tính, kiểu số thực 64-bit chỉ có thể biểu diễn giá trị tối đa khoảng $1.8 \times 10^{308}$.  
Nếu trong quá trình tính toán, một điểm số $z_c$ đạt giá trị $z_c = 1000$:
$$e^{1000} \approx 1.97 \times 10^{434} \implies \text{TRÀN SỐ! (Máy tính báo lỗi inf hoặc NaN)}$$
Khi xuất hiện `inf`, phép chia `inf / inf` sẽ cho ra kết quả `NaN` (Not a Number), làm sụp đổ hoàn toàn quá trình huấn luyện mô hình!

#### Giải pháp toán học thông minh:
Hàm Softmax có một tính chất bất biến kỳ diệu: **Nếu trừ tất cả các phần tử $z_k$ cho cùng một hằng số $M$, giá trị xác suất Softmax không hề thay đổi!**
$$\frac{e^{z_c - M}}{\sum_{k=0}^{C-1} e^{z_k - M}} = \frac{e^{-M} \cdot e^{z_c}}{e^{-M} \cdot \sum_{k=0}^{C-1} e^{z_k}} = \frac{e^{z_c}}{\sum_{k=0}^{C-1} e^{z_k}}$$

Ta chọn $M = \max(z_0, z_1, \dots, z_{C-1})$ (giá trị lớn nhất trong vector):
- Khi đó số lớn nhất sẽ trở thành: $z_{\max} - z_{\max} = 0 \implies e^0 = 1$.
- Tất cả các số còn lại đều $\le 0 \implies e^{z_k - z_{\max}} \le 1$.
- Số mũ không bao giờ vượt quá 1, nguy cơ tràn số bị **triệt tiêu hoàn toàn $100\%$**!

```python
def softmax_stable(Z):
    # Z có kích thước (C, N) - mỗi cột là điểm số của một mẫu
    # Trừ giá trị lớn nhất trên từng cột (axis=0, keepdims=True)
    e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    A = e_Z / e_Z.sum(axis=0)
    return A
```

---

### 3.4. Mã hóa One-hot (One-hot Encoding)
Trong bài toán đa lớp, nhãn của mẫu thứ $i$ ban đầu là một số nguyên: $y_i \in \{0, 1, \dots, C-1\}$.  
Ta không thể để nguyên số nguyên $0, 1, 2$ để huấn luyện vì mô hình sẽ ngộ nhận sai lầm rằng lớp 2 "lớn gấp đôi" lớp 1, hoặc lớp 2 "ở gần" lớp 1 hơn lớp 0!

**Mã hóa One-hot**: Chuyển nhãn số nguyên thành vector $C$ chiều, trong đó **chỉ có duy nhất vị trí nhãn thực tế mang giá trị 1**, tất cả các vị trí khác mang giá trị 0:
Ví dụ với $C = 3$ lớp:
- Nhãn $0 \implies \mathbf{y} = [1, 0, 0]^T$
- Nhãn $1 \implies \mathbf{y} = [0, 1, 0]^T$
- Nhãn $2 \implies \mathbf{y} = [0, 0, 1]^T$

```python
from scipy import sparse

def convert_labels(y, C):
    # Biến vector nhãn 1D độ dài N thành ma trận One-hot shape (C, N)
    N = len(y)
    # y là chỉ số dòng, np.arange(N) là chỉ số cột
    Y = sparse.coo_matrix((np.ones_like(y), (y, np.arange(N))), shape=(C, N)).toarray()
    return Y
```

---

### 3.5. Hàm mất mát Categorical Cross-Entropy
Với $N$ mẫu dữ liệu, hàm mất mát toàn thể là:
$$J(\mathbf{W}) = -\sum_{i=1}^N \sum_{c=0}^{C-1} y_{ic} \ln(a_{ic})$$
Trong đó:
- $y_{ic}$ là phần tử thứ $c$ của vector nhãn One-hot của mẫu $i$.
- $a_{ic}$ là xác suất dự đoán qua hàm Softmax rằng mẫu $i$ thuộc về lớp $c$.

**Bản chất thực tế cực kỳ đơn giản**:  
Vì vector nhãn One-hot chỉ có duy nhất vị trí lớp đúng mang giá trị $1$, còn tất cả các vị trí khác đều bằng $0$, nên phép tổng trên thực chất chỉ rút gọn thành:
$$J(\mathbf{W}) = -\sum_{i=1}^N \ln(a_{i, \text{lớp đúng}})$$
- Nếu mô hình dự đoán xác suất của lớp đúng càng cao ($a \to 1$), thì $-\ln(1) = 0 \implies$ Mất mát bằng $0$.
- Nếu mô hình dự đoán xác suất của lớp đúng quá thấp ($a \to 0$), thì $-\ln(0) \to +\infty \implies$ Bị phạt cực nặng!

---

### 3.6. Đạo hàm và Quy tắc cập nhật Stochastic Gradient Descent (SGD)
Đạo hàm của hàm mất mát Categorical Cross-Entropy theo ma trận trọng số $\mathbf{W}$ đối với một mẫu dữ liệu $(\mathbf{x}_i, \mathbf{y}_i)$ có dạng đại số tuyệt đẹp:
$$\nabla_\mathbf{W} J = \bar{\mathbf{x}}_i (\mathbf{a}_i - \mathbf{y}_i)^T$$
Trong đó:
- $\bar{\mathbf{x}}_i \in \mathbb{R}^{(d+1) \times 1}$: Vector đặc trưng của mẫu thứ $i$ (đã thêm số 1).
- $\mathbf{a}_i \in \mathbb{R}^{C \times 1}$: Vector xác suất dự đoán từ hàm Softmax.
- $\mathbf{y}_i \in \mathbb{R}^{C \times 1}$: Vector nhãn One-hot thực tế.
- $(\mathbf{a}_i - \mathbf{y}_i)^T \in \mathbb{R}^{1 \times C}$: Vector sai số dự đoán.

**Công thức cập nhật trọng số**:
$$\mathbf{W}_{\text{new}} = \mathbf{W}_{\text{old}} - \eta \cdot \nabla_\mathbf{W} J = \mathbf{W}_{\text{old}} + \eta \bar{\mathbf{x}}_i (\mathbf{y}_i - \mathbf{a}_i)^T$$

---

## 4. DỰ ÁN THỰC TẾ LỚN: NHẬN DẠNG CHỮ SỐ VIẾT TAY MNIST

### 4.1. Bản chất dữ liệu ảnh $28 \times 28$ & Không gian 784 chiều
Bộ dữ liệu MNIST (Modified National Institute of Standards and Technology) được xem là "bài toán Hello World" kinh điển của thị giác máy tính và học máy:
- Gồm **60,000 ảnh huấn luyện** và **10,000 ảnh kiểm tra** của các chữ số viết tay từ 0 đến 9 ($C = 10$ lớp).
- Mỗi bức ảnh là ảnh xám (Grayscale) có kích thước: $28 \times 28$ pixel.
- Mỗi pixel là một số nguyên từ $0$ (nền trắng) đến $255$ (nét mực đen).
- **Trải phẳng (Flattening)**: Để đưa vào mô hình Softmax Regression, ta trải phẳng ma trận 2 chiều $28 \times 28$ thành một vector 1 chiều có **$d = 28 \times 28 = 784$ thuộc tính**!
- Sau khi thêm biến giả bias $x_0 = 1$, ma trận trọng số $\mathbf{W}$ sẽ có kích thước:
  $$\mathbf{W} \in \mathbb{R}^{785 \times 10}$$
  Mô hình sẽ học $785 \times 10 = 7,850$ tham số trọng số để nhận diện chữ số!

---

### 4.2. Cấu trúc file nhị phân IDX và cách đọc bằng Python
Dữ liệu MNIST được lưu dưới dạng file nén nhị phân (`.gz`) theo định dạng byte đặc biệt của tác giả Yann LeCun:
- `train-images-idx3-ubyte.gz`: Chứa dữ liệu ảnh. 16 byte đầu là tiêu đề (Magic number: 4 byte, Số ảnh: 4 byte, Số dòng: 4 byte, Số cột: 4 byte). Từ byte thứ 16 trở đi là dữ liệu thô của từng pixel.
- `train-labels-idx1-ubyte.gz`: Chứa nhãn. 8 byte đầu là tiêu đề (Magic number: 4 byte, Số nhãn: 4 byte). Từ byte thứ 8 trở đi là các byte nhãn (giá trị từ 0 đến 9).

Đoạn code đọc trực tiếp bằng thư viện `gzip` và `numpy.frombuffer`:
```python
import gzip
import numpy as np

def load_mnist_images(filename, num_images):
    with gzip.open(filename, 'rb') as f:
        f.read(16)  # Bỏ qua 16 byte header
        buf = f.read(28 * 28 * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        # Chuẩn hóa giá trị pixel về đoạn [0, 1] và duỗi phẳng thành vector 784 chiều
        data = data.reshape(num_images, 784) / 255.0
        return data

def load_mnist_labels(filename, num_images):
    with gzip.open(filename, 'rb') as f:
        f.read(8)   # Bỏ qua 8 byte header
        buf = f.read(num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return labels
```

---

## 5. GIẢI PHẪU MÃ NGUỒN TỪNG DÒNG (LINE-BY-LINE CODE BREAKDOWN)

Dưới đây là mã nguồn hoàn chỉnh tự xây dựng thuật toán Softmax Regression bằng NumPy từ tài liệu bài thực hành Tuần 4:

```python
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

# 1. Chuyển đổi nhãn thành ma trận One-hot
def convert_labels(y, C):
    N = len(y)
    Y = sparse.coo_matrix((np.ones_like(y), (y, np.arange(N))), shape=(C, N)).toarray()
    return Y

# 2. Hàm Softmax ổn định số học
def softmax_stable(Z):
    # Z có shape (C, N)
    e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    A = e_Z / e_Z.sum(axis=0)
    return A

# 3. Thuật toán huấn luyện Stochastic Gradient Descent
def softmax_regression(X, y, W_init, eta, tol=1e-4, max_count=10000):
    W = [W_init]
    C = W_init.shape[1]
    Y = convert_labels(y, C)   # Shape (C, N)
    N = X.shape[1]              # Số lượng mẫu
    d = X.shape[0]              # Số chiều đặc trưng (đã có dòng số 1)
    count = 0
    check_w_after = 20
    
    while count < max_count:
        # Xáo trộn thứ tự các mẫu để đảm bảo tính ngẫu nhiên (Stochastic)
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)      # Vector cột (d, 1)
            yi = Y[:, i].reshape(C, 1)      # Vector One-hot (C, 1)
            
            # Dự đoán xác suất cho mẫu xi: A = softmax(W^T * xi)
            ai = softmax_stable(np.dot(W[-1].T, xi))
            
            # Cập nhật trọng số: W_new = W + eta * xi * (yi - ai)^T
            W_new = W[-1] + eta * xi.dot((yi - ai).T)
            count += 1
            
            # Tiêu chuẩn dừng: Sau mỗi 20 bước, kiểm tra sự thay đổi của W
            if count % check_w_after == 0:
                if np.linalg.norm(W_new - W[-check_w_after]) < tol:
                    return W
            W.append(W_new)
            
    return W

# 4. Dự đoán nhãn cho mẫu mới
def pred(W, X):
    # W là ma trận nghiệm cuối cùng (d, C)
    # X có kích thước (d, N)
    A = softmax_stable(W.T.dot(X))
    # Nhãn dự đoán là chỉ số có xác suất lớn nhất dọc theo từng cột
    return np.argmax(A, axis=0)
```

---

### Kỹ thuật vẽ vùng ranh giới phân chia đa lớp bằng `plt.contourf`
```python
# Tạo lưới tọa độ 2D từ x_min đến x_max, y_min đến y_max
xm = np.arange(-2, 11, 0.025)
ym = np.arange(-3, 10, 0.025)
xx, yy = np.meshgrid(xm, ym)

# Ghép tọa độ lưới thành ma trận dữ liệu kiểm thử XX có shape (3, số_điểm_lưới)
xx1 = xx.ravel().reshape(1, xx.size)
yy1 = yy.ravel().reshape(1, yy.size)
XX = np.concatenate((np.ones((1, xx.size)), xx1, yy1), axis=0)

# Dự đoán nhãn cho toàn bộ các điểm trên lưới
Z = pred(W[-1], XX)
Z = Z.reshape(xx.shape)

# Vẽ các vùng màu thể hiện ranh giới phân chia của 3 lớp
plt.contourf(xx, yy, Z, 200, cmap='jet', alpha=0.1)
plt.show()
```

---

## 6. CHI TIẾT CÁC BÀI TẬP THỰC HÀNH TUẦN 4

### 1. Phân loại 3 cụm Gauss 2D (Ví dụ B.1)
- Dùng hàm tự viết từ NumPy để phân chia 3 cụm điểm và vẽ đồ thị contour 2D trực quan.

### 2. Phân loại hoa Iris (Ví dụ B.2 - Bài tập tự thực hành 1)
- Đọc dữ liệu hoa Iris từ file CSV hoặc `sklearn.datasets.load_iris`.
- Chia tỉ lệ Train:Validation = 4:1 (phân bổ đều mỗi loài 40 train, 10 val).
- Huấn luyện cả bằng code NumPy tự viết và bằng thư viện `sklearn.linear_model.LogisticRegression(multi_class='multinomial', solver='sag')`.
- Đánh giá bằng Accuracy và Confusion Matrix.

### 3. Phân loại 20 chủ đề văn bản tin tức 20 Newsgroups (Ví dụ B.3 - Bài tập 2)
- Hơn 20,000 bài báo được số hóa thành vector túi từ thưa (Sparse matrix) có số chiều rất lớn.
- Khám phá cách sử dụng bộ tối ưu hóa SAG (Stochastic Average Gradient) để xử lý dữ liệu quy mô lớn và thưa.

### 4. Phân loại 7 loại kính (Glass Identification - Ví dụ B.4 - Bài tập 3)
- File dữ liệu `glass.csv` gồm 214 mẫu với 9 thuộc tính hóa lý: Chiết suất (RI), hàm lượng Na, Mg, Al, Si, K, Ca, Ba, Fe.
- Nhãn: $1, 2, 3, 5, 6, 7$ (không có loại 4).
- Bài toán thể hiện việc dữ liệu trong thực tế các nhãn có thể không liên tục, cần mã hóa cẩn thận trước khi huấn luyện Softmax Regression.

### 5. Nhận dạng chữ số viết tay MNIST (Ví dụ 4 - Bài tập lớn)
- Giải nén file nén `.gz` và đọc dữ liệu nhị phân.
- Lấy 5,000 ảnh để huấn luyện nhanh và kiểm tra trên 10,000 ảnh test.
- Đạt độ chính xác trên $90\%$ chỉ với mô hình hồi quy tuyến tính Softmax đơn giản!

---

## 7. BỘ 10 CÂU HỎI VẤN ĐÁP BẢO VỆ BÀI TẬP TUẦN 4 KÈM ĐÁP ÁN CHUẨN

1. **Câu 1: Hàm Softmax khác hàm Sigmoid ở điểm nào? Mối quan hệ giữa chúng là gì?**  
   *Đáp*: 
   - Hàm Sigmoid chỉ dùng cho phân loại nhị phân ($C=2$), nhận vào 1 số thực $z$ và trả về 1 xác suất.
   - Hàm Softmax dùng cho phân loại nhiều lớp ($C \ge 3$), nhận vào 1 vector $C$ điểm số và trả về 1 vector phân phối xác suất có tổng bằng 1.0.
   - Softmax chính là sự tổng quát hóa tự nhiên của Sigmoid. Khi $C = 2$, hàm Softmax hoàn toàn tương đương với hàm Sigmoid.

2. **Câu 2: Kỹ thuật `softmax_stable` là gì và tại sao bắt buộc phải sử dụng trong lập trình thực tế?**  
   *Đáp*: Khi tính $e^z$ với $z$ lớn (ví dụ $z = 1000$), máy tính sẽ bị tràn số dương (`Overflow`) ra `inf` hoặc `NaN`. Kỹ thuật `softmax_stable` khắc phục bằng cách trừ đi giá trị lớn nhất $\max(z)$ trên từng vector trước khi tính số mũ ($e^{z - \max(z)}$). Việc này đảm bảo số mũ lớn nhất luôn là $e^0 = 1$, các số khác $\le 1$, triệt tiêu hoàn toàn nguy cơ tràn số mà không làm thay đổi giá trị xác suất.

3. **Câu 3: Mã hóa One-hot (One-hot Encoding) là gì và tại sao không dùng trực tiếp nhãn số nguyên 0, 1, 2, 3...?**  
   *Đáp*: Mã hóa One-hot biến nhãn số nguyên thành vector nhị phân chỉ có duy nhất vị trí nhãn đúng bằng 1, các vị trí khác bằng 0. Nếu dùng trực tiếp số nguyên, mô hình sẽ ngộ nhận sai lầm về thứ bậc tự nhiên (nghĩ rằng lớp 2 lớn hơn lớp 1, hoặc nhãn 3 gần nhãn 2 hơn nhãn 0), làm sai lệch hoàn toàn quá trình tối ưu hóa.

4. **Câu 4: Hàm mất mát trong Softmax Regression là gì? Viết công thức.**  
   *Đáp*: Là hàm **Categorical Cross-Entropy**: $J(\mathbf{W}) = -\sum_{i=1}^N \sum_{c=0}^{C-1} y_{ic} \ln(a_{ic})$. Vì vector One-hot chỉ có 1 phần tử bằng 1, hàm mất mát thực chất là $-\ln(a_{\text{lớp đúng}})$.

5. **Câu 5: Trong bài toán nhận dạng chữ số viết tay MNIST, một bức ảnh $28 \times 28$ được biểu diễn như thế nào trong mô hình?**  
   *Đáp*: Bức ảnh ma trận $28 \times 28$ được duỗi phẳng (flatten) thành một vector đặc trưng 1 chiều có $d = 784$ phần tử. Mỗi phần tử là cường độ xám của 1 pixel (chuẩn hóa về khoảng $[0, 1]$ bằng cách chia cho 255.0). Sau khi thêm hệ số bias, vector đầu vào có 785 chiều.

6. **Câu 6: Kích thước của ma trận trọng số $\mathbf{W}$ trong bài toán MNIST là bao nhiêu?**  
   *Đáp*: Vì không gian đặc trưng có $d = 784$ chiều (cộng thêm 1 bias là 785) và bài toán có $C = 10$ lớp (từ số 0 đến số 9), ma trận trọng số $\mathbf{W}$ sẽ có kích thước là $785 \times 10$ (chứa tổng cộng 7,850 trọng số cần học).

7. **Câu 7: Kỹ thuật PCA (Principal Component Analysis) trong Ví dụ A.2 dùng để làm gì?**  
   *Đáp*: Dùng để giảm số chiều của dữ liệu từ không gian 4 chiều ban đầu xuống không gian 2 chiều để có thể vẽ và trực quan hóa các cụm loài hoa Iris lên mặt phẳng tọa độ, đồng thời bảo toàn tối đa lượng thông tin (phương sai) của dữ liệu.

8. **Câu 8: Trong hàm tự viết `highest_rank`, nếu có 2 nhãn cùng đạt số phiếu bằng nhau trong $K$ láng giềng thì xử lý thế nào?**  
   *Đáp*: Python sẽ chọn nhãn xuất hiện đầu tiên trong danh sách đếm. Để tránh điều này, trong thực tế người ta có thể kết hợp tính thêm khoảng cách trung bình đến các láng giềng của từng lớp (weighted voting) hoặc giảm $K$ đi 1 để phá vỡ thế hòa.

9. **Câu 9: Thuật toán tối ưu SAG (Stochastic Average Gradient) trong thư viện `scikit-learn` có ưu điểm gì khi huấn luyện Softmax?**  
   *Đáp*: SAG là thuật toán tối ưu hóa kết hợp giữa tốc độ siêu nhanh của Stochastic Gradient Descent (mỗi bước chỉ lấy 1 mẫu) và độ mượt mà ổn định của Batch Gradient Descent (bằng cách lưu trữ giá trị trung bình gradient của các bước trước). Nó đặc biệt hiệu quả trên các tập dữ liệu có số lượng mẫu rất lớn và số chiều thưa như 20 Newsgroups.

10. **Câu 10: Tại sao Softmax Regression còn được gọi là "Mạng nơ-ron nhân tạo 1 lớp" (Single-layer Neural Network)?**  
    *Đáp*: Vì kiến trúc của nó tương đương chính xác với một lớp mạng nơ-ron dày đặc (Dense/Linear layer) nhận đầu vào là vector $\mathbf{x}$, nhân với ma trận trọng số $\mathbf{W}$, rồi đưa trực tiếp qua hàm kích hoạt Softmax để xuất ra xác suất ở tầng đầu ra, không có các tầng ẩn (hidden layers) ở giữa.
