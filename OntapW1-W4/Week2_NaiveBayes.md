# Week 2 — Naive Bayes cho người bắt đầu từ số 0

> Mục tiêu không phải là thuộc công thức. Mục tiêu là bạn nhìn một bài toán và tự trả lời được: **Naive Bayes đang học gì, dự đoán bằng cách nào, và tại sao kết quả đó hợp lý?**

---

## 0. Học file này như thế nào?

Đừng đọc một mạch rồi cố nhớ. Hãy học ba lượt:

1. **Lượt 1 — hiểu câu chuyện:** đọc từ mục 1 đến mục 6, chưa cần code.
2. **Lượt 2 — tính bằng tay:** làm mục 7 và 8 bằng giấy hoặc máy tính.
3. **Lượt 3 — code:** chạy mục 11, sau đó mới đọc phần nâng cao.

Nếu gặp ký hiệu lạ, xem bảng thuật ngữ ở cuối file. Mỗi ký hiệu đều được giải thích trước khi dùng.

Sau Week 2, bạn cần làm được:

- Hiểu Naive Bayes dùng để **phân loại** thứ gì.
- Giải thích dự đoán bằng lời, không cần nhìn công thức.
- Tính được một ví dụ nhỏ bằng tay.
- Biết lúc nào dùng Gaussian, Multinomial hoặc Bernoulli Naive Bayes.
- Chạy được bài Iris trong repo và hiểu từng dòng chính.

---

## 1. Trước hết: Machine Learning đang làm gì?

Hãy tưởng tượng bạn có 100 email. Con người đã đọc và gắn nhãn:

- 30 email là `spam`.
- 70 email là `không spam`.

Mỗi email còn có các thông tin như:

- Có từ “khuyến mãi” không?
- Có đường link không?
- Từ “miễn phí” xuất hiện bao nhiêu lần?

Ta đưa 100 email này cho máy. Máy quan sát xem những dấu hiệu nào thường xuất hiện ở spam và những dấu hiệu nào thường xuất hiện ở email bình thường.

Khi có email thứ 101 chưa biết nhãn, máy dùng điều đã học để đoán. Đó là một bài toán **phân loại**.

### 1.1 Bốn từ bắt buộc phải hiểu

| Từ | Nghĩa dễ hiểu | Ví dụ email |
|---|---|---|
| Mẫu dữ liệu | một đối tượng cần quan sát | một email |
| Feature/đặc trưng | thông tin mô tả mẫu | số lần có từ “miễn phí” |
| Label/nhãn | đáp án đúng cần dự đoán | spam hoặc không spam |
| Model/mô hình | quy tắc máy học từ dữ liệu | Naive Bayes |

Trong code thường dùng:

- `X`: bảng feature.
- `y`: cột nhãn đúng.
- `X_train`, `y_train`: phần dùng để học.
- `X_test`, `y_test`: phần giấu đi để kiểm tra.
- `y_pred`: nhãn mô hình dự đoán.

```text
X                                      y
+--------------+----------+            +----------+
| so_tu_free   | co_link  |            | nhan     |
+--------------+----------+            +----------+
| 3            | 1        |            | spam     |
| 0            | 0        |            | thuong   |
| 1            | 1        |            | spam     |
+--------------+----------+            +----------+
```

`X` chứa bằng chứng; `y` chứa đáp án.

---

## 2. Naive Bayes dùng để làm gì?

Naive Bayes dùng để chọn **một nhãn trong nhiều nhãn có thể có**, dựa trên các dấu hiệu của mẫu mới.

Các ví dụ trong repo:

| Bài toán | Feature | Nhãn cần đoán |
|---|---|---|
| Iris | chiều dài/rộng đài hoa và cánh hoa | loại hoa |
| Ung thư vú | các số đo của tế bào | lành tính/ác tính |
| Email | số lần từng từ xuất hiện | spam/không spam |
| Triệu chứng | có/không từng triệu chứng | loại bệnh |
| Đơn hàng | thông tin món/đơn hàng | loại ẩm thực |

Naive Bayes hữu ích khi cần một mô hình đơn giản, rất nhanh, làm việc được với dữ liệu ít hoặc nhiều feature — nhất là văn bản. Nó cũng là baseline tốt để kiểm tra xem mô hình phức tạp hơn có thật sự đáng dùng không.

Naive Bayes **không** dự đoán trực tiếp một số liên tục như giá nhà. Đó là bài toán hồi quy của Week 1.

---

## 3. Ý tưởng cốt lõi — chưa cần công thức

Giả sử email mới chứa từ “khuyến mãi”, một đường link và ba lần từ “miễn phí”. Naive Bayes hỏi:

1. Các dấu hiệu này **giống nhóm spam đến mức nào**?
2. Các dấu hiệu này **giống nhóm không spam đến mức nào**?

Mỗi lớp nhận một điểm:

```text
điểm của lớp
= mức phổ biến ban đầu của lớp
× mức phù hợp của dấu hiệu 1
× mức phù hợp của dấu hiệu 2
× ...
```

Nếu điểm spam lớn hơn điểm không spam, mô hình đoán spam. Đây là toàn bộ tinh thần của Naive Bayes.

---

## 4. Xác suất từ số 0

### 4.1 Xác suất là tỷ lệ

Nếu có 100 email và 30 email là spam:

```text
xác suất gặp spam = số email spam / tổng email
                  = 30 / 100
                  = 0.30 = 30%
```

Ký hiệu `P(Spam)=0.30`, đọc là “xác suất spam bằng 0.30”.

### 4.2 Xác suất có điều kiện

Trong 30 email spam, có 24 email chứa từ “khuyến mãi”:

```text
xác suất có từ khuyến mãi, khi biết email là spam
= 24 / 30
= 0.80
```

Ký hiệu:

$$
P(KhuyenMai\mid Spam)=0.80
$$

Dấu `|` đọc là **“khi đã biết”**.

Hai xác suất sau không giống nhau:

- `P(KhuyenMai | Spam)`: đã biết email spam, hỏi khả năng có từ khuyến mãi.
- `P(Spam | KhuyenMai)`: đã thấy từ khuyến mãi, hỏi khả năng email là spam.

Naive Bayes giúp ta đi từ loại thứ nhất — dễ đếm trong dữ liệu train — sang loại thứ hai — thứ ta cần để dự đoán.

### 4.3 Vì sao không thể chỉ nhìn một dấu hiệu?

“Khuyến mãi” có thể xuất hiện trong cả email spam lẫn email hợp pháp. Machine Learning không hỏi “dấu hiệu này có luôn đúng không?”, mà hỏi “dấu hiệu này làm lớp nào hợp lý hơn?”.

---

## 5. Bayes là cập nhật niềm tin

Trước khi đọc email mới, ta biết 30% email là spam. Sau khi thấy từ “khuyến mãi”, ta cập nhật niềm tin vì từ này xuất hiện nhiều hơn trong spam.

| Tên | Ý nghĩa | Ví dụ |
|---|---|---|
| Prior | niềm tin trước khi thấy mẫu mới | `P(Spam)=0.30` |
| Likelihood | bằng chứng hợp với lớp đến đâu | `P(KM|Spam)=0.80` |
| Posterior | niềm tin sau khi thấy bằng chứng | `P(Spam|KM)` |

Công thức Bayes:

$$
P(Spam\mid KM)=
\frac{P(KM\mid Spam)P(Spam)}{P(KM)}
$$

Đọc bằng lời:

```text
xác suất spam sau khi thấy “khuyến mãi”
= mức “khuyến mãi” phù hợp với spam
× mức spam phổ biến ban đầu
÷ mức “khuyến mãi” phổ biến nói chung
```

### 5.1 Ví dụ đầy đủ bằng 100 email

- 30 spam; 24 trong số đó có “khuyến mãi”.
- 70 email thường; 14 trong số đó có “khuyến mãi”.
- Tổng cộng `24 + 14 = 38` email chứa từ này.

Nếu chọn ngẫu nhiên một email trong 38 email đó:

$$
P(Spam\mid KM)=\frac{24}{38}\approx0.632
$$

Bayes cho cùng kết quả:

$$
P(KM\mid Spam)=24/30=0.8
$$

$$
P(Spam)=30/100=0.3,\qquad P(KM)=38/100=0.38
$$

$$
P(Spam\mid KM)=\frac{0.8\times0.3}{0.38}\approx0.632
$$

Niềm tin spam tăng từ 30% lên khoảng 63.2% sau khi thấy bằng chứng.

---

## 6. Tại sao có chữ “Naive” — ngây thơ?

Một email có nhiều dấu hiệu. Tính chính xác xác suất của mọi tổ hợp là rất khó. Có 20 feature nhị phân thì đã có hơn một triệu tổ hợp.

Naive Bayes đơn giản hóa bằng giả sử:

> Khi đã biết email là spam hay không spam, các feature không ảnh hưởng lẫn nhau.

Nhờ vậy:

$$
P(KM,Link,Free\mid Spam)
$$

được xấp xỉ thành:

$$
P(KM\mid Spam)
\times P(Link\mid Spam)
\times P(Free\mid Spam)
$$

Giả sử này “ngây thơ” vì trong đời thật các từ có liên hệ. Nhưng nó làm bài toán rất đơn giản và thường vẫn phân loại tốt.

Naive Bayes không nói feature độc lập hoàn toàn; nó giả sử độc lập **sau khi đã biết lớp**.

---

## 7. Tự dự đoán spam bằng tay

Giả sử dữ liệu train có 10 email:

| Nhóm | Số email | Có “khuyến mãi” | Có link |
|---|---:|---:|---:|
| Spam | 4 | 3 | 4 |
| Thường | 6 | 1 | 2 |

Email mới vừa có “khuyến mãi”, vừa có link.

### Bước 1 — Prior

$$
P(Spam)=4/10=0.4,\qquad P(Thuong)=6/10=0.6
$$

### Bước 2 — Likelihood

$$
P(KM\mid Spam)=3/4=0.75
$$

$$
P(Link\mid Spam)=4/4=1.0
$$

$$
P(KM\mid Thuong)=1/6\approx0.167
$$

$$
P(Link\mid Thuong)=2/6\approx0.333
$$

### Bước 3 — Tính điểm

$$
Score(Spam)=0.4\times0.75\times1.0=0.30
$$

$$
Score(Thuong)=0.6\times0.167\times0.333\approx0.033
$$

### Bước 4 — So sánh

`0.30 > 0.033`, nên dự đoán **spam**.

Ta chưa cần chia cho xác suất chung của bằng chứng vì mẫu số giống nhau cho mọi lớp. Khi chỉ cần chọn lớp, so sánh score là đủ.

```text
Với mỗi lớp:
    lấy prior
    nhân likelihood của từng feature
Chọn lớp có score lớn nhất
```

---

## 8. Xác suất bằng 0 và Laplace smoothing

Giả sử trong 4 email spam train chưa từng có từ “trúng thưởng”:

$$
P(TrungThuong\mid Spam)=0/4=0
$$

Nếu nhân với 0, toàn bộ score spam bằng 0, dù các dấu hiệu khác rất mạnh. Dữ liệu train nhỏ không nên khiến mô hình kết luận “không bao giờ”.

Laplace smoothing giả vờ mỗi khả năng đã được nhìn thấy thêm một lần. Với feature nhị phân có hai khả năng `có/không`:

$$
P(CoTrungThuong\mid Spam)=\frac{0+1}{4+2}=\frac{1}{6}
$$

Trong MultinomialNB với `d` từ trong từ điển:

$$
P(tu_j\mid c)=\frac{count_{cj}+\alpha}{total_c+\alpha d}
$$

- `alpha=1`: Laplace smoothing.
- `alpha` lớn kéo xác suất gần đều hơn.
- Smoothing giải quyết **zero probability**, không tự sửa dữ liệu xấu.

---

## 9. Ba loại Naive Bayes — chọn theo kiểu feature

### 9.1 GaussianNB — số đo liên tục

Ví dụ: chiều dài cánh hoa `1.4 cm`, tuổi `37`, huyết áp `125.5`. Mô hình học trung bình và độ phân tán của từng feature trong từng lớp.

Dùng cho Iris và Breast Cancer trong repo.

### 9.2 MultinomialNB — số lần xuất hiện

```text
từ điển: [free, sale, meeting]
email A: [3,    1,    0]
email B: [0,    0,    2]
```

Số `3` nghĩa là `free` xuất hiện ba lần. Feature phải không âm. Dùng cho văn bản và spam.

### 9.3 BernoulliNB — chỉ có hoặc không

```text
từ điển: [free, sale, meeting]
email A: [1,    1,    0]
email B: [0,    0,    1]
```

`1` là có, `0` là không. Dùng cho triệu chứng hoặc văn bản khi sự hiện diện quan trọng hơn số lần.

### 9.4 Bảng quyết định nhanh

| Feature là gì? | Chọn |
|---|---|
| Số đo liên tục | GaussianNB |
| Số đếm/tần suất không âm | MultinomialNB |
| Có/không, 0/1 | BernoulliNB |

“Tuổi” là số nguyên nhưng là đại lượng đo về ý nghĩa; không phải cứ số nguyên là Multinomial.

---

## 10. Gaussian Naive Bayes hoạt động thế nào?

Giả sử chỉ xét chiều dài cánh hoa:

| Loài | Trung bình | Độ lệch chuẩn |
|---|---:|---:|
| Setosa | 1.5 | 0.2 |
| Versicolor | 4.3 | 0.5 |
| Virginica | 5.5 | 0.6 |

Hoa mới có cánh dài `1.6`. Nó gần trung bình Setosa và xa hai lớp còn lại, nên đây là bằng chứng mạnh cho Setosa.

GaussianNB dùng đường cong hình chuông để đo “giá trị này bình thường đến đâu trong lớp”:

$$
P(x\mid c)=\frac{1}{\sqrt{2\pi\sigma_c^2}}
\exp\left(-\frac{(x-\mu_c)^2}{2\sigma_c^2}\right)
$$

- `x`: giá trị mẫu mới.
- `c`: lớp đang xét.
- `mu_c`: trung bình trong lớp.
- `sigma_c`: độ lệch chuẩn trong lớp.
- `P(x|c)`: mức phù hợp của `x` với lớp `c`.

Không cần tính công thức này bằng tay trong mọi bài. Cần hiểu: càng gần vùng thường gặp của lớp thì likelihood thường càng lớn.

---

## 11. Bài Iris đầu tiên — giải thích từng bước

Chạy từ thư mục `D:\MachineLearning`:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Đọc dữ liệu
df = pd.read_csv("Week2/iris.csv")
df.columns = [name.replace(".", "_") for name in df.columns]

# 2. Tách feature X và nhãn y
feature_columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]
X = df[feature_columns]
y = df["variety"]

# 3. Chia dữ liệu để học và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Tạo và huấn luyện mô hình
model = GaussianNB()
model.fit(X_train, y_train)

# 5. Dự đoán mẫu mô hình chưa học
y_pred = model.predict(X_test)

# 6. So sánh với đáp án thật
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
```

### 11.1 Trước khi học, nhìn dữ liệu

```python
print(df.head())
print(df.shape)
print(df.dtypes)
print(df.isna().sum())
print(X.shape)  # dự kiến (150, 4)
print(y.shape)  # dự kiến (150,)
```

Mỗi hàng `X` là một bông hoa, bốn cột là bốn số đo, phần tử cùng hàng trong `y` là loài hoa.

### 11.2 Tại sao chia train/test?

- Train: mô hình nhìn `X_train` và đáp án `y_train` để học.
- Test: mô hình nhận `X_test`, dự đoán rồi mới so với `y_test`.

Đánh giá trên train giống như cho học sinh làm lại đúng đề đã học. Test mới kiểm tra khả năng áp dụng vào câu chưa thấy.

`random_state=42` giúp lần sau chia giống lần trước. `stratify=y` giữ tỷ lệ ba loài gần giống nhau ở train và test.

### 11.3 `fit` học những gì?

```python
print("Các lớp:", model.classes_)
print("Prior:", model.class_prior_)
print("Mean mỗi lớp/feature:\n", model.theta_)
print("Variance mỗi lớp/feature:\n", model.var_)
```

`fit` chủ yếu tính prior, mean và variance cho mỗi lớp/feature; nó không phải phép màu.

### 11.4 `predict` làm gì?

Với từng hoa test, mô hình tính score cho ba loài rồi chọn score lớn nhất.

```python
print("Thứ tự lớp:", model.classes_)
print("Xác suất ba mẫu đầu:\n", model.predict_proba(X_test.iloc[:3]))
```

Mỗi hàng xác suất có tổng gần bằng 1. Thứ tự các cột khớp với `model.classes_`.

---

## 12. Accuracy và confusion matrix cho người mới

Nếu đoán đúng 28 trong 30 hoa:

$$
Accuracy=28/30\approx93.3\%
$$

Confusion matrix ví dụ:

```text
              dự đoán
             A   B   C
thực tế A   10   0   0
thực tế B    0   9   1
thực tế C    0   1   9
```

Đọc từng hàng: 10 mẫu A đúng hết; B có 9 đúng, 1 nhầm C; C có 9 đúng, 1 nhầm B. Đường chéo là đúng, ngoài đường chéo là lỗi.

Với lớp “ác tính”:

- **Precision:** Trong các ca máy báo ác tính, bao nhiêu ca thật sự ác tính?
- **Recall:** Trong tất cả ca ác tính thật, máy tìm được bao nhiêu?

Trong y tế, false negative — bệnh thật nhưng máy báo không bệnh — thường rất nguy hiểm, nên Recall lớp bệnh quan trọng.

$$
Precision=\frac{TP}{TP+FP},\qquad Recall=\frac{TP}{TP+FN}
$$

Chưa cần thuộc công thức; hãy nhớ hai câu hỏi bằng lời.

---

## 13. GaussianNB từ đầu — đọc sau khi hiểu phần trên

```python
import numpy as np

class GaussianNBScratch:
    def fit(self, X, y):
        self.classes = np.unique(y)

        # Mỗi hàng là mean của một lớp
        self.means = np.array([
            X[y == c].mean(axis=0) for c in self.classes
        ])

        # Số nhỏ 1e-9 ngăn chia cho 0
        self.variances = np.array([
            X[y == c].var(axis=0) for c in self.classes
        ]) + 1e-9

        # Tỷ lệ mẫu thuộc từng lớp
        self.log_priors = np.log([
            np.mean(y == c) for c in self.classes
        ])
        return self

    def predict(self, X):
        scores = []
        for mean, variance, log_prior in zip(
            self.means, self.variances, self.log_priors
        ):
            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * variance)
                + (X - mean) ** 2 / variance,
                axis=1,
            )
            scores.append(log_prior + log_likelihood)

        score_table = np.column_stack(scores)
        best_class_index = np.argmax(score_table, axis=1)
        return self.classes[best_class_index]
```

Nếu có 120 mẫu train, 4 feature, 3 lớp:

| Biến | Shape | Ý nghĩa |
|---|---|---|
| `X` | `(120, 4)` | 120 hoa, mỗi hoa 4 số đo |
| `y` | `(120,)` | 120 nhãn |
| `means` | `(3, 4)` | mean của 4 feature trong 3 lớp |
| `variances` | `(3, 4)` | variance tương ứng |
| `score_table` cho 30 mẫu test | `(30, 3)` | score của 30 mẫu với 3 lớp |

Hiểu shape trước khi debug công thức.

---

## 14. Vì sao code thật dùng log?

Nếu có 1.000 feature và mỗi likelihood khoảng `0.01`, tích `0.01^1000` nhỏ đến mức máy có thể làm tròn thành 0. Đây là **underflow**.

Ta dùng:

$$
\log(a\times b)=\log a+\log b
$$

Thay vì nhân nhiều số nhỏ, ta cộng log:

$$
\log Score(c)=\log P(c)+\sum_j\log P(x_j\mid c)
$$

Lớp có score lớn nhất vẫn có log-score lớn nhất. Log-trick chỉ giúp tính an toàn; nó không đổi ý tưởng thuật toán.

---

## 15. Multinomial và Bernoulli bằng một email

Email: `FREE FREE FREE, meeting tomorrow`.

```text
từ điển:                  [free, meeting, tomorrow]
Multinomial representation [3,    1,       1]
Bernoulli representation   [1,    1,       1]
```

Multinomial biết `free` lặp ba lần; Bernoulli chỉ biết nó có xuất hiện.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

train_texts = [
    "khuyen mai mien phi",
    "mien phi nhan qua",
    "hop nhom ngay mai",
    "lich hoc may",
]
train_labels = ["spam", "spam", "thuong", "thuong"]

model = make_pipeline(
    CountVectorizer(),
    MultinomialNB(alpha=1.0),
)
model.fit(train_texts, train_labels)
print(model.predict(["mien phi khuyen mai"]))
```

Pipeline làm hai việc: `CountVectorizer` biến câu thành vector đếm; `MultinomialNB` học xác suất từ theo lớp. Không fit vectorizer trên test vì như vậy preprocessing đã nhìn dữ liệu test.

---

## 16. Naive Bayes thật sự học gì?

Nó không học luật cứng kiểu “nếu cánh hoa ngắn hơn 2 thì Setosa”. Nó học:

- Lớp nào phổ biến bao nhiêu?
- Trong mỗi lớp, từng feature thường trông như thế nào?

Khi dự đoán, nó hỏi:

```text
nếu mẫu này thuộc A thì các feature có hợp lý không?
nếu thuộc B thì sao?
nếu thuộc C thì sao?
```

Rồi chọn lời giải thích hợp lý nhất. Vì mô tả mỗi lớp sinh feature thế nào, Naive Bayes được gọi là mô hình **generative**.

---

## 17. Khi nào Naive Bayes có thể sai?

### Feature phụ thuộc mạnh

Hai feature gần như cùng nói một điều có thể khiến mô hình đếm cùng bằng chứng hai lần và quá tự tin.

### Train không đại diện

Nếu train chỉ chứa một kiểu spam, mô hình không thể học những kiểu chưa xuất hiện.

### Chọn sai biến thể

- Đưa số đo liên tục vào Bernoulli mà không biến đổi.
- Đưa giá trị âm vào Multinomial.
- Dùng Gaussian cho dữ liệu đếm rất lệch.

### Tin tuyệt đối vào `predict_proba`

Do giả định độc lập, Naive Bayes có thể cho xác suất gần 0 hoặc 1 quá mức. Dự đoán lớp có thể tốt nhưng xác suất chưa chắc được hiệu chỉnh tốt.

---

## 18. Học các bài trong repo theo thứ tự nào?

### Bài 1 — Iris: dễ nhất

`Week2/work/bai1_iris_gaussian_nb.py`

Hiểu `X/y`, train/test, mean/variance/prior và so scratch với sklearn.

### Bài 2 — Breast Cancer

`Week2/work/bai2_cancer_gaussian_nb.py`

Học missing value, Recall và hậu quả false negative trong y tế.

### Bài 3 — Văn bản Bắc/Nam

`Week2/work/bai3_text_b_n_nb.py`

Học cách biến câu chữ thành vector, phân biệt Multinomial/Bernoulli và tính score tài liệu mới.

### Bài 4–6 — sau khi hiểu ba bài trên

- Đơn hàng: dữ liệu danh mục và nhiều lớp.
- Triệu chứng: feature nhị phân.
- Spam: ma trận từ lớn và ảnh hưởng kích thước train.

Nếu chưa giải thích được Iris bằng lời, chưa nên sang spam.

---

## 19. Lộ trình học trong 5 buổi

### Buổi 1 — hiểu bài toán

Đọc mục 1–6. Tự nghĩ ba bài toán phân loại và chỉ ra mẫu, feature, label.

### Buổi 2 — tính tay

Làm lại bảng spam mục 7 không nhìn lời giải. Đổi số rồi tính lại. Giải thích prior, likelihood, posterior bằng lời.

### Buổi 3 — Iris với sklearn

Chạy mục 11 từng đoạn. In `shape`, `classes_`, `class_prior_`, `theta_`, `var_`, rồi xem `predict_proba` của một mẫu.

### Buổi 4 — scratch

Đọc mục 13, ghi shape sau mỗi dòng quan trọng, so dự đoán scratch với sklearn.

### Buổi 5 — văn bản

Chạy mục 15, in vocabulary và ma trận đếm, so MultinomialNB với BernoulliNB.

---

## 20. Bài tập từ dễ đến khó

### Mức 1 — nhận biết

Chọn mô hình cho: bốn số đo Iris; số lần mỗi từ trong email; 30 triệu chứng có/không. Đáp án lần lượt là Gaussian, Multinomial, Bernoulli.

### Mức 2 — tính tay

| Lớp | Số mẫu | Có A | Có B |
|---|---:|---:|---:|
| 0 | 5 | 1 | 2 |
| 1 | 5 | 4 | 3 |

Tính score cho mẫu có cả A và B, sau đó thử Laplace smoothing.

### Mức 3 — code có hướng dẫn

Chạy Iris, in một hàng `X_test`, in xác suất ba lớp và kiểm tra lớp có xác suất lớn nhất đúng bằng `predict`.

### Mức 4 — hiểu lỗi

Lấy các mẫu Iris đoán sai, so từng feature với mean ba lớp và viết 3–5 câu giải thích vì sao dễ nhầm.

### Mức 5 — tự làm

Làm Breast Cancer và báo cáo feature/label, missing value, confusion matrix, Recall ác tính và một giới hạn của thí nghiệm.

---

## 21. Lỗi người mới thường mắc

- Học thuộc Bayes nhưng không biết `X` và `y` là gì.
- Nhầm `P(A|B)` với `P(B|A)`.
- Không chia train/test mà đánh giá trên train.
- Chọn biến thể theo tên dataset thay vì kiểu feature.
- Quên smoothing khiến score bằng 0.
- Nhân nhiều xác suất nhỏ trực tiếp thay vì dùng log.
- Chỉ nhìn Accuracy trong bài y tế.
- Fit vectorizer hoặc preprocessing bằng cả test.
- Nghĩ “naive” nghĩa là thuật toán dở.

---

## 22. Câu hỏi tự kiểm tra

1. Naive Bayes giải quyết loại bài toán gì?
2. Trong bài spam, đâu là mẫu, feature và label?
3. Prior khác likelihood thế nào?
4. Dấu `|` trong `P(A|B)` đọc là gì?
5. Tại sao `P(Spam|KM)` không bằng `P(KM|Spam)`?
6. Chữ “naive” nói đến giả định nào?
7. Vì sao bỏ mẫu số khi chỉ chọn lớp?
8. Smoothing giải quyết vấn đề gì?
9. Dữ liệu nào dùng GaussianNB?
10. MultinomialNB khác BernoulliNB ở đâu?
11. `fit` của GaussianNB học những gì?
12. Vì sao đánh giá trên test?
13. Accuracy cao có đủ trong bài ung thư không?
14. Log-trick dùng để làm gì?
15. Vì sao Naive Bayes có thể quá tự tin?

Nếu chưa trả lời được câu 1–8 mà không nhìn tài liệu, quay lại mục 3–8 trước khi code tiếp.

---

## 23. Tóm tắt Week 2 trong một trang

```text
BÀI TOÁN
    Chọn một nhãn cho mẫu mới.

DỮ LIỆU
    X = bằng chứng/feature
    y = nhãn đúng

HUẤN LUYỆN
    Học lớp nào phổ biến bao nhiêu.
    Học feature thường trông thế nào trong mỗi lớp.

DỰ ĐOÁN
    Với từng lớp:
        score = prior × likelihood của các feature
    Chọn lớp có score lớn nhất.

NAIVE
    Tạm coi feature độc lập khi đã biết lớp.

BA BIẾN THỂ
    Gaussian    → số đo liên tục
    Multinomial → số đếm không âm
    Bernoulli   → có/không, 0/1

HAI KỸ THUẬT
    Smoothing → tránh xác suất 0
    Log       → tránh số quá nhỏ

ĐÁNH GIÁ
    Test set + confusion matrix.
    Chọn Precision/Recall theo hậu quả của lỗi.
```

---

## 24. Từ điển thuật ngữ

| Thuật ngữ | Nghĩa ngắn gọn |
|---|---|
| Classification | chọn một lớp/nhãn |
| Feature | thông tin dùng để dự đoán |
| Label | đáp án cần dự đoán |
| Train set | dữ liệu cho mô hình học |
| Test set | dữ liệu chưa thấy để kiểm tra |
| Prior | xác suất lớp trước bằng chứng mới |
| Likelihood | bằng chứng hợp với lớp đến đâu |
| Posterior | xác suất lớp sau khi thấy bằng chứng |
| Conditional probability | xác suất khi đã biết điều kiện |
| Gaussian | phân phối hình chuông cho số đo liên tục |
| Mean | giá trị trung bình |
| Variance | mức phân tán quanh trung bình |
| Smoothing | thêm bằng chứng giả nhỏ để tránh xác suất 0 |
| Underflow | số quá nhỏ bị máy làm tròn thành 0 |
| Confusion matrix | bảng đếm đúng/sai theo từng lớp |
| Precision | cảnh báo dương đáng tin đến đâu |
| Recall | tìm được bao nhiêu mẫu dương thật |

---

## 25. Checklist hoàn thành Week 2

- [ ] Tôi giải thích được Naive Bayes dùng để làm gì mà không dùng công thức.
- [ ] Tôi phân biệt được mẫu, feature, label, train và test.
- [ ] Tôi hiểu dấu `|` trong xác suất có điều kiện.
- [ ] Tôi tự tính được ví dụ spam hai feature.
- [ ] Tôi giải thích được prior, likelihood và posterior.
- [ ] Tôi hiểu chữ “naive” nói đến giả định nào.
- [ ] Tôi chọn đúng Gaussian, Multinomial hoặc Bernoulli theo kiểu feature.
- [ ] Tôi hiểu smoothing và log-trick giải quyết hai lỗi khác nhau.
- [ ] Tôi chạy được Iris và hiểu `fit`, `predict`, `predict_proba`.
- [ ] Tôi đọc được confusion matrix bằng lời.
- [ ] Tôi hoàn thành `Week2/work/bai1_iris_gaussian_nb.py` trước khi sang bài khó hơn.

