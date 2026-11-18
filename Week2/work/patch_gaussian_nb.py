import json
import os

nb_path = os.path.join(os.path.dirname(__file__), "..", "Gaussian_NB.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Fix cell 12: Breast cancer loading and splitting
nb["cells"][12]["source"] = [
    "# Đọc tệp dữ liệu breast-cancer-wisconsin.data\n",
    "col_names = ['id', 'clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity',\n",
    "             'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',\n",
    "             'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']\n",
    "df = pd.read_csv('breast-cancer-wisconsin.data', header=None, names=col_names)\n",
    "\n",
    "# Tiền xử lý: Xử lý 16 giá trị thiếu '?' trong bare_nuclei bằng trung vị (median)\n",
    "df['bare_nuclei'] = pd.to_numeric(df['bare_nuclei'].replace('?', np.nan))\n",
    "df['bare_nuclei'] = df.groupby('class')['bare_nuclei'].transform(lambda s: s.fillna(s.median()))\n",
    "\n",
    "print('5 dòng đầu của dữ liệu sau xử lý:')\n",
    "print(df.head())\n",
    "\n",
    "# Tách dữ liệu Test: 80 mẫu lành tính (class 2) và 40 mẫu ác tính (class 4)\n",
    "test_benign = df[df['class'] == 2].sample(80, random_state=42)\n",
    "test_malignant = df[df['class'] == 4].sample(40, random_state=42)\n",
    "test_df = pd.concat([test_benign, test_malignant])\n",
    "\n",
    "# Dữ liệu Training là phần còn lại (579 mẫu)\n",
    "train_df = df.drop(test_df.index)\n",
    "print(f'Tập Train: {train_df.shape[0]} mẫu, Tập Test: {test_df.shape[0]} mẫu')\n"
]

# Cell 13: Train and evaluate GaussianNB on Cancer
nb["cells"][13]["source"] = [
    "# Huấn luyện mô hình GaussianNB và đánh giá trên tập Test\n",
    "feature_cols = [c for c in col_names if c not in ['id', 'class']]\n",
    "X_train = train_df[feature_cols].values\n",
    "y_train = train_df['class'].values\n",
    "\n",
    "X_test = test_df[feature_cols].values\n",
    "y_test = test_df['class'].values\n",
    "\n",
    "model_cancer = GaussianNB()\n",
    "model_cancer.fit(X_train, y_train)\n",
    "y_pred_cancer = model_cancer.predict(X_test)\n",
    "\n",
    "print(f'Độ chính xác (Accuracy): {accuracy_score(y_test, y_pred_cancer)*100:.2f}%')\n",
    "print(f'Precision (Lớp ác tính 4): {precision_score(y_test, y_pred_cancer, pos_label=4)*100:.2f}%')\n",
    "print(f'Recall (Lớp ác tính 4):    {recall_score(y_test, y_pred_cancer, pos_label=4)*100:.2f}%')\n",
    "print(f'F1-Score (Lớp ác tính 4):  {f1_score(y_test, y_pred_cancer, pos_label=4)*100:.2f}%')\n",
    "print('\\nBáo cáo phân loại chi tiết:\\n', classification_report(y_test, y_pred_cancer, target_names=['Lành tính (2)', 'Ác tính (4)']))\n"
]

# Add cell 14: Bài tập mở rộng Iris 4 đặc trưng nếu chưa có
cell_iris_ext = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# BÀI TẬP MỞ RỘNG: PHÂN LOẠI HOA IRIS VỚI ĐỦ 4 THUỘC TÍNH (TRAIN:TEST = 4:1)\n",
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "X_iris = iris[['1_sepal_length', '2_sepal_width', '3_petal_length', '4_petal_width']].values\n",
        "y_iris = iris['species'].replace({'setosa': 0, 'versicolor': 1, 'virginica': 2}).values\n",
        "\n",
        "X_tr, X_te, y_tr, y_te = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42, stratify=y_iris)\n",
        "gnb_iris = GaussianNB()\n",
        "gnb_iris.fit(X_tr, y_tr)\n",
        "y_pred_iris = gnb_iris.predict(X_te)\n",
        "\n",
        "print('Độ chính xác Iris với 4 đặc trưng:', accuracy_score(y_te, y_pred_iris))\n",
        "print('\\nClassification Report:\\n', classification_report(y_te, y_pred_iris, target_names=['setosa', 'versicolor', 'virginica']))\n"
    ]
}
if len(nb["cells"]) == 14:
    nb["cells"].append(cell_iris_ext)

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Patched {nb_path} successfully!")
