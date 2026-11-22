"""
GIAI ĐOẠN 1: Chuẩn bị dữ liệu và xây dựng mô hình cơ sở
Bài toán: Dự đoán hành khách Titanic có sống sót hay không (Phân loại nhị phân)
Mô hình cơ sở: Logistic Regression
Chạy: python phase1_baseline.py
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay, mean_squared_error)
from common import load_data, add_features, RANDOM_STATE

# ---------------------------------------------------------------
# BƯỚC 1: Chọn và khám phá dữ liệu
# ---------------------------------------------------------------
df = load_data()
print("=" * 60, "\n1. KHÁM PHÁ DỮ LIỆU\n", "=" * 60)
print("Shape:", df.shape)
print("\n--- info() ---")
df.info()
print("\n--- describe() ---")
print(df.describe().round(2))
print("\n--- Số giá trị khuyết mỗi cột ---")
print(df.isna().sum()[df.isna().sum() > 0])
print("\n--- Tỉ lệ sống sót ---")
print(df["Survived"].value_counts(normalize=True).round(3))

# Trực quan hóa EDA
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
sns.countplot(x="Survived", data=df, ax=ax[0, 0]); ax[0, 0].set_title("Phân bố nhãn Survived")
sns.barplot(x="Sex", y="Survived", data=df, ax=ax[0, 1]); ax[0, 1].set_title("Tỉ lệ sống theo giới tính")
sns.barplot(x="Pclass", y="Survived", data=df, ax=ax[0, 2]); ax[0, 2].set_title("Tỉ lệ sống theo hạng vé")
sns.histplot(data=df, x="Age", hue="Survived", bins=30, ax=ax[1, 0], kde=True); ax[1, 0].set_title("Phân bố tuổi")
sns.boxplot(x="Pclass", y="Fare", data=df, ax=ax[1, 1]); ax[1, 1].set_title("Fare theo hạng vé (thấy outlier)")
sns.heatmap(df.isna().T, cbar=False, ax=ax[1, 2]); ax[1, 2].set_title("Bản đồ giá trị khuyết")
plt.tight_layout(); plt.savefig("figures/01_eda.png", dpi=120); plt.close()

# ---------------------------------------------------------------
# BƯỚC 2: Làm sạch dữ liệu
# ---------------------------------------------------------------
print("\n" + "=" * 60, "\n2. LÀM SẠCH DỮ LIỆU\n", "=" * 60)
data = add_features(df)  # tạo Title, FamilySize, IsAlone, HasCabin; bỏ cột vô ích

# Chia train/test TRƯỚC khi tính thống kê (median, mode, ngưỡng outlier) để tránh rò rỉ dữ liệu
X = data.drop(columns="Survived")
y = data["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)  # 80% train / 20% test
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# 2a. Xử lý NaN: Age -> median (theo train), Embarked -> mode (theo train)
age_median = X_train["Age"].median()
emb_mode = X_train["Embarked"].mode()[0]
for part in (X_train, X_test):
    part["Age"] = part["Age"].fillna(age_median)
    part["Embarked"] = part["Embarked"].fillna(emb_mode)

# 2b. Xử lý outlier bằng IQR (cắt - clip - về ngưỡng), áp dụng cho Fare và Age
def iqr_bounds(s):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

for col in ["Fare", "Age"]:
    lo, hi = iqr_bounds(X_train[col])
    n_out = ((X_train[col] < lo) | (X_train[col] > hi)).sum()
    print(f"Outlier {col}: {n_out} giá trị ngoài [{lo:.1f}, {hi:.1f}] -> clip")
    X_train[col] = X_train[col].clip(lo, hi)
    X_test[col] = X_test[col].clip(lo, hi)

# ---------------------------------------------------------------
# BƯỚC 3: Tiền xử lý - mã hóa biến phân loại & chuẩn hóa (Scaling)
# ---------------------------------------------------------------
print("\n" + "=" * 60, "\n3. TIỀN XỬ LÝ\n", "=" * 60)
cat_cols = ["Sex", "Embarked", "Title"]
X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)  # đảm bảo cùng cột

num_cols = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "Pclass"]
scaler = StandardScaler().fit(X_train[num_cols])  # chỉ fit trên train
X_train[num_cols] = scaler.transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])
print("Các đặc trưng sau tiền xử lý:", list(X_train.columns))

# ---------------------------------------------------------------
# BƯỚC 4: Huấn luyện mô hình cơ sở - Logistic Regression
# ---------------------------------------------------------------
print("\n" + "=" * 60, "\n4. MÔ HÌNH CƠ SỞ: LOGISTIC REGRESSION\n", "=" * 60)
base = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
base.fit(X_train, y_train)

# ---------------------------------------------------------------
# BƯỚC 5: Đánh giá ban đầu
# ---------------------------------------------------------------
y_pred = base.predict(X_test)
acc_train = accuracy_score(y_train, base.predict(X_train))
acc_test = accuracy_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)  # với nhãn 0/1, MSE = tỉ lệ dự đoán sai
print(f"Accuracy (train): {acc_train:.4f}")
print(f"Accuracy (test) : {acc_test:.4f}")
print(f"MSE (test)      : {mse:.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=["Chết", "Sống"]))

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Chết", "Sống"],
                                        cmap="Blues", ax=ax[0])
ax[0].set_title(f"Confusion Matrix - Logistic Regression\n(Accuracy test = {acc_test:.3f})")
coef = pd.Series(base.coef_[0], index=X_train.columns).sort_values()
coef.plot(kind="barh", ax=ax[1], color=["#d9534f" if v < 0 else "#5cb85c" for v in coef])
ax[1].set_title("Hệ số Logistic Regression (ảnh hưởng đến khả năng sống sót)")
plt.tight_layout(); plt.savefig("figures/02_baseline_results.png", dpi=120); plt.close()

# Lưu kết quả để Giai đoạn 2 so sánh
pd.Series({"accuracy_test": acc_test, "accuracy_train": acc_train, "mse_test": mse}
          ).to_json("models/baseline_metrics.json")
print("\nĐã lưu biểu đồ vào thư mục figures/")
