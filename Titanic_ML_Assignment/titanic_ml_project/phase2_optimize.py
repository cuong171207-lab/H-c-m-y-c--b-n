"""
GIAI ĐOẠN 2: Tối ưu hóa và hoàn thiện dự án
- Thử mô hình nâng cao: KNN, Decision Tree, Random Forest
- Xây dựng Pipeline hoàn chỉnh (tiền xử lý + mô hình)
- Tối ưu siêu tham số bằng GridSearchCV + Cross-Validation (5-fold)
- So sánh với mô hình cơ sở, lưu mô hình ra .pkl bằng joblib
Chạy: python phase2_optimize.py   (sau khi đã chạy phase1_baseline.py)
"""
import warnings
warnings.filterwarnings("ignore")
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay, roc_curve, roc_auc_score)
from common import load_data, add_features, log_fare, RANDOM_STATE

# ---------------------------------------------------------------
# 1. Đọc dữ liệu thô và chia Train/Test (80/20) - cùng random_state với Giai đoạn 1
# ---------------------------------------------------------------
df = load_data()
X = df.drop(columns="Survived")   # Pipeline nhận dữ liệu THÔ (còn Name, Cabin, ...)
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ---------------------------------------------------------------
# 2. Xây dựng bộ tiền xử lý (ColumnTransformer)
# ---------------------------------------------------------------
num_cols = ["Age", "SibSp", "Parch", "FamilySize", "Pclass"]
fare_col = ["Fare"]
cat_cols = ["Sex", "Embarked", "Title"]
bin_cols = ["IsAlone", "HasCabin"]

preprocess = ColumnTransformer([
    # Số: điền NaN bằng median, chuẩn hóa
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    # Fare: log1p để giảm ảnh hưởng outlier, rồi chuẩn hóa
    ("fare", Pipeline([("imp", SimpleImputer(strategy="median")),
                       ("log", FunctionTransformer(log_fare)),
                       ("sc", StandardScaler())]), fare_col),
    # Phân loại: điền NaN bằng mode, One-Hot
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ("bin", "passthrough", bin_cols),
])

def make_pipeline(model):
    """Pipeline đầy đủ: tạo đặc trưng -> tiền xử lý -> mô hình."""
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("preprocess", preprocess),
        ("model", model),
    ])

# ---------------------------------------------------------------
# 3. Lưới siêu tham số cho từng mô hình + GridSearchCV (5-fold CV)
# ---------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
candidates = {
    "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                            {"model__C": [0.01, 0.1, 1, 10, 100]}),
    "KNN": (KNeighborsClassifier(),
            {"model__n_neighbors": [3, 5, 7, 9, 11, 15],
             "model__weights": ["uniform", "distance"],
             "model__p": [1, 2]}),
    "Decision Tree": (DecisionTreeClassifier(random_state=RANDOM_STATE),
                      {"model__max_depth": [3, 4, 5, 6, 8, None],
                       "model__min_samples_split": [2, 5, 10],
                       "model__min_samples_leaf": [1, 2, 4]}),
    "Random Forest": (RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
                      {"model__n_estimators": [200],
                       "model__max_depth": [4, 6, 8],
                       "model__min_samples_split": [2, 5],
                       "model__min_samples_leaf": [1, 2, 4],
                       "model__max_features": ["sqrt", 0.5]}),
}

results, best_models = [], {}
for name, (model, grid) in candidates.items():
    # Mô hình mặc định (chưa tối ưu) để so sánh trước/sau tối ưu
    default_cv = cross_val_score(make_pipeline(model), X_train, y_train, cv=cv).mean()
    gs = GridSearchCV(make_pipeline(model), grid, cv=cv, scoring="accuracy", n_jobs=-1)
    gs.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, gs.predict(X_test))
    best_models[name] = gs.best_estimator_
    results.append({"Mô hình": name, "CV acc (mặc định)": default_cv,
                    "CV acc (sau GridSearch)": gs.best_score_, "Test acc": test_acc,
                    "Tham số tốt nhất": {k.replace("model__", ""): v for k, v in gs.best_params_.items()}})
    print(f"\n[{name}] CV mặc định={default_cv:.4f} | CV tốt nhất={gs.best_score_:.4f} | Test={test_acc:.4f}")
    print("   Best params:", gs.best_params_)

res = pd.DataFrame(results)

# ---------------------------------------------------------------
# 4. Chọn mô hình tốt nhất theo CV (KHÔNG chọn theo Test để tránh overfit tập test)
# ---------------------------------------------------------------
best_name = res.loc[res["CV acc (sau GridSearch)"].idxmax(), "Mô hình"]
best_pipe = best_models[best_name]
print("\n" + "=" * 60, f"\nMÔ HÌNH TỐT NHẤT (theo CV): {best_name}\n", "=" * 60)

y_pred = best_pipe.predict(X_test)
y_proba = best_pipe.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
print(f"Accuracy test: {acc:.4f} | ROC-AUC: {auc:.4f}")
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Chết", "Sống"]))

# ---------------------------------------------------------------
# 5. So sánh với mô hình cơ sở (Giai đoạn 1)
# ---------------------------------------------------------------
base = json.load(open("models/baseline_metrics.json"))
print(f"Cơ sở (Logistic Regression GĐ1): Test acc = {base['accuracy_test']:.4f}")
print(f"Mô hình tối ưu ({best_name})   : Test acc = {acc:.4f}  "
      f"(chênh lệch {100 * (acc - base['accuracy_test']):+.2f} điểm %)")

# --- Biểu đồ ---
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
plot_df = res.melt(id_vars="Mô hình", value_vars=["CV acc (mặc định)", "CV acc (sau GridSearch)", "Test acc"],
                   var_name="Loại", value_name="Accuracy")
sns.barplot(data=plot_df, x="Mô hình", y="Accuracy", hue="Loại", ax=ax[0])
ax[0].axhline(base["accuracy_test"], color="red", ls="--", label="Cơ sở GĐ1 (test)")
ax[0].set_ylim(0.7, 0.9); ax[0].legend(fontsize=8); ax[0].set_title("So sánh các mô hình")
ax[0].tick_params(axis="x", rotation=15)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Chết", "Sống"],
                                        cmap="Greens", ax=ax[1])
ax[1].set_title(f"Confusion Matrix - {best_name}\n(Accuracy test = {acc:.3f})")
plt.tight_layout(); plt.savefig("figures/03_model_comparison.png", dpi=120); plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
for name, p in best_models.items():
    fpr, tpr, _ = roc_curve(y_test, p.predict_proba(X_test)[:, 1])
    ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, p.predict_proba(X_test)[:, 1]):.3f})")
ax.plot([0, 1], [0, 1], "k--"); ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("Đường ROC"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("figures/04_roc.png", dpi=120); plt.close()

# Độ quan trọng đặc trưng (nếu mô hình tốt nhất là cây)
final_model = best_pipe.named_steps["model"]
if hasattr(final_model, "feature_importances_"):
    ohe = best_pipe.named_steps["preprocess"].named_transformers_["cat"].named_steps["ohe"]
    names = num_cols + fare_col + list(ohe.get_feature_names_out(cat_cols)) + bin_cols
    imp = pd.Series(final_model.feature_importances_, index=names).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.plot(kind="barh", ax=ax, color="#2a9d8f"); ax.set_title(f"Độ quan trọng đặc trưng - {best_name}")
    plt.tight_layout(); plt.savefig("figures/05_feature_importance.png", dpi=120); plt.close()

# ---------------------------------------------------------------
# 6. Đóng gói: lưu Pipeline hoàn chỉnh ra file .pkl bằng joblib
# ---------------------------------------------------------------
joblib.dump(best_pipe, "models/titanic_best_model.pkl")
print("\nĐã lưu mô hình: models/titanic_best_model.pkl")

# Kiểm tra: nạp lại và dự đoán thử trên dữ liệu thô -> kết quả phải giống nhau
loaded = joblib.load("models/titanic_best_model.pkl")
assert (loaded.predict(X_test) == y_pred).all()
print("Kiểm tra nạp lại mô hình: OK")

res.to_csv("models/model_comparison.csv", index=False)
json.dump({"best_model": best_name, "accuracy_test": acc, "roc_auc": auc,
           "baseline_accuracy_test": base["accuracy_test"],
           "best_params": res.loc[res["Mô hình"] == best_name, "Tham số tốt nhất"].iloc[0]},
          open("models/final_metrics.json", "w"), ensure_ascii=False, indent=2, default=str)
