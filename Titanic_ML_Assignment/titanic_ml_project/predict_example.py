"""Ví dụ nạp mô hình .pkl và dự đoán cho hành khách mới (dữ liệu thô, chưa xử lý)."""
import joblib
import pandas as pd
import common  # cần import để joblib tìm thấy hàm add_features / log_fare đã dùng trong Pipeline

model = joblib.load("models/titanic_best_model.pkl")
new_passengers = pd.DataFrame([
    {"PassengerId": 1, "Pclass": 1, "Name": "Smith, Mrs. Anna", "Sex": "female", "Age": 30,
     "SibSp": 1, "Parch": 0, "Ticket": "X1", "Fare": 80.0, "Cabin": "B22", "Embarked": "S"},
    {"PassengerId": 2, "Pclass": 3, "Name": "Doe, Mr. John", "Sex": "male", "Age": 25,
     "SibSp": 0, "Parch": 0, "Ticket": "X2", "Fare": 7.9, "Cabin": None, "Embarked": "S"},
])
for p, pr in zip(model.predict(new_passengers), model.predict_proba(new_passengers)[:, 1]):
    print("Sống" if p else "Chết", f"(xác suất sống {pr:.2f})")
