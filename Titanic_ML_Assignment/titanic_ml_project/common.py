"""
common.py - Các hàm dùng chung cho cả 2 giai đoạn.
Các hàm này KHÔNG học gì từ dữ liệu (stateless) nên dùng trước khi chia train/test không gây rò rỉ dữ liệu.
"""
import re
import numpy as np
import pandas as pd

RANDOM_STATE = 42
DATA_PATH = "data/titanic.csv"


def load_data(path=DATA_PATH):
    """Đọc dữ liệu thô từ file CSV."""
    return pd.read_csv(path)


def extract_title(name: str) -> str:
    """Lấy danh xưng (Mr, Mrs, Miss...) từ cột Name và gộp các danh xưng hiếm."""
    m = re.search(r",\s*([^\.]+)\.", name)
    title = m.group(1).strip() if m else "Unknown"
    if title in ("Mlle", "Ms"):
        title = "Miss"
    elif title == "Mme":
        title = "Mrs"
    elif title not in ("Mr", "Mrs", "Miss", "Master"):
        title = "Rare"
    return title


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo đặc trưng mới (Feature Engineering) và bỏ các cột không dùng.
    - Title: danh xưng trích từ tên
    - FamilySize: kích thước gia đình = SibSp + Parch + 1
    - IsAlone: đi một mình hay không
    - HasCabin: có thông tin cabin hay không (Cabin thiếu ~77% nên không dùng trực tiếp)
    """
    df = df.copy()
    df["Title"] = df["Name"].apply(extract_title)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    return df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")


def log_fare(X):
    """Biến đổi log1p cho Fare (phân phối lệch phải, nhiều ngoại lệ). Dùng trong Pipeline."""
    return np.log1p(X)
