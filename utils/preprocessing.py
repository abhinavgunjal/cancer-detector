import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_data():
    df = pd.read_csv("data/lung_cancer.csv")

    # =========================
    # TARGET FIX
    # =========================
    df['PULMONARY_DISEASE'] = df['PULMONARY_DISEASE'].astype(str).str.upper()
    df['PULMONARY_DISEASE'] = df['PULMONARY_DISEASE'].map({'YES': 1, 'NO': 0})

    # =========================
    # FEATURES & TARGET
    # =========================
    X = df.drop('PULMONARY_DISEASE', axis=1)
    y = df['PULMONARY_DISEASE']

    # =========================
    # NUMERIC CONVERSION
    # =========================
    X = X.apply(pd.to_numeric, errors='coerce')

    # =========================
    # HANDLE MISSING VALUES
    # =========================
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # =========================
    # SPLIT
    # =========================
    return train_test_split(X, y, test_size=0.2, random_state=42)


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)