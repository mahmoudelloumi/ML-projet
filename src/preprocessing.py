# src/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import handle_aberrant_values


# ---------------------------------------------------------------------------
# Ordre des features ordinales (du plus faible au plus fort)
# ---------------------------------------------------------------------------

ORDINAL_FEATURES = {
    "AgeCategory":        ["18-24", "25-34", "35-44", "45-54", "55-64", "65+", "Inconnu"],
    "SpendingCategory":   ["Low", "Medium", "High", "VIP"],
    "LoyaltyLevel":       ["Nouveau", "Jeune", "\u00c9tabli", "Ancien", "Inconnu"],
    "BasketSizeCategory": ["Petit", "Moyen", "Grand", "Inconnu"],
    "PreferredTimeOfDay": ["Nuit", "Matin", "Midi", "Apr\u00e8s-midi", "Soir"],
}


def load_data(path):
    return pd.read_csv(path)


def feature_engineering(df):
    """
    Nettoyage, correction des valeurs aberrantes et création de nouvelles features.
    Compatible avec ou sans la colonne Churn.
    """
    df = df.copy()

    # --- Valeurs aberrantes (SupportTickets, Satisfaction) ---
    df = handle_aberrant_values(df)

    # --- Parsing RegistrationDate ---
    if "RegistrationDate" in df.columns:
        df["RegistrationDate"] = pd.to_datetime(
            df["RegistrationDate"],
            dayfirst=True,
            errors="coerce",
            format="mixed"
        )
        df["RegYear"]      = df["RegistrationDate"].dt.year
        df["RegMonth"]     = df["RegistrationDate"].dt.month
        df["RegDay"]       = df["RegistrationDate"].dt.day
        df["RegDayOfWeek"] = df["RegistrationDate"].dt.dayofweek
        df["IsWeekendReg"] = df["RegistrationDate"].dt.dayofweek.isin([5, 6]).astype(int)

    # --- Log-transformations (distributions asymétriques) ---
    if "MonetaryTotal" in df.columns:
        df["LogMonetaryTotal"] = np.log1p(df["MonetaryTotal"].clip(lower=0))
    if "Frequency" in df.columns:
        df["LogFrequency"] = np.log1p(df["Frequency"])

    # --- Feature engineering RFM ---
    if "MonetaryTotal" in df.columns and "Recency" in df.columns:
        df["MonetaryPerDay"] = df["MonetaryTotal"] / (df["Recency"] + 1)
    if "MonetaryTotal" in df.columns and "Frequency" in df.columns:
        df["AvgBasketValue"] = df["MonetaryTotal"] / (df["Frequency"] + 1)
    if "Recency" in df.columns and "CustomerTenureDays" in df.columns:
        df["TenureRatio"] = df["Recency"] / (df["CustomerTenureDays"] + 1)
    if "Frequency" in df.columns and "CustomerTenureDays" in df.columns:
        df["FrequencyPerDay"] = df["Frequency"] / (df["CustomerTenureDays"] + 1)
    if "Recency" in df.columns and "Frequency" in df.columns:
        df["Recency_x_Frequency"] = df["Recency"] * df["Frequency"]

    # --- Suppression des colonnes inutiles / fuite de données ---
    cols_to_drop = [
        "CustomerID",
        "NewsletterSubscribed",   # constante (toujours "Yes")
        "RegistrationDate",       # parsée ci-dessus
        "LastLoginIP",            # identifiant unique
        "ChurnRiskCategory",      # fuite de données (dérivé de Churn)
        "RFMSegment",             # fuite de données
        "Recency",                # remplacé par les features engineered
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    return df


def split_data(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def build_preprocessor(X):
    """
    Pipeline de prétraitement à trois branches :
      - Numériques  : imputation médiane + StandardScaler
      - Ordinaux    : imputation mode + OrdinalEncoder (avec ordre métier)
      - Nominaux    : imputation mode + OneHotEncoder
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Features ordinales présentes dans le dataset
    ordinal_features    = [f for f in ORDINAL_FEATURES if f in X.columns]
    ordinal_categories  = [ORDINAL_FEATURES[f] for f in ordinal_features]

    # Features nominales = tout le reste (object/string) sauf les ordinaux
    all_cat = X.select_dtypes(include=["object", "string"]).columns.tolist()
    nominal_features = [f for f in all_cat if f not in ordinal_features]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=ordinal_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )),
    ])

    nominal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = [("num", numeric_pipeline, numeric_features)]
    if ordinal_features:
        transformers.append(("ord", ordinal_pipeline, ordinal_features))
    if nominal_features:
        transformers.append(("cat", nominal_pipeline, nominal_features))

    return ColumnTransformer(transformers=transformers)


def main():

    print("Chargement des données...")
    df = load_data("data/raw/retail_customers_COMPLETE_CATEGORICAL.csv")

    print("Feature engineering & nettoyage...")
    df = feature_engineering(df)

    print("Split train/test (80/20 stratifié)...")
    X_train, X_test, y_train, y_test = split_data(df)

    print("Construction et fit du préprocesseur...")
    preprocessor = build_preprocessor(X_train)

    # Fit UNIQUEMENT sur le train (évite le data leakage)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed  = preprocessor.transform(X_test)

    os.makedirs("models",          exist_ok=True)
    os.makedirs("data/train_test", exist_ok=True)

    joblib.dump(preprocessor, "models/preprocessor.pkl")

    def to_dense(M):
        return M.toarray() if hasattr(M, "toarray") else M

    pd.DataFrame(to_dense(X_train_processed)).to_csv("data/train_test/X_train.csv", index=False)
    pd.DataFrame(to_dense(X_test_processed)).to_csv("data/train_test/X_test.csv",  index=False)
    y_train.to_csv("data/train_test/y_train.csv", index=False)
    y_test.to_csv("data/train_test/y_test.csv",   index=False)

    print(f"Preprocessing terminé — X_train : {X_train_processed.shape}, X_test : {X_test_processed.shape}")


if __name__ == "__main__":
    main()
