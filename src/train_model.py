# src/train_model.py

import pandas as pd
import numpy as np
import joblib
import os
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    classification_report,
    roc_auc_score, ConfusionMatrixDisplay
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def load_data():
    X_train = pd.read_csv("data/train_test/X_train.csv")
    X_test  = pd.read_csv("data/train_test/X_test.csv")
    y_train = pd.read_csv("data/train_test/y_train.csv").values.ravel()
    y_test  = pd.read_csv("data/train_test/y_test.csv").values.ravel()
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Comparaison de modèles
# ---------------------------------------------------------------------------

CANDIDATES = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest":       RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
}


def compare_models(X_train, X_test, y_train, y_test, save_path=None):
    """
    Entraîne plusieurs classifieurs et compare leurs performances.
    Retourne un DataFrame récapitulatif et le nom du meilleur modèle (ROC-AUC).
    """
    results = []

    for name, clf in CANDIDATES.items():
        clf.fit(X_train, y_train)
        y_pred  = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        cv_auc = cross_val_score(clf, X_train, y_train,
                                 cv=5, scoring="roc_auc").mean()

        results.append({
            "Modèle":        name,
            "Accuracy":      round((y_pred == y_test).mean(), 4),
            "ROC-AUC Test":  round(roc_auc_score(y_test, y_proba), 4),
            "ROC-AUC CV":    round(cv_auc, 4),
        })
        print(f"  {name:25s} | Acc={results[-1]['Accuracy']:.4f} "
              f"| AUC-test={results[-1]['ROC-AUC Test']:.4f} "
              f"| AUC-CV={results[-1]['ROC-AUC CV']:.4f}")

    results_df = pd.DataFrame(results).sort_values("ROC-AUC Test", ascending=False)
    best_name  = results_df.iloc[0]["Modèle"]

    print(f"\nMeilleur modèle : {best_name}")

    if save_path:
        results_df.to_csv(save_path, index=False)
        print(f"  Comparaison sauvegardée : {save_path}")

    return results_df, best_name


# ---------------------------------------------------------------------------
# Optimisation des hyperparamètres (GridSearchCV)
# ---------------------------------------------------------------------------

PARAM_GRIDS = {
    "Random Forest": {
        "n_estimators":      [100, 200],
        "max_depth":         [None, 10, 20],
        "min_samples_split": [2, 5],
        "class_weight":      ["balanced"],
    },
    "Gradient Boosting": {
        "n_estimators":  [100, 200],
        "max_depth":     [3, 5],
        "learning_rate": [0.05, 0.1],
    },
    "Logistic Regression": {
        "C":             [0.01, 0.1, 1, 10],
        "solver":        ["lbfgs", "saga"],
        "class_weight":  ["balanced"],
        "max_iter":      [1000],
    },
}


def tune_model(X_train, y_train, model_name):
    """
    GridSearchCV sur le modèle sélectionné.
    Retourne le meilleur estimateur.
    """
    base_clf  = CANDIDATES[model_name]
    param_grid = PARAM_GRIDS.get(model_name, {})

    if not param_grid:
        print(f"  Pas de grille définie pour {model_name}, modèle retourné tel quel.")
        return base_clf

    print(f"\nGridSearchCV sur {model_name} (cv=5, scoring=roc_auc)...")
    grid = GridSearchCV(
        base_clf.__class__(**({} if model_name == "Gradient Boosting"
                               else {"random_state": 42} if "random_state"
                                    in base_clf.__class__.__init__.__code__.co_varnames
                               else {})),
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    print(f"  Meilleurs paramètres : {grid.best_params_}")
    print(f"  Meilleur AUC-CV      : {grid.best_score_:.4f}")
    return grid.best_estimator_


# ---------------------------------------------------------------------------
# Évaluation détaillée
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name="Modèle", save_dir=None):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*60}")
    print(f"ÉVALUATION : {model_name}")
    print(f"{'='*60}")
    print("\nClassification Report :\n")
    print(classification_report(y_test, y_pred,
                                target_names=["Fidèle (0)", "Churner (1)"]))
    print(f"ROC-AUC : {roc_auc_score(y_test, y_proba):.4f}")

    # Matrice de confusion
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["Fidèle", "Churner"],
        cmap="Blues", ax=ax
    )
    ax.set_title(f"Matrice de confusion — {model_name}", fontweight="bold")
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"confusion_matrix_{model_name.replace(' ', '_')}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Matrice sauvegardée : {path}")
    plt.close()

    # Importance des features (si disponible)
    if hasattr(model, "feature_importances_") and save_dir:
        importances = pd.Series(model.feature_importances_).nlargest(20)
        fig, ax = plt.subplots(figsize=(8, 6))
        importances.sort_values().plot.barh(ax=ax, color="steelblue")
        ax.set_title(f"Top 20 Features — {model_name}", fontweight="bold")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        path = os.path.join(save_dir, f"feature_importance_{model_name.replace(' ', '_')}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Importance features sauvegardée : {path}")

    return roc_auc_score(y_test, y_proba)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def main():

    print("Chargement des données train/test...")
    X_train, X_test, y_train, y_test = load_data()

    # Rééquilibrage SMOTE (sur le train uniquement)
    print("\nSMOTE — rééquilibrage des classes...")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"  Avant : {dict(zip(*np.unique(y_train,  return_counts=True)))}")
    print(f"  Après : {dict(zip(*np.unique(y_train_bal, return_counts=True)))}")

    os.makedirs("reports", exist_ok=True)

    # 1. Comparaison des modèles
    print("\n--- Comparaison des modèles ---")
    results_df, best_name = compare_models(
        X_train_bal, X_test, y_train_bal, y_test,
        save_path="reports/model_comparison.csv"
    )
    print("\n" + results_df.to_string(index=False))

    # 2. Optimisation du meilleur modèle
    best_model = tune_model(X_train_bal, y_train_bal, best_name)

    # 3. Évaluation finale
    evaluate_model(best_model, X_test, y_test,
                   model_name=best_name, save_dir="reports")

    # 4. Sauvegarde
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/churn_model.pkl")
    print(f"\nModèle sauvegardé : models/churn_model.pkl ({best_name})")


if __name__ == "__main__":
    main()
