# app/app.py

import os
import sys
import io
import pandas as pd

from flask import Flask, render_template, request, redirect, url_for, flash
import joblib

# Accès aux modules src depuis le dossier app/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from preprocessing import feature_engineering  # noqa: E402

app = Flask(__name__)
app.secret_key = "retail_churn_secret"

# ---------------------------------------------------------------------------
# Chemins des modèles (relatifs à la racine du projet)
# ---------------------------------------------------------------------------
MODEL_PATH       = os.path.join(BASE_DIR, "models", "churn_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")


def load_models():
    model       = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files or request.files["file"].filename == "":
        flash("Veuillez sélectionner un fichier CSV.", "danger")
        return redirect(url_for("index"))

    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        flash("Format invalide — seuls les fichiers .csv sont acceptés.", "danger")
        return redirect(url_for("index"))

    try:
        df_raw = pd.read_csv(io.StringIO(file.stream.read().decode("utf-8")))
    except Exception as e:
        flash(f"Erreur de lecture du fichier : {e}", "danger")
        return redirect(url_for("index"))

    # Conserver CustomerID pour l'affichage
    customer_ids = df_raw["CustomerID"].tolist() if "CustomerID" in df_raw.columns else list(range(len(df_raw)))

    # Churn réel si présent (pour comparaison)
    actual_churn = df_raw["Churn"].tolist() if "Churn" in df_raw.columns else None

    try:
        model, preprocessor = load_models()
    except FileNotFoundError:
        flash("Modèles introuvables — lancez d'abord preprocessing.py puis train_model.py.", "danger")
        return redirect(url_for("index"))

    # Prétraitement
    X = df_raw.copy()
    if "Churn" in X.columns:
        X = X.drop(columns=["Churn"])

    try:
        X = feature_engineering(X)
        X_processed = preprocessor.transform(X)
    except Exception as e:
        flash(f"Erreur lors du prétraitement : {e}", "danger")
        return redirect(url_for("index"))

    # Prédictions
    predictions  = model.predict(X_processed)
    probabilities = (model.predict_proba(X_processed)[:, 1] * 100).round(1)

    # Construction du tableau de résultats
    rows = []
    for i, (cid, pred, prob) in enumerate(zip(customer_ids, predictions, probabilities)):
        row = {
            "customer_id": cid,
            "prediction":  "Churner" if pred == 1 else "Fidèle",
            "probability": float(prob),
            "risk_class":  _risk_class(prob),
            "actual":      actual_churn[i] if actual_churn else None,
        }
        rows.append(row)

    # Trier par probabilité décroissante
    rows.sort(key=lambda r: r["probability"], reverse=True)

    total     = len(rows)
    churners  = sum(1 for r in rows if r["prediction"] == "Churner")
    fideles   = total - churners
    avg_risk  = round(float(probabilities.mean()), 1)

    stats = {
        "total":      total,
        "churners":   churners,
        "fideles":    fideles,
        "avg_risk":   avg_risk,
        "churn_rate": round(churners / total * 100, 1),
    }

    # --- Données pour les graphiques ---

    # 1. Distribution des niveaux de risque
    risk_counts = {"Critique (>=75%)": 0, "Eleve (50-75%)": 0,
                   "Modere (25-50%)": 0, "Faible (<25%)": 0}
    for r in rows:
        p = r["probability"]
        if p >= 75:   risk_counts["Critique (>=75%)"] += 1
        elif p >= 50: risk_counts["Eleve (50-75%)"]   += 1
        elif p >= 25: risk_counts["Modere (25-50%)"]  += 1
        else:         risk_counts["Faible (<25%)"]     += 1

    # 2. Histogramme des probabilités (tranches de 10 %)
    prob_hist = [0] * 10
    for r in rows:
        idx = min(int(r["probability"] // 10), 9)
        prob_hist[idx] += 1

    # 3. Top 10 clients à risque pour le graphique bar
    top10 = rows[:10]

    charts = {
        "risk_labels":  list(risk_counts.keys()),
        "risk_values":  list(risk_counts.values()),
        "hist_labels":  [f"{i*10}-{i*10+10}%" for i in range(10)],
        "hist_values":  prob_hist,
        "top10_ids":    [str(r["customer_id"]) for r in top10],
        "top10_probs":  [r["probability"] for r in top10],
    }

    return render_template("result.html", rows=rows, stats=stats,
                           charts=charts, filename=file.filename)


def _risk_class(prob):
    if prob >= 75:
        return "danger"
    elif prob >= 50:
        return "warning"
    else:
        return "success"


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)
