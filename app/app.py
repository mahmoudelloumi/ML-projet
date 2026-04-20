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


@app.route("/predict_client", methods=["POST"])
def predict_client():
    from flask import jsonify
    data = request.form

    def flt(key, default=0.0):
        try:
            return float(data.get(key, default))
        except (ValueError, TypeError):
            return float(default)

    def sti(key, default=0):
        try:
            return int(float(data.get(key, default)))
        except (ValueError, TypeError):
            return int(default)

    recency  = flt("Recency", 100)
    freq     = max(flt("Frequency", 10), 1)
    monetary = flt("MonetaryTotal", 500)
    tenure   = flt("CustomerTenureDays", 365)
    mon_avg  = monetary / freq

    row = {
        "Recency":                    recency,
        "Frequency":                  freq,
        "MonetaryTotal":              monetary,
        "MonetaryAvg":                mon_avg,
        "MonetaryStd":                mon_avg * 0.3,
        "MonetaryMin":                mon_avg * 0.2,
        "MonetaryMax":                mon_avg * 2.5,
        "TotalQuantity":              freq * 5,
        "AvgQuantityPerTransaction":  5.0,
        "MinQuantity":                1,
        "MaxQuantity":                20,
        "CustomerTenureDays":         tenure,
        "FirstPurchaseDaysAgo":       tenure + recency,
        "PreferredDayOfWeek":         2,
        "PreferredHour":              14,
        "PreferredMonth":             6,
        "WeekendPurchaseRatio":       0.25,
        "AvgDaysBetweenPurchases":    max(tenure / freq, 1),
        "UniqueProducts":             max(int(freq * 0.8), 1),
        "UniqueDescriptions":         max(int(freq * 0.7), 1),
        "AvgProductsPerTransaction":  3.0,
        "UniqueCountries":            1,
        "NegativeQuantityCount":      0,
        "ZeroPriceCount":             0,
        "CancelledTransactions":      0,
        "ReturnRatio":                0.02,
        "TotalTransactions":          int(freq),
        "UniqueInvoices":             int(freq),
        "AvgLinesPerInvoice":         5.0,
        "Age":                        35.0,
        "RegistrationDate":           "2020-01-01",
        "NewsletterSubscribed":       "Yes",
        "LastLoginIP":                "0.0.0.0",
        "SupportTicketsCount":        flt("SupportTicketsCount", 1),
        "SatisfactionScore":          flt("SatisfactionScore", 3),
        "AgeCategory":                "25-34",
        "SpendingCategory":           data.get("SpendingCategory", "Medium"),
        "CustomerType":               "Actif",
        "FavoriteSeason":             "Printemps",
        "PreferredTimeOfDay":         "Matin",
        "Region":                     "UK",
        "LoyaltyLevel":               data.get("LoyaltyLevel", "Établi"),
        "WeekendPreference":          "Semaine",
        "BasketSizeCategory":         "Moyen",
        "ProductDiversity":           "Explorateur",
        "Gender":                     "Unknown",
        "AccountStatus":              "Active",
        "Country":                    "United Kingdom",
    }

    df_row = pd.DataFrame([row])

    try:
        model, preprocessor = load_models()
    except FileNotFoundError:
        return jsonify({"error": "Modèles introuvables"}), 500

    try:
        X = feature_engineering(df_row)
        X_processed = preprocessor.transform(X)
    except Exception as e:
        return jsonify({"error": f"Erreur prétraitement : {e}"}), 400

    pred  = int(model.predict(X_processed)[0])
    prob  = round(float(model.predict_proba(X_processed)[0, 1]) * 100, 1)
    label = "Churner" if pred == 1 else "Fidèle"
    risk  = _risk_class(prob)

    return jsonify({"prediction": label, "probability": prob, "risk": risk})


DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "retail_customers_COMPLETE_CATEGORICAL.csv")


def _run_predictions(df_raw):
    customer_ids = df_raw["CustomerID"].tolist() if "CustomerID" in df_raw.columns else list(range(len(df_raw)))
    actual_churn = df_raw["Churn"].tolist() if "Churn" in df_raw.columns else None

    model, preprocessor = load_models()

    X = df_raw.copy()
    if "Churn" in X.columns:
        X = X.drop(columns=["Churn"])

    X = feature_engineering(X)
    X_processed = preprocessor.transform(X)

    predictions   = model.predict(X_processed)
    probabilities = (model.predict_proba(X_processed)[:, 1] * 100).round(1)

    rows = []
    for i, (cid, pred, prob) in enumerate(zip(customer_ids, predictions, probabilities)):
        rows.append({
            "customer_id": cid,
            "prediction":  "Churner" if pred == 1 else "Fidèle",
            "probability": float(prob),
            "risk_class":  _risk_class(prob),
            "actual":      actual_churn[i] if actual_churn else None,
        })

    rows.sort(key=lambda r: r["probability"], reverse=True)

    total    = len(rows)
    churners = sum(1 for r in rows if r["prediction"] == "Churner")
    stats = {
        "total":      total,
        "churners":   churners,
        "fideles":    total - churners,
        "avg_risk":   round(float(probabilities.mean()), 1),
        "churn_rate": round(churners / total * 100, 1),
    }

    risk_counts = {"Critique (>=75%)": 0, "Eleve (50-75%)": 0,
                   "Modere (25-50%)": 0, "Faible (<25%)": 0}
    prob_hist = [0] * 10
    for r in rows:
        p = r["probability"]
        if p >= 75:   risk_counts["Critique (>=75%)"] += 1
        elif p >= 50: risk_counts["Eleve (50-75%)"]   += 1
        elif p >= 25: risk_counts["Modere (25-50%)"]  += 1
        else:         risk_counts["Faible (<25%)"]     += 1
        prob_hist[min(int(p // 10), 9)] += 1

    top10 = rows[:10]
    charts = {
        "risk_labels": list(risk_counts.keys()),
        "risk_values": list(risk_counts.values()),
        "hist_labels": [f"{i*10}-{i*10+10}%" for i in range(10)],
        "hist_values": prob_hist,
        "top10_ids":   [str(r["customer_id"]) for r in top10],
        "top10_probs": [r["probability"] for r in top10],
    }
    return rows, stats, charts


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

    try:
        rows, stats, charts = _run_predictions(df_raw)
    except FileNotFoundError:
        flash("Modèles introuvables — lancez d'abord preprocessing.py puis train_model.py.", "danger")
        return redirect(url_for("index"))
    except Exception as e:
        flash(f"Erreur : {e}", "danger")
        return redirect(url_for("index"))

    return render_template("result.html", rows=rows, stats=stats,
                           charts=charts, filename=file.filename)


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        df_raw = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        flash("Fichier de données introuvable.", "danger")
        return redirect(url_for("index"))

    try:
        rows, stats, charts = _run_predictions(df_raw)
    except FileNotFoundError:
        flash("Modèles introuvables — lancez d'abord preprocessing.py puis train_model.py.", "danger")
        return redirect(url_for("index"))
    except Exception as e:
        flash(f"Erreur : {e}", "danger")
        return redirect(url_for("index"))

    return render_template("result.html", rows=rows, stats=stats,
                           charts=charts, filename="retail_customers_COMPLETE_CATEGORICAL.csv")


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
