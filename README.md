# Analyse Comportementale Clientèle Retail

**Projet Machine Learning — ENIS GI2 · Mahmoud Elloumi · 2025-2026**

---

## Objectif

Prédire le **churn client** (départ) d'un e-commerce retail à partir de données comportementales et démographiques, avec déploiement d'un dashboard interactif Flask.

---

## Structure du projet

```
ML-Projet/
├── app/
│   ├── app.py                  # Serveur Flask (routes, prédictions)
│   ├── templates/
│   │   ├── index.html          # Dashboard principal
│   │   └── result.html         # Page résultats batch
│   └── static/
│       └── style.css           # Thème dark tech
├── data/
│   ├── raw/                    # Dataset brut CSV
│   └── train_test/             # Données splitées
├── models/
│   ├── churn_model.pkl         # Modèle Random Forest entraîné
│   └── preprocessor.pkl        # Pipeline de prétraitement
├── src/
│   ├── preprocessing.py        # Feature engineering & préprocesseur
│   ├── train_model.py          # Entraînement du modèle
│   ├── predict.py              # Script de prédiction
│   └── utils.py                # Fonctions utilitaires
├── notebooks/                  # Analyses exploratoires
├── rapport/
│   └── ML.pdf                  # Rapport final
├── requirements.txt
└── README.md
```

---

## Pipeline Machine Learning

| Étape | Détail |
|---|---|
| Dataset | 52 features comportementales et démographiques |
| Nettoyage | Valeurs aberrantes (SupportTickets, Satisfaction) |
| Feature Engineering | MonetaryPerDay, AvgBasketValue, TenureRatio… |
| Encodage | OrdinalEncoder + OneHotEncoder |
| Normalisation | StandardScaler |
| Rééquilibrage | SMOTE |
| Optimisation | GridSearchCV (cv=5, scoring=AUC-ROC) |
| Modèle final | **Random Forest · AUC = 1.000** |

---

## Installation

```bash
git clone https://github.com/mahmoudelloumi/ML-projet.git
cd ML-projet
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Entraînement du modèle

```bash
python src/preprocessing.py
python src/train_model.py
```

## Lancer le dashboard

```bash
python app/app.py
```

Ouvrir : [http://localhost:5000](http://localhost:5000)

---

## Fonctionnalités du dashboard

- **Analyse batch** — charge automatiquement le dataset et affiche les prédictions pour tous les clients
- **Prédiction client unique** — formulaire interactif (8 champs) avec résultat instantané côte à côte
- **Visualisations** — distribution des risques, histogramme des probabilités, top 10 clients à risque
- **Statistiques** — taux de churn, nombre de churners/fidèles, risque moyen

---

## Auteur

**Mahmoud Elloumi** — ENIS GI2 · [mahmoud.elloumi@enis.tn](mailto:mahmoud.elloumi@enis)
