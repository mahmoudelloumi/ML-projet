import pandas as pd
import joblib
from preprocessing import feature_engineering

def predict_new_clients(csv_path):
    print(f"Chargement des données depuis {csv_path}...")
    df = pd.read_csv(csv_path)

    # Si la cible "Churn" est (étonnamment) là, on la garde de côté pour comparer
    actual_churn = None
    if "Churn" in df.columns:
        actual_churn = df["Churn"]
        X = df.drop(columns=["Churn"])
    else:
        X = df

    # Feature Engineering
    print("Application du Feature Engineering...")
    X = feature_engineering(X)

    # Chargement du preprocessor et transformation
    print("Application du Preprocessing...")
    preprocessor = joblib.load("models/preprocessor.pkl")
    X_processed = preprocessor.transform(X)

    # Chargement du modèle de prédiction
    print("Chargement du modèle de Machine Learning...")
    model = joblib.load("models/churn_model.pkl")

    # Prédiction de la probabilité et de la classe
    print("Prédiction en cours...")
    predictions = model.predict(X_processed)
    probabilities = model.predict_proba(X_processed)[:, 1]

    # Ajout des résultats aux features pour consultation
    results = df.copy()
    results["Prediction_Churn"] = predictions
    results["Probabilite_Depart_%"] = (probabilities * 100).round(2)

    # Affichage des 5 premiers résultats avec les probabilités les plus élevées
    print("\n--- RÉSULTATS DES PRÉDICTIONS (Top 5 risques de départ) ---")
    
    # Identifier les clients et leurs probabilités
    top_5 = results.sort_values(by="Probabilite_Depart_%", ascending=False).head(5)
    
    for index, row in top_5.iterrows():
        print(f"\nClient ID: {row['CustomerID']}")
        print(f"Probabilité de départ: {row['Probabilite_Depart_%']} %")
        print(f"Valeur monétaire: {row['MonetaryTotal']} €")
        if actual_churn is not None:
             print(f"Vrai statut (0=Fidèle, 1=Départ): {actual_churn.iloc[index]}")
    
    print("\nLes prédictions ont réussi. Vous pouvez les utiliser dans votre logiciel.")

if __name__ == "__main__":
    # Testons sur un échantillon pour la démonstration
    predict_new_clients("data/raw/retail_customers_COMPLETE_CATEGORICAL.csv")
