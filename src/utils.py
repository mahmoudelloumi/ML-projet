# src/utils.py

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend non-interactif (scripts / Flask)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Qualité des données
# ---------------------------------------------------------------------------

def check_data_quality(df):
    """Analyse complète de la qualité du dataset."""
    print("=" * 60)
    print("RAPPORT DE QUALITÉ DES DONNÉES")
    print("=" * 60)
    print(f"\nDimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes")

    # Valeurs manquantes
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "Manquants": missing,
        "Pourcentage (%)": missing_pct
    }).query("Manquants > 0").sort_values("Pourcentage (%)", ascending=False)

    if not missing_df.empty:
        print("\nValeurs manquantes :")
        print(missing_df.to_string())
    else:
        print("\nAucune valeur manquante détectée.")

    # Doublons
    duplicates = df.duplicated().sum()
    print(f"\nDoublons : {duplicates}")

    # Résumé des types
    print(f"\nTypes de colonnes :\n{df.dtypes.value_counts().to_string()}")

    return missing_df


# ---------------------------------------------------------------------------
# 2. Valeurs aberrantes
# ---------------------------------------------------------------------------

def handle_aberrant_values(df):
    """
    Corrige les valeurs aberrantes connues du dataset retail :
      - SupportTicketsCount : -1 et 999 → NaN
      - SatisfactionScore   : -1, 0 et 99  → NaN
      - MonetaryTotal       : clip à 0 (retours excessifs)
    """
    df = df.copy()

    if "SupportTicketsCount" in df.columns:
        n = df["SupportTicketsCount"].isin([-1, 999]).sum()
        df["SupportTicketsCount"] = df["SupportTicketsCount"].replace([-1, 999], np.nan)
        if n:
            print(f"  handle_aberrant_values : {n} valeurs aberrantes corrigees -> SupportTicketsCount")

    if "SatisfactionScore" in df.columns:
        n = df["SatisfactionScore"].isin([-1, 0, 99]).sum()
        df["SatisfactionScore"] = df["SatisfactionScore"].replace([-1, 0, 99], np.nan)
        if n:
            print(f"  handle_aberrant_values : {n} valeurs aberrantes corrigees -> SatisfactionScore")

    if "MonetaryTotal" in df.columns:
        df["MonetaryTotal"] = df["MonetaryTotal"].clip(lower=0)

    return df


# ---------------------------------------------------------------------------
# 3. Corrélation & Multicolinéarité
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df, threshold=0.8, figsize=(18, 14), save_path=None):
    """
    Génère une heatmap de corrélation et liste les paires fortement corrélées.
    Retourne un DataFrame des paires au-dessus du seuil.
    """
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr, mask=mask, annot=False,
        cmap="RdYlGn", center=0, vmin=-1, vmax=1,
        linewidths=0.1, ax=ax
    )
    ax.set_title("Matrice de Corrélation — Features Numériques", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Heatmap sauvegardée : {save_path}")
    plt.close()

    # Paires à haute corrélation
    high_corr = [
        {"Feature 1": corr.columns[i], "Feature 2": corr.columns[j],
         "Corrélation": round(corr.iloc[i, j], 3)}
        for i in range(len(corr.columns))
        for j in range(i + 1, len(corr.columns))
        if abs(corr.iloc[i, j]) >= threshold
    ]
    hc_df = pd.DataFrame(high_corr).sort_values("Corrélation", key=abs, ascending=False) \
        if high_corr else pd.DataFrame()

    if not hc_df.empty:
        print(f"\nPairs fortement corrélées (|r| ≥ {threshold}) :")
        for _, row in hc_df.iterrows():
            print(f"  {row['Feature 1']} ↔ {row['Feature 2']} : {row['Corrélation']}")

    return hc_df


def compute_vif(X_numeric):
    """
    Calcule le VIF (Variance Inflation Factor) pour détecter la multicolinéarité.
    VIF > 10 = multicolinéarité sévère.
    """
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        print("  statsmodels non installé — pip install statsmodels")
        return pd.DataFrame()

    X = X_numeric.select_dtypes(include=["int64", "float64"]).dropna()
    vif_df = pd.DataFrame({
        "Feature": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    }).sort_values("VIF", ascending=False)

    print("\nVIF — Variance Inflation Factor (seuil critique : 10)")
    print(vif_df.to_string(index=False))
    return vif_df


# ---------------------------------------------------------------------------
# 4. ACP (Analyse en Composantes Principales)
# ---------------------------------------------------------------------------

def apply_pca(X_train, X_test, n_components=0.95, save_path=None):
    """
    Réduit la dimensionnalité via ACP.

    Paramètres
    ----------
    n_components : int ou float
        Nombre de composantes (int) ou variance cumulée cible (float, ex: 0.95).

    Retourne
    --------
    X_train_pca, X_test_pca, pca (objet fitted)
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    n = pca.n_components_
    var = pca.explained_variance_ratio_.sum() * 100
    print(f"\nACP : {X_train.shape[1]} features → {n} composantes ({var:.1f}% de variance)")

    # Courbe de variance expliquée cumulée
    pca_full = PCA().fit(X_train)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(cumvar) + 1), cumvar, marker=".", linewidth=1.5, color="steelblue")
    ax.axhline(y=95, color="red", linestyle="--", label="95 % variance")
    ax.axhline(y=90, color="orange", linestyle="--", label="90 % variance")
    ax.axvline(x=n, color="green", linestyle="--", label=f"{n} composantes sélectionnées")
    ax.set_xlabel("Nombre de composantes principales")
    ax.set_ylabel("Variance expliquée cumulée (%)")
    ax.set_title("Analyse en Composantes Principales (ACP)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Graphique ACP sauvegardé : {save_path}")
    plt.close()

    return X_train_pca, X_test_pca, pca


# ---------------------------------------------------------------------------
# 5. Distribution des classes (Churn)
# ---------------------------------------------------------------------------

def plot_class_distribution(y, title="Distribution Churn", save_path=None):
    """Visualise la distribution de la variable cible."""
    values = pd.Series(y).value_counts().sort_index()
    labels = ["Fidèle (0)", "Churner (1)"]
    colors = ["#2ecc71", "#e74c3c"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    bars = ax1.bar(labels, values.values, color=colors, edgecolor="black")
    ax1.set_title("Distribution absolue")
    ax1.set_ylabel("Nombre de clients")
    for bar, val in zip(bars, values.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(val), ha="center", fontweight="bold")

    ax2.pie(values.values, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=90)
    ax2.set_title("Distribution relative")

    fig.suptitle(title, fontweight="bold", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Graphique distribution sauvegardé : {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 6. Clustering K-Means
# ---------------------------------------------------------------------------

def kmeans_clustering(X, n_clusters_range=range(2, 8), save_path=None):
    """
    Clustering K-Means avec méthode du coude (Elbow) et score de silhouette.
    Retourne le nombre optimal de clusters et les labels.
    """
    inertias = []
    silhouettes = []

    for k in n_clusters_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels, sample_size=min(2000, len(X))))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(list(n_clusters_range), inertias, marker="o", color="steelblue")
    ax1.set_title("Méthode du coude (Inertie)")
    ax1.set_xlabel("Nombre de clusters (k)")
    ax1.set_ylabel("Inertie")
    ax1.grid(True, alpha=0.3)

    ax2.plot(list(n_clusters_range), silhouettes, marker="o", color="darkorange")
    ax2.set_title("Score de Silhouette")
    ax2.set_xlabel("Nombre de clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Sélection du nombre optimal de clusters", fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Graphique clustering sauvegardé : {save_path}")
    plt.close()

    best_k = list(n_clusters_range)[int(np.argmax(silhouettes))]
    print(f"\nNombre optimal de clusters (silhouette) : k = {best_k}")

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels_final = km_final.fit_predict(X)

    return best_k, labels_final, km_final
