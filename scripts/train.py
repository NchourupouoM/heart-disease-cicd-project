# scripts/train.py
"""
Script d'entraînement du modèle de classification des maladies cardiaques.

Ce script effectue les étapes suivantes :
1.  Charge le jeu de données 'heart.csv'.
2.  Identifie les caractéristiques numériques et catégorielles.
3.  Crée un pipeline de prétraitement avec scikit-learn pour :
    - Mettre à l'échelle les caractéristiques numériques (StandardScaler).
    - Encoder les caractéristiques catégorielles (OneHotEncoder).
4.  Définit un pipeline complet qui enchaîne le prétraitement et le modèle de
    régression logistique.
5.  Divise les données en ensembles d'entraînement et de test.
6.  Entraîne le pipeline complet sur les données d'entraînement.
7.  Sauvegarde l'objet pipeline entraîné pour une utilisation ultérieure.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

def create_and_train_pipeline():
    """
    Crée, entraîne et sauvegarde un pipeline de prétraitement et de modélisation.

    Cette fonction encapsule tout le processus : chargement des données, définition
    du pipeline, entraînement et sauvegarde.

    Side-effects:
        - Crée les dossiers 'models' et 'data/processed' s'ils n'existent pas.
        - Sauvegarde les données de test (X_test, y_test) dans 'data/processed'.
        - Sauvegarde le pipeline entraîné dans 'models/heart_disease_pipeline.joblib'.
    """
    # Créer les dossiers nécessaires
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    # 1. Charger les données
    print("Chargement des données...")
    df = pd.read_csv('data/heart.csv')

    # Séparer les features (X) et la target (y)
    X = df.drop('target', axis=1)
    y = df['target']

    # 2. Définir les colonnes numériques et catégorielles
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    # 3. Créer le transformateur de prétraitement
    # Ce transformateur applique différentes transformations à différentes colonnes.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # 4. Créer le pipeline complet
    # Il enchaîne le préprocesseur et le modèle de régression logistique.
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

    # 5. Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Sauvegarder les données de test pour l'évaluation séparée
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)

    # 6. Entraîner le pipeline
    print("Entraînement du pipeline...")
    pipeline.fit(X_train, y_train)
    print("Pipeline entraîné avec succès.")

    # 7. Sauvegarder le pipeline
    joblib.dump(pipeline, 'models/heart_disease_pipeline.joblib')
    print("Pipeline sauvegardé dans 'models/heart_disease_pipeline.joblib'")

if __name__ == "__main__":
    create_and_train_pipeline()