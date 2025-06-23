# scripts/evaluate.py
"""
Script d'évaluation de la performance du pipeline de classification.

Ce script effectue les étapes suivantes :
1.  Charge le pipeline de modèle entraîné.
2.  Charge les données de test (X_test, y_test).
3.  Utilise le pipeline pour faire des prédictions sur les données de test.
4.  Calcule la métrique de performance (accuracy).
5.  Sauvegarde la métrique dans un fichier JSON pour le workflow CI/CD.
"""
import os
import json
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

def evaluate_model_performance():
    """
    Charge le pipeline, évalue sa performance et sauvegarde les métriques.

    Side-effects:
        - Crée le dossier 'metrics' s'il n'existe pas.
        - Sauvegarde les métriques calculées dans 'metrics/metrics.json'.
    """
    os.makedirs('metrics', exist_ok=True)

    # 1. Charger le pipeline et les données de test
    print("Chargement du pipeline et des données de test...")
    try:
        pipeline = joblib.load('models/heart_disease_pipeline.joblib')
        X_test = pd.read_csv('data/processed/X_test.csv')
        y_test = pd.read_csv('data/processed/y_test.csv').squeeze() # .squeeze() pour convertir en Series
    except FileNotFoundError as e:
        print(f"Erreur : Un fichier est manquant. Assurez-vous que le script d'entraînement a été exécuté. Détails : {e}")
        return

    # 2. Faire des prédictions
    print("Évaluation du modèle...")
    y_pred = pipeline.predict(X_test)

    # 3. Évaluer le modèle
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Précision (Accuracy) du modèle : {accuracy:.4f}")

    # 4. Sauvegarder la métrique
    metrics = {'accuracy': accuracy}
    with open('metrics/metrics.json', 'w') as f:
        json.dump(metrics, f)

    print("Métrique sauvegardée dans 'metrics/metrics.json'")

if __name__ == "__main__":
    evaluate_model_performance()