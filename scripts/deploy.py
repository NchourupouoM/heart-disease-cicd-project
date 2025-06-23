# scripts/deploy.py
"""
Script de déploiement du modèle sur le Hub Hugging Face.

Ce script utilise la bibliothèque `huggingface_hub` pour téléverser le pipeline
de modèle entraîné vers un repository sur le Hub. Il s'authentifie en utilisant
un token d'API qui doit être fourni via une variable d'environnement.
"""
import os
from huggingface_hub import HfApi, HfFolder
from huggingface_hub.utils import HfHubHTTPError

def deploy_to_hf():
    """
    Téléverse le pipeline de modèle sur le Hub Hugging Face.

    Lit le token d'API depuis la variable d'environnement `HF_API_KEY`.
    Définit le nom du repository sur le Hub.
    Crée le repository s'il n'existe pas.
    Téléverse le fichier du pipeline et un fichier README (Model Card).

    Raises:
        ValueError: Si le token `HF_API_KEY` n'est pas trouvé.
    """
    # Le token est lu depuis les secrets GitHub
    hf_token = os.environ.get("HF_API_KEY")
    if not hf_token:
        raise ValueError("Token API Hugging Face non trouvé. Définir le secret HF_API_KEY.")

    # S'authentifier
    HfFolder.save_token(hf_token)
    
    # Nom du repo sur le Hub Hugging Face (à adapter)
    repo_name = "NchourupouoM/heart-disease-classifier" # REMPLACEZ PAR VOTRE NOM D'UTILISATEUR HF

    print(f"Préparation du déploiement sur le repository : {repo_name}")
    api = HfApi()

    # Créer le repo s'il n'existe pas
    try:
        api.create_repo(repo_id=repo_name, repo_type="model", exist_ok=True)
        print(f"Repository '{repo_name}' créé ou déjà existant.")
    except HfHubHTTPError as e:
        print(f"Erreur lors de la création du repository (il peut déjà exister sous une autre organisation) : {e}")
        # On continue, car l'upload peut quand même fonctionner si le repo existe.

    # Uploader le pipeline du modèle
    try:
        api.upload_file(
            path_or_fileobj="models/heart_disease_pipeline.joblib",
            path_in_repo="heart_disease_pipeline.joblib",
            repo_id=repo_name,
            repo_type="model",
        )
        print(f"Fichier du pipeline téléversé avec succès sur '{repo_name}'.")

    except Exception as e:
        print(f"Erreur lors du téléversement du pipeline : {e}")

if __name__ == "__main__":
    deploy_to_hf()