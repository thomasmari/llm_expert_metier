
import os
import getpass
import json
from dotenv import load_dotenv
from langsmith import Client
from pathlib import Path


def setup_env_variables(auto:bool=False, verbose:bool=False):
    """Set up environment variable if not already set !"""

    if auto :
        load_dotenv()
        if verbose :
            print("Environment variables loaded from .env file.")

    if not os.environ.get("LANGSMITH_TRACING"):
        os.environ["LANGSMITH_TRACING"] = "true"
    elif verbose:
        print("Environment variable LANGSMITH_TRACING : ✔️")

    if not os.environ.get("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter API key for Langsmith: ")
    elif verbose:
        print("Environment variable LANGSMITH_API_KEY : ✔️")

    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")
    elif verbose:
        print("Environment variable GOOGLE_API_KEY : ✔️")

def load_QA(fichier_json:str) -> list[dict]:
    """Lit un fichier JSON contenant des questions-réponses et retourne une liste de dictionnaire"""
    qa_list = []

    try:
        with open(fichier_json, 'r', encoding='utf-8') as f:
            qa_list = json.load(f)["qa_pairs"]
    except FileNotFoundError:
        print(f"Erreur: Le fichier {fichier_json} n'existe pas")
        return []

    return qa_list

def test_langsmithAPI():
    setup_env_variables(auto=True, verbose=True)
    try:
        client = Client()
        # Test simple pour vérifier la connexion
        projects = client.list_projects()
        print("Connexion réussie!")
        print(f"Projets disponibles: {[p.name for p in projects]}")
    except Exception as e:
        print(f"Erreur de connexion: {e}")

def create_file_if_not_exists(file_path:str) -> bool:
    """
    Vérifie si un fichier existe, le crée si nécessaire avec son arborescence.
    Renvoie vrai si le fichier existait déjà et faux si on vient de le créer
    """
    try:
        # Création d'un objet Path
        path = Path(file_path)

        # Validation basique du chemin
        if not file_path or file_path.isspace():
            raise ValueError("Le chemin du fichier ne peut pas être vide")
        
        # Vérification si le fichier existe déjà
        if path.exists():
            if path.is_file():
                return True
            else:
                raise ValueError(f"'{file_path}' existe mais n'est pas un fichier")
        
        # Création de l'arborescence si elle n'existe pas
        parent_dir = path.parent
        if not parent_dir.exists():
            parent_dir.mkdir(parents=True, exist_ok=True)

        # Création du fichier vide
        path.touch()
        
        return False
        
    except PermissionError as e:
        raise OSError(f"Permission refusée pour '{file_path}': {e}")
    except OSError as e:
        raise OSError(f"Erreur système lors de la création de '{file_path}': {e}")
    except Exception as e:
        raise ValueError(f"Erreur lors du traitement de '{file_path}': {e}")

if __name__ == "__main__":

    # Test loading QA
    qa_list = load_QA("benchmark_data/QA.json")
    for d in qa_list:
        print("-"*10)
        print(f"Question : {d["question"]}")
        print(f"Réponse  : {d["answer"]}")
        print(f"Auteur du couple Q/A : {d["author"]}")
        print(f"Tag du couple Q/A : {d["tag"]}")
