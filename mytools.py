
import os
import getpass
import json
from dotenv import load_dotenv

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

if __name__ == "__main__":
    # Test loading QA
    qa_list = load_QA("benchmark_data/QA.json")
    for d in qa_list:
        print("-"*10)
        print(f"Question : {d["question"]}")
        print(f"Réponse  : {d["answer"]}")
        print(f"Auteur du couple Q/A : {d["author"]}")
        print(f"Tag du couple Q/A : {d["tag"]}")