# Utilisation du chunker pour créer une liste de chunk a mettre dans notre RAG
# Auteur : Xavier BEDNAREK
# Date   : 2025-09-09
import os
import getpass
import time
from tqdm import tqdm

from chunker import chunk_code_penal
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Si on veut juste tester ce script on mets CHUNK_LIMIT_FOR_TEST=True
# Si on veut charger tous les chunks on mets CHUNK_LIMIT_FOR_TEST=False
CHUNK_LIMIT_FOR_TEST = True

# On a 100 requete par minutes max sur l'API Google
RPM_LIMIT = 100
NAP_DURATION = 61 # secondes

# Chemin vers le PDF du code pénal
file_path = "data/Code_penal.pdf"

print("1 - 🖊️ Gestion de l'environnement.", flush=True)
# Gestion de l'API LongChain :
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Une Clé API pour Langsmith: ")

# Gestion de l'API Google pour l'embedding
os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Splits en utilisant le code généré par Claude.ai
print("2 - ✂️  Chunk du code pénal ...", end=" ", flush=True)
all_splits = chunk_code_penal(file_path)
print("✅")

# Exploration des splits
print(f"   --> Nombre de splits au total : {len(all_splits):d}")

# Suppresion des splits qui n'ont pas de numero d'article et des doublons
liste_articles = []
set_numero_articles = set()
for chunk in all_splits:
    num_article = chunk.metadata.get('article_numero', 'N/A')
    if num_article != 'N/A'and num_article not in set_numero_articles:
        set_numero_articles.add(num_article)
        liste_articles.append(chunk)

print(f"   --> Nombre de splits correspondant à des articles de loi : {len(liste_articles):d}.")
print("   --> On devrait en trouver 1297 (mais OK).") 

print("3 - 📥  Embedding", flush=True)

# Parametrage de Chroma pour la base de donnée sémantique
vector_store = Chroma(
    collection_name="code_penal",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

# Si je ne fait que tester je limite :
if CHUNK_LIMIT_FOR_TEST:
    liste_articles = liste_articles[0:30]

# Embedding :
for i in tqdm(range(len(liste_articles))):
    chunk = liste_articles[i]
    # Indexation de chunks dans le RAG
    ids = vector_store.add_documents(documents=[chunk])
    # On fait des pauses pour ne pas cramer notre compteur API
    if (i+1)%(RPM_LIMIT-5) == 0:
        print(f"\nPause après {i+1} documents...")
        time.sleep(NAP_DURATION)

# Test de l'embedding avec un requêtes simple :
print("4 - 📤  Test de requête", flush=True)
requete = "A qui est applicable le code pénal ?"
print('    On envoie la requête suivante : "{requete}"')
results = vector_store.similarity_search(requete)
print(f"    --> La requête dans le RAG a renvoyé {len(results):d} chunks.")
print(f"    Les voici :")
for i, res in enumerate(results):
    text = "============= chunk " + str(i+1) + " =================="
    print(text)
    print(res)
    print("="*len(text))