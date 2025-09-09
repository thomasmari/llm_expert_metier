# Un peu d'exploration de la base de donnée mais sans requète API (donc en n'utilisant pas le RAG ...)
# Juste pour voir si on a bien réussi à charger la base !
# Auteur : Xavier BEDNAREK
# Date   : 2025-09-09
import os
import getpass
import time
from tqdm import tqdm

from langchain_chroma import Chroma

# Parametrage de Chroma
vector_store = Chroma(
    collection_name="code_penal",
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

# Nombre de vecteurs dans le RAG :
collection = vector_store._collection
count = collection.count()
print(f"Nombre de vecteurs dans le vector store : {count}")

# Récupéreration de tous les documents
print("\n=== Récupérer tous les documents ===")
all_data = vector_store.get()
print(f"Clés disponibles : {all_data.keys()}")
print(f"Nombre d'IDs : {len(all_data['ids'])}")

# Afficher les premiers documents
for i in range(min(3, len(all_data['ids']))):  # Afficher les 3 premiers
    print(f"\nDocument {i+1}:")
    print(f"ID: {all_data['ids'][i]}")
    print(f"Métadonnées: {all_data['metadatas'][i]}")
    print(f"Contenu: {all_data['documents'][i][:200]}...")  # Premiers 200 caractères

# Récupérer avec des filtres sur les métadonnées
print("\n=== Filtrer par métadonnées ===")
# Exemple : récupérer tous les documents d'un article spécifique
try:
    # Vous devez adapter le filtre selon vos métadonnées
    filtered_docs = vector_store.get(
        where={"article_numero": "131-7"}  # Exemple : article numéro 131-7
    )
    print(f"Documents trouvés avec filtre : {len(filtered_docs['ids'])}")
    
    if filtered_docs['ids']:
        print(f"Premier document filtré:")
        print(f"ID: {filtered_docs['ids'][0]}")
        print(f"Métadonnées: {filtered_docs['metadatas'][0]}")
        print(f"Contenu: {filtered_docs['documents'][0][:200]}...")
        
except Exception as e:
    print(f"Erreur avec le filtre : {e}")