import fitz
import os
import logging

logging.basicConfig(filename='logs/pipeline.log', level=logging.INFO)

def extract_text_from_pdf(pdf_path):
    logging.info(f"Ouverture du fichier {pdf_path}")
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    logging.info(f"{len(chunks)} chunks créés")
    return chunks
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

def embed_and_store(chunks):
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_texts(chunks, embedding=embeddings, persist_directory="./db")
    db.persist()
    return db
def query_db(db, query, k=5):
    results = db.similarity_search(query, k=k)
    return [r.page_content for r in results]
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def evaluate(db, query_path="examples/queries_legal.json"):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    queries = json.load(open(query_path))

    for q in queries:
        user_query = q['query']
        expected = q['expected']
        results = db.similarity_search(user_query, k=3)
        sims = [cosine_similarity(
            model.encode([r.page_content]), model.encode([expected])
        )[0][0] for r in results]
        print(f"Query: {user_query}\nBest match score: {max(sims):.2f}\n")
[
  {
    "query": "Quelle est la durée maximale d'un CDD selon le code du travail ?",
    "expected": "La durée maximale d'un contrat à durée déterminée est de 18 mois."
  },
  {
    "query": "Quels sont les droits du salarié en cas de licenciement abusif ?",
    "expected": "Le salarié peut obtenir des dommages et intérêts fixés par le conseil des prud'hommes."
  },
  {
    "query": "Citer un arrêt de la Cour de cassation relatif au harcèlement moral.",
    "expected": "Cass. soc., 10 nov. 2009, n° 07-45.321"
  }
]
