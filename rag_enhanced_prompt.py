# Ustilise la vector db pour amméliorer un prompt avec du rag
# Auteur : Thomas Mari
# Date   : 2025-09-10
import sys      #args
from langchain_chroma import Chroma     #vector db
#Embeddings api creds
import getpass
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings #embedding prompt
from langchain_google_genai import GoogleGenerativeAI # LLM
from mytools import setup_env_variables


# Set-up environment 
setup_env_variables(auto=True, verbose=True)

#setting 
is_test = True
collection_name ="code_penal"
db_path = "./chroma_langchain_db" # Where to save data locally, remove if not necessary
try:
    user_prompt = str(sys.argv[1])
except Exception:
    print("Usage: python rag_enhanced_prompt.py <prompt_string>")
    quit()
if is_test:
    print(f"user_prompt:{user_prompt}")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


# Parametrage de Chroma
vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=embeddings,
    persist_directory=db_path,  # Where to save data locally, remove if not necessary
)
if is_test:
        print(f"data base loaded")
similarity_result = vector_store.similarity_search(user_prompt)
if is_test:
        print(f"similarity search sucess, ... Computing prompt")
system_prompt_rag = f"<system prompt>:Tu est un juriste et tu réponds à des questions juridique. Ton but est de répondre au <user prompt>, et tu peux utiliser les données <rag data> issues d'une recherche de similarité dans une bases de données vecteurisé du code pénal."+\
                    f"<rag data>:{str(similarity_result)}\n"+\
                    f"<user prompt>:{user_prompt}"
print(f"promt:{system_prompt_rag}")
llm = GoogleGenerativeAI(model="gemini-2.5-pro")
print("result with gemini-2.5-pro:")
llm_answer = llm.invoke(system_prompt_rag)
print(llm_answer)
