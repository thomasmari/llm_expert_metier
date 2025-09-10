# -*- coding: utf8 -*-
#
# Auteur : Xavier BEDNAREK
# Date : 10/09/2025

from mytools import setup_env_variables
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.documents import Document

class AIExpertLawyer() : 
    """Classe définissant un Agent IA expert en droit penal
    """
    
    # --------------------------------------------------------------------------
    #                                                               Constructeur
    # --------------------------------------------------------------------------

    def __init__(self, *, system_prompt: str| None = None, chroma_collection_name: str = "code_penal", chroma_db_path : str = "./chroma_langchain_db", llm_model : str = "gemini-2.5-flash-lite", temperature : float = 0.3, top_p: float = 0.8, nb_chunk : int = 4) -> None:
        """Constructeur de l'Agent IA"""

        # Setup des variables d'environnement
        setup_env_variables(auto=True, verbose=False)

        # 1 - Paramétrage de la base de donnée sémantique (Chroma)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

        self._vector_store = Chroma(
            collection_name=chroma_collection_name,
            embedding_function=embeddings,
            persist_directory=chroma_db_path,  # Where to save data locally, remove if not necessary
        )

        # 2 - Paramétrage du LLM
        self._llm = GoogleGenerativeAI(model=llm_model, 
                                       temperature=temperature,  # Entre 0.0 et 1.0
                                       top_p=top_p               # Entre 0.0 et 1.0
                                       )

        # 3 - Meta prompt utilisé :
        if not system_prompt : # On fait un prompt par défaut
            self._system_prompt = ("<system prompt>:Tu es un juriste expert et tu réponds à des questions juridiques. Ton but est de répondre au <user prompt>, et tu peux utiliser les données <rag data> issues d'une recherche de similarité dans une base de données vectorisée du code pénal.\n\n" +
                                   "<rag data>:\n{rag_data}\n\n"+
                                   "<user prompt>:\n{user_prompt}\n")
        else :
            self._system_prompt = system_prompt

        # 4 - Paramétrage de la façon dont sont faites les requêtes dans la base de donnée sémantique
        self._nb_chunks = nb_chunk

    # --------------------------------------------------------------------------
    #                                                                   Méthodes
    # --------------------------------------------------------------------------

    def __str__(self) -> str:
        return (
         "==========================================\n"   
         "AIExpertLawyer with following parameters :\n" +
        f"   - system_prompt :\n     ---------------\n     {self._system_prompt.strip().replace("\n","\n     ")}\n     ---------------\n" +
        f"   - llm :\n     {str(self._llm).replace("\n","\n     ")}\n" + 
        f"   - nb_chunks : {self._nb_chunks}\n" + 
        "=========================================="   
        )
    
    def ask(self, question:str) -> str:
        """Demande quelque chose à notre agent"""

        # 1 - Requête dans la base de donnée sémantique :
        similarity_results = self.request_in_semantic_db(question)

        # 2 - Création du prompt à envoyer au LLM
        prompt = self._system_prompt.format(rag_data=similarity_results, user_prompt=question)

        # 3 - Appelle du LLM
        return self._llm.invoke(prompt)

    def request_in_semantic_db(self, query:str) -> list[Document] :
        """Fait une requête dans la base de donnée sémantique"""
        return self._vector_store.similarity_search(query=query, k=self._nb_chunks)


if __name__=='__main__':

    # Test

    # Création de l'agent
    expert = AIExpertLawyer(temperature=0.25, nb_chunk=3, top_p=0.5)
    print("🔎  Info sur l'Agent utilisé :")
    print(expert)

    # Question :
    question = "Quel est la peine de prison la plus longue envisageable ?"
    print("### Question :")
    print(f"> {question}", flush=True)

    # Réponse :
    reponse = expert.ask(question)
    print("###  Réponse :")
    print(f"> {reponse.replace("\n", "\n> ")}")
