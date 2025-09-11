# -*- coding: utf8 -*-
#
# Auteur : Xavier BEDNAREK
# Date : 10/09/2025

from mytools import setup_env_variables, load_QA, create_file_if_not_exists
from aiexpertlawyer import AIExpertLawyer
import datetime
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.documents import Document
import re

class AIJudge() : 
    """Classe définissant un Agent IA qui va juger les réponses de notre expert
       en droit penal et lui proposer un nouveau prompt système. 
    """
    
    # --------------------------------------------------------------------------
    #                                                               Constructeur
    # --------------------------------------------------------------------------

    def __init__(self, *, system_prompt: str| None = None, llm_model : str = "gemini-2.5-flash-lite", temperature : float = 0.3, top_p: float = 0.8, verbose: bool = True, logfile : str|None = "logs/log_AIJudge.txt") -> None:
        """Constructeur de l'Agent IA"""

        # Setup des variables d'environnement
        setup_env_variables(auto=True, verbose=verbose)

        # 0 - Gestion de la verbosité
        self._verbose = verbose

        # 1 - Paramétrage du LLM :
        self._llm = GoogleGenerativeAI(model=llm_model, 
                                       temperature=temperature,  # Entre 0.0 et 1.0
                                       top_p=top_p               # Entre 0.0 et 1.0
                                       )

        # 2 - Paramétrage du system prompt utilisé :
        if not system_prompt : # On fait un prompt par défaut
            self._system_prompt = ("Tu es un expert en prompt de LLM et un juriste expert. Tu es en charge d'évaluer les réponses d'un LLM à différentes questions en te basant sur les réponses attendues.\n" +
                            "**Voici la listes des questions/réponses attendues/réponses données par le LLM**:\n{qa_text}\n\n"+
                            "Donne une note globale à ces réponses, un entier entre 0 et 10. La note de zéro est donnée si les réponses sont très mauvaises et la note de 10 si tu juges les réponses excellentes.\n"+
                            "Donne cette note entre les deux balises <note> et <fin_note>, par exemple <note>5<fin_note>.\n"+
                            "Propose aussi un nouveau prompt système pour améliorer le LLM sachant que son prompt est actuellement :\n"+
                            '"{expert_system_prompt}"\n'+
                            "Tu dois me proposer ce nouveau prompt système là encore entre deux balises <newprompt> et <fin_newprompt> et en utilisant absoluement les variables rag_data et user_prompt pour qu'il puisse fonctionner correctement. Par exemple :\n"
                            "<newprompt>Utilise les **données du RAG** pour répondre à la **question**.\n"+
                            "données du RAG :\n{rag_data}\n "+
                            "question :\n{user_prompt}\n "+
                            "<fin_newprompt>.")
        else :
            self._system_prompt = system_prompt

        # 3 - Chargement des données QA
        # Je ne vais pas tout charger, juste les données que j'ai tagée avec le tag suivant :
        reftag = "use-for-eval"
        qa_data = load_QA("benchmark_data/QA.json")
        # Filtre selon le tag
        self._qa_data = []
        for d in qa_data:
            if d["tag"] == reftag :
                self._qa_data.append(d)

        if self._verbose :
            print(f"Je vais utiliser un dataset de {len(self._qa_data)} questions/réponses !")
            print("! Attention, plus il y en a, plus je consomme de requêtes !")

        # 4 - Gestion des logs
        self._logfile = logfile
        if self._logfile  is not None :
            create_file_if_not_exists(self._logfile)
        self.log("="*80 + "\n ! ! CREATING NEW AIJudge ! ! ")

    # --------------------------------------------------------------------------
    #                                                                   Méthodes
    # --------------------------------------------------------------------------

    def __str__(self) -> str:
        # Format QA Dataset to str
        qa_dataset_str = ""
        for d in self._qa_data: 
            qa_dataset_str += f'---\nQuestion (de {d["author"]}) : \n - "{d["question"]}"\nRéponse modèle : \n - "{d["answer"]}"\n---\n'

        return (
         "==========================================\n"   
         "AIJudge with following parameters :\n" +
        f"   - system_prompt :\n     ---------------\n     {self._system_prompt.strip().replace("\n","\n     ")}\n     ---------------\n" +
        f"   - llm :\n     {str(self._llm).replace("\n","\n     ")}\n" + 
        f"   - question/answer dataset :\n     {qa_dataset_str.replace("\n","\n     ")}\n" + 
        "=========================================="   
        )
    
    def log(self, text:str):
        """Log quelque chose dans le log file (pour le suivit des requêtes par ex)"""
        if self._logfile is not None :
            with open(self._logfile, 'a', encoding='utf-8') as f:
                f.write(("-"*80)+"\n> Log (" + datetime.datetime.now().strftime("%A, %d. %B %Y %H:%M:%S") + ") :\n" +text+"\n")
    
    def evaluate(self, expert:AIExpertLawyer) -> tuple[int, str]:
        """Demande à notre agent d'évaluer un AIExpertLawyer
        Renvoie la note (sur 10) et le nouveau prompt à tester !
        """

        # 1 - Demande à l'expert de répondre à des questions :
        qa_text = ""
        counter = 0
        if self._verbose :
            print("C'est partit pour l'interrogatoire !")
        for d in self._qa_data:
            counter += 1
            if self._verbose :
                print(f"Question n°{counter} (= une requête API !) ...", end = "", flush=True)
            response = expert.ask(d["question"])
            if self._verbose :
                print(" Ok (la question est vite répondue !).", flush=True)
            # Completion du text de question réponse :
            qa_text += f"---\n"
            qa_text += f'**Question** :\n "{d["question"]}"\n'
            qa_text += f'**Réponse type attendue** :\n "{d["answer"]}"\n'
            qa_text += f'**Réponse fournie par le LLM** :\n "{response}"\n'
        qa_text += f"---\n"
        # 2 - Création du prompt à envoyer au LLM
        if self._verbose :
            print(f"Jugement     (= une requête API !) ...", end = "", flush=True)
        prompt = self._system_prompt.format(qa_text=qa_text, expert_system_prompt=expert.get_system_prompt(), rag_data="{rag_data}", user_prompt="{user_prompt}")
        self.log("On va invoquer le LLM du juge avec le prompt suivant :\n"+prompt)

        # 3 - Appelle du LLM
        jugement = self._llm.invoke(prompt)
        self.log("Voici la réponse au prompt précédent :\n"+jugement)

        # 4 - Parse de la réponse pour recupérer la note et le nouveau prompt amélioré
        note, new_prompt = self.extract_note_and_prompt(jugement)

        if self._verbose :
            print(f" Ok (la cour a rendu son verdict : {note:d}/10!).", flush=True)

        # 5 - Vérifie que la proposition de prompt est bien formattée et dispose des variables attendues
        if (new_prompt.find("{rag_data}") >= 0) and (new_prompt.find("{user_prompt}") >= 0) :
            if self._verbose :
                print(" (Et le prompt proposé semble correct)", flush=True)
        else :
            raise ValueError("Le prompt proposé ne sera pas utilisable !")

        return note, new_prompt

    @staticmethod
    def extract_note_and_prompt(input_text:str) -> tuple[int, str]:
        """
        Extrait l'entier entre les balises <note> et <fin_note> 
        et le texte entre <newprompt> et <fin_newprompt>
        
        Args:
            input_text (str): Le texte d'entrée contenant les balises
            
        Returns:
            tuple: (note_value, newprompt_text) ou (None, None) si non trouvé
        Auteur : Claude.ai
        """
        
        # Pattern pour extraire l'entier entre <note> et <fin_note>
        note_pattern = r'<note>(\d+)<fin_note>'
        note_match = re.search(note_pattern, input_text)
        note_value = int(note_match.group(1)) if note_match else None
        
        # Pattern pour extraire le texte entre <newprompt> et <fin_newprompt>
        # Le flag re.DOTALL permet au '.' de correspondre aux sauts de ligne
        prompt_pattern = r'<newprompt>(.*?)<fin_newprompt>'
        prompt_match = re.search(prompt_pattern, input_text, re.DOTALL)
        newprompt_text = prompt_match.group(1).strip() if prompt_match else None

        return note_value, newprompt_text


if __name__=='__main__':

    # Test

    # Création de l'agent
    juge = AIJudge(temperature=0.25, top_p=0.5, logfile=None)
    print("🔎  Info sur l'Agent utilisé :")
    print(juge)

    # Création de l'expert à évaluer:
    expert = AIExpertLawyer(temperature=0.25, nb_chunk=3, top_p=0.5, logfile=None)
    note, proposition_prompt = juge.evaluate(expert)

    # Affichage de la note et du prompt proposé
    print(f"Note donnée par le juge : {note:d}")
    print(f"Prompt proposé par le juge :\n {proposition_prompt}")

    