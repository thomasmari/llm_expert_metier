# La première chose à faire est de se créer un fichier .env à partir du fichier
# .env.example 
cp .env.example .env
# Puis remplissage du fichier .env avec les clé API notamment

# Il faut avoir installé uv : 
curl -LsSf https://astral.sh/uv/install.sh | sh

# Et mettre à jour l'env d'exécution de python
uv sync

# Commande a exécuter pour lancer l'interface (depuis le dossier où se trouve
# ce README.md) :
uv run src/main.py

# ---------------------------------------------------------------------------- #

# Sinon on peut lancer aussi directement
uv run src/interface.py

# Notre application repose sur un Agent IA expert en droit penal qui est 
# implémenté dans la classe AIExpertLawyer. On peut la tester simplement avec :
uv run src/aiexpertlawyer.py

# On a aussi créé un autre Agent qui peut juger les réponses de l'expert et lui
# proposer de s'améliorer. On peut le tester simplement comme avec :
uv run src/aijudge.py

# On peut aussi imaginer boucler les jugements pour améliorer itérativement
# l'expert. C'est ce qui est fait par exemple avec :
uv run src/optim_prompt.py

# On utilise la base de donnée (RAG) qui est dans le dossier chroma_langchain_db
# qui a été crée avec :
uv run src/fill_rag.py

# On peut explorer la base de donnée (RAG) (sans LLM donc sans utiliser de requêtes
# API) avec :
uv run src/explore_db.py

# ---------------------------------------------------------------------------- #
# Structure du projet :

├── chroma_langchain_db   # Base de donnée (RAG)
├── data 
│   ├── Code_penal.pdf    # Code pénal (source des chunks)
│   └── QA.json           # Dataset d'évaluation de l'expert
├── .env                  # Fichier qu'on a créé à partir de .env.example 
├── .env.example          # Fichier d'exemple pour configurer son .env
├── .git                  # Dossier git (pour le versionning)
├── .gitignore            # Fichier gitignore (pour le versionning)
├── interface          
│   └── index.html        # HTML utilisé par l'interface
├── pyproject.toml        # Dépendances (gérée par uv)
├── README.md             # Ce fichier
└── src
    ├── aiexpertlawyer.py # Définition de la classe AIExpertLawyer
    ├── aijudge.py        # Définition de la AIJudge
    ├── chunker.py        # Fonctions pour créer les chunks
    ├── explore_db.py     # Script d'exploration simple de la base de donnée (RAG)
    ├── fill_rag.py       # Script de création et remplissage de la base de donnée (RAG)
    ├── interface.py      # Définition de l'interface avec FastAPI
    ├── main.py           # Point d'entrée du code
    ├── mytools.py        # Diverses fonctions utiles
    └── optim_prompt.py   # Script pour juger/optimiser un expert
