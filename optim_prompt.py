# Petit script pour optimiser le prompt de notre Agent IA expert en code pénal
# Utilise pour cela un juge !

from aiexpertlawyer import AIExpertLawyer
from aijudge import AIJudge
import time
from tqdm import tqdm

# Nombre de bouclage pour optimiser le prompt
N = 3

# Durée de la pause entre deux jugement
pause_inter_seance = 61 # secondes

# On va écrire des logs ici :
log_juge = "logs/juge.log"
log_expert = "logs/expert.log"

# Paramètres du juge
temperature_juge = 0.5 # Pour qu'il soit un peu imaginatif dans la création d'un prompt
top_p_juge = 0.5

# Paramètres des experts
temperature_experts = 0.25
top_p_experts = 0.5
nb_chunk_experts = 4

# Question finale à poser à notre expert optimisé
finale_question = "Est-ce qu'on utilise toujours la guillotine ?"

# RQ : le jeu de données d'entrainement (question réponse) peut être changer en
# taguant correctement les questions dans le fichier QA.json

# RQ : on aurait pu aussi chercher à optimiser la temperature etc etc mais
# on aura pas le temps ici

# RQ : il n'est pas impossible que le prompt proposé par le juge ne soit pas 
# adéquate et alors on aura un erreur !

################################################################################

# On créer le juge :
juge = AIJudge(temperature=temperature_juge, top_p=top_p_juge, logfile=log_juge, verbose=False)

# On créer un premier expert :
expert = AIExpertLawyer(temperature=temperature_experts, nb_chunk=nb_chunk_experts, top_p=top_p_experts, logfile=log_expert)
print("Prompt de notre premier expert :\n"+("-"*10)+f"\n{expert.get_system_prompt()}\n"+("-"*10)+"\n")

# Et on boucle (pas trop de fois pour pas trop utiliser notre quotat)
for i in range(N):

    # Jugement : 
    note, proposition_prompt = juge.evaluate(expert)

    # Affichage pour le suvit :
    print(f"\nJugement numéro {i+1} : note = {note}/10")
    print("On va créer un nouvel expert avec le prompt système suivant :\n"+("-"*10)+f"\n{proposition_prompt}\n"+("-"*10)+"\n")

    # Création d'un nouvel expert :
    expert = AIExpertLawyer(temperature=temperature_experts, nb_chunk=nb_chunk_experts, top_p=top_p_experts, logfile=log_expert, system_prompt=proposition_prompt)

    # Petite pause pour éviter certain problèmes de requêtes
    print("Mais d'abord une petite pause !")
    for i in tqdm(range(pause_inter_seance*10)):
        time.sleep(0.1)


# Enfin on pose la dernière question à notre expert "optimisé" :
print("="*80+"\n")
print(f"""Question finale pour l'expert optimisé : "{finale_question}".""")
response = expert.ask(finale_question)
print(f"""Réponse de l'expert optimisé :\n"{response}"\n""")
