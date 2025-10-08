"""
Script interactif pour explorer le système de recommandation d'anime.

Ce programme propose une interface en ligne de commande qui demande à
l'utilisateur ce qu'il souhaite faire : afficher les premières lignes du
jeu de données, obtenir des recommandations en fournissant des titres
favoris, afficher une explication du modèle ou quitter le programme.

Remarque : l'utilisation d'un grand modèle de langage (LLM) pour répondre
aux questions n'est pas possible dans ce contexte car aucune connexion à
un service externe n'est disponible. À la place, une explication
préparée du fonctionnement du modèle est fournie.
"""

import sys

from data_load import load_data
from preprocess import preprocess_synopsis
from vectorize import vectorize_synopsis
from recommend import recommend_anime


def main() -> None:
    # Chargement et préparation des données
    df = load_data('Anime.csv')
    df = preprocess_synopsis(df)
    _, cosine_sim, indices = vectorize_synopsis(df)

    # Explication statique du modèle
    explanation = (
        "Ce système de recommandation est basé sur la similarité du synopsis.\n"
        "Chaque résumé est converti en vecteur TF‑IDF, puis la similarité cosinus\n"
        "entre ces vecteurs est calculée. En fournissant une liste d'anime\n"
        "favoris, le programme combine les scores de similarité pour recommander\n"
        "les titres les plus proches.\n"
        "\n"
        "Aucun grand modèle de langage (LLM) externe n'est utilisé, car la\n"
        "connexion à des services distants n'est pas disponible."
    )

    while True:
        print("\nQue souhaitez-vous faire ?")
        print("1. Afficher les 5 premières lignes du jeu de données")
        print("2. Obtenir des recommandations d'anime")
        print("3. Afficher l'explication du modèle")
        print("4. Quitter")

        choice = input("Votre choix (1-4) : ").strip()

        if choice == '1':
            print(df.head(5))
        elif choice == '2':
            fav_input = input(
                "Entrez les titres favoris séparés par des virgules (exactement comme dans la colonne 'Title') : "
            )
            favorites = [title.strip() for title in fav_input.split(',') if title.strip()]
            if not favorites:
                print("Aucun favori fourni. Veuillez réessayer.")
                continue
            top = recommend_anime(favorites, top_n=10, df=df, cosine_sim=cosine_sim, indices=indices)
            if top.empty:
                print("Aucune recommandation disponible (vérifiez les titres saisis).")
            else:
                print("\nRecommandations :")
                for title in top['Title']:
                    print(f"- {title}")
        elif choice == '3':
            print("\n" + explanation + "\n")
        elif choice == '4':
            print("Au revoir !")
            sys.exit(0)
        else:
            print("Choix invalide. Veuillez sélectionner une option entre 1 et 4.")


if __name__ == '__main__':
    main()
