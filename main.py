"""
Script principal de démonstration pour le système de recommandation d'anime.

Ce script illustre l'utilisation des différents modules développés pour
charger les données, les prétraiter, les vectoriser et obtenir une liste
de recommandations à partir de favoris. Il peut être exécuté directement
en ligne de commande.
"""

from data_load import load_data
from preprocess import preprocess_synopsis
from vectorize import vectorize_synopsis
from recommend import recommend_anime


def main():
    # Chemin du fichier de données
    filepath = 'Anime.csv'

    # Chargement du jeu de données
    df = load_data(filepath)

    # Prétraitement de la colonne Synopsis
    df = preprocess_synopsis(df)

    # Vectorisation et calcul de la similarité
    _, cosine_sim, indices = vectorize_synopsis(df)

    # Exemple de favoris (à modifier selon les besoins)
    favorites = ['Shingeki no Kyojin', 'Sword Art Online', 'Naruto']

    # Obtention des recommandations
    top_recommendations = recommend_anime(favorites, top_n=10, df=df, cosine_sim=cosine_sim, indices=indices)

    # Affichage des titres recommandés
    print("Recommandations basées sur les favoris :")
    for title in top_recommendations['Title']:
        print(f"- {title}")


if __name__ == '__main__':
    main()
