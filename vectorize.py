"""
Module de vectorisation des synopsis.

Ce module contient une fonction qui transforme les synopsis en une matrice
TF‑IDF et calcule une matrice de similarité cosinus. Il renvoie aussi un
index inversé pour passer rapidement d'un titre d'anime à l'indice de son
synopsis dans la matrice.
"""

from typing import Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def vectorize_synopsis(df: pd.DataFrame) -> Tuple:
    """Vectorise les synopsis et calcule la similarité cosinus.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame contenant au moins les colonnes `Title` et `Synopsis`.

    Returns
    -------
    tuple
        Un triplet (tfidf_matrix, cosine_sim, indices) où :
        * tfidf_matrix est la matrice TF‑IDF des synopsis ;
        * cosine_sim est la matrice de similarité cosinus ;
        * indices est une série associant chaque titre à son index dans
          le DataFrame.
    """
    # Création du vectoriseur avec suppression des stop words anglais
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['Synopsis'])

    # Calcul de la similarité cosinus entre tous les synopsis
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Création de l'index inversé (titre -> position)
    indices = pd.Series(df.index, index=df['Title']).drop_duplicates()

    return tfidf_matrix, cosine_sim, indices
