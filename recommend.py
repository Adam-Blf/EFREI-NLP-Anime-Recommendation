"""
Module de recommandation d'anime basé sur la similarité du synopsis.

Ce module fournit une fonction permettant de recommander des titres
d'anime similaires à ceux fournis comme favoris. La fonction s'appuie
sur la matrice de similarité cosinus calculée à l'aide des synopsis.
"""

from typing import List
import numpy as np
import pandas as pd


def recommend_anime(
    favorites: List[str],
    top_n: int,
    df: pd.DataFrame,
    cosine_sim: np.ndarray,
    indices: pd.Series,
) -> pd.DataFrame:
    """Recommande des anime en se basant sur la similarité du synopsis.

    Parameters
    ----------
    favorites : List[str]
        Liste de titres d'anime considérés comme favoris.
    top_n : int
        Nombre de recommandations à retourner.
    df : pandas.DataFrame
        DataFrame complet contenant au moins les colonnes `Title` et `Synopsis`.
    cosine_sim : numpy.ndarray
        Matrice de similarité cosinus entre les synopsis.
    indices : pandas.Series
        Série associant chaque titre d'anime à l'indice correspondant dans
        la matrice `cosine_sim`.

    Returns
    -------
    pandas.DataFrame
        DataFrame des recommandations avec les colonnes `Title` et `Synopsis`.
    """
    if not favorites:
        return pd.DataFrame(columns=['Title', 'Synopsis'])

    # Initialisation d'un tableau de scores à zéro
    sim_scores = np.zeros(len(df))

    # Sommation des similarités pour chaque favori présent
    for fav in favorites:
        if fav not in indices:
            # Si le titre n'est pas connu, on continue sans l'ajouter
            continue
        idx = indices[fav]
        sim_scores += cosine_sim[idx]

    # On retire de la liste les favoris pour ne pas les recommander à nouveau
    fav_indices = [indices[f] for f in favorites if f in indices]
    scores = [(i, score) for i, score in enumerate(sim_scores) if i not in fav_indices]

    # Classement des scores par ordre décroissant
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Sélection des meilleurs indices
    top_indices = [i for i, _ in scores[:top_n]]

    # Construction du DataFrame de résultats
    return df.iloc[top_indices][['Title', 'Synopsis']]
