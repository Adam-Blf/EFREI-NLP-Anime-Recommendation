"""
Module de chargement des données pour le système de recommandation d'anime.

Ce module définit une fonction simple pour charger le fichier CSV contenant
les informations sur les anime. L'objectif est de séparer cette étape
de prétraitement de manière claire, afin de pouvoir la réutiliser dans
différents scripts ou notebooks.
"""

import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """Charge le jeu de données à partir d'un fichier CSV.

    Parameters
    ----------
    filepath : str
        Chemin complet vers le fichier CSV à charger.

    Returns
    -------
    pandas.DataFrame
        DataFrame contenant l'intégralité des données du fichier.
    """
    return pd.read_csv(filepath)
