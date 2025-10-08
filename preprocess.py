"""
Module de prétraitement des synopsis pour le système de recommandation d'anime.

Cette étape consiste à préparer la colonne `Synopsis` pour la vectorisation.
On se contente ici de remplacer les valeurs manquantes par des chaînes vides,
car le nettoyage détaillé (mise en minuscules, suppression de la ponctuation,
filtrage des stop words) est géré par le `TfidfVectorizer` dans la phase de
vectorisation.
"""

import pandas as pd


def preprocess_synopsis(df: pd.DataFrame) -> pd.DataFrame:
    """Prépare la colonne Synopsis en remplaçant les NaN par des chaînes vides.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame contenant la colonne `Synopsis` à nettoyer.

    Returns
    -------
    pandas.DataFrame
        Le DataFrame avec la colonne `Synopsis` nettoyée.
    """
    df = df.copy()
    df['Synopsis'] = df['Synopsis'].fillna('')
    return df
