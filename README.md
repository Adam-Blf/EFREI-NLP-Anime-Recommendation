# Recommandation d'anime par similarité de synopsis

Projet de TP 2.3 NLP. Recommande des anime proches de favoris à partir des synopsis en TF‑IDF + similarité cosinus.

## Structure
- `data_load.py` charge le CSV `Anime.csv`
- `preprocess.py` prépare la colonne `Synopsis`
- `vectorize.py` calcule TF‑IDF et la matrice de similarité
- `recommend.py` renvoie les meilleurs titres
- `main.py` exemple simple en console
- `interactive.py` menu interactif en console
- `anime_recommendation_tp2_3.ipynb` notebook fourni
- `Anime.csv` dataset

## Installation rapide
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Lancer une démo
```bash
python main.py
```

## Mode interactif
```bash
python interactive.py
```

## Notes
- Les titres saisis doivent correspondre exactement à la colonne `Title`.
- Le vectoriseur retire les stop words anglais, aucun LLM externe n'est utilisé.
