# EFREI NLP - Système de Recommandation d'Anime / Anime Recommendation System

[🇫🇷 Version Française](#version-française) | [🇬🇧 English Version](#english-version)

---

## <a name="version-française"></a>🇫🇷 Version Française

Moteur de recommandation basé sur **TF-IDF + similarité cosinus** qui suggère des animes similaires à vos favoris en utilisant le traitement du langage naturel sur les synopsis. Projet NLP pratique démontrant la vectorisation de texte, les métriques de similarité et les outils CLI interactifs.

### ✨ Fonctionnalités

- 🎯 **Filtrage Basé sur le Contenu** : recommandation d'animes basée sur la similarité des synopsis
- 📊 **Vectorisation TF-IDF** : extraction de features significatives des descriptions textuelles
- 🔍 **Similarité Cosinus** : calcul efficace des scores de similarité par paires
- 💬 **CLI Interactif** : exploration des recommandations avec autocomplétion et historique
- 📓 **Notebook Jupyter** : analyse et visualisation du pipeline de recommandation
- 🚀 **Léger** : aucune API externe ni framework ML complexe

### 🛠️ Stack Technologique

| Composant | Technologie | Objectif |
|-----------|-------------|----------|
| **Moteur NLP** | scikit-learn (TfidfVectorizer) | Extraction de features textuelles |
| **Traitement Données** | pandas | Gestion CSV et opérations DataFrame |
| **Métrique Similarité** | cosine_similarity (sklearn) | Calcul des scores de recommandation |
| **Shell Interactif** | Python REPL | Requêtes de recommandation conviviales |
| **Analyse** | Jupyter Notebook | Analyse exploratoire et visualisation |
| **Langage** | Python 3.9+ | Logique applicative |

### 📁 Structure du Projet

```
EFREI-NLP-Anime-Recommendation/
├── data_load.py           # Utilitaires de chargement CSV
├── preprocess.py          # Nettoyage et normalisation de texte
├── vectorize.py           # Calcul TF-IDF et matrice de similarité
├── recommend.py           # Logique de recommandation
├── main.py                # Démo console simple
├── interactive.py         # CLI interactif avec menu
├── requirements.txt       # Dépendances Python
├── data/
│   ├── Anime.csv          # Dataset (titre, synopsis, genre, etc.)
│   └── anime_recommendation_tp2_3.ipynb  # Notebook d'analyse
└── README.md
```

### 🚀 Démarrage Rapide

#### Prérequis

- Python 3.9 ou supérieur
- Gestionnaire de paquets pip

#### Installation

```bash
# Clonez le dépôt
cd EFREI-NLP-Anime-Recommendation

# Créez un environnement virtuel (recommandé)
python -m venv .venv

# Activez l'environnement
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Installez les dépendances
pip install -r requirements.txt
```

#### Lancer la Démo Simple

```bash
python main.py
```

#### Mode Interactif

```bash
python interactive.py
```

### 🎯 Fonctionnement

1. **Prétraitement** : lowercasing, suppression ponctuation, stop words anglais
2. **Vectorisation** : TF-IDF avec unigrammes + bigrammes
3. **Calcul Similarité** : matrice de similarité cosinus
4. **Recommandation** : extraction des N meilleurs scores

### 🗺️ Feuille de Route

- [ ] Correspondance floue des titres
- [ ] Filtrage hybride (content + collaboratif)
- [ ] Pondération par genre
- [ ] Interface web (Flask/Streamlit)
- [ ] Persistance de modèle automatique
- [ ] Support multi-langues

---

## <a name="english-version"></a>🇬🇧 English Version

A **TF-IDF + cosine similarity** based recommendation engine that suggests anime titles similar to your favorites using natural language processing on synopses. Practical NLP project demonstrating text vectorization, similarity metrics, and interactive CLI tools.

### ✨ Features

- 🎯 **Content-Based Filtering**: recommend anime based on synopsis similarity
- 📊 **TF-IDF Vectorization**: extract meaningful features from text descriptions
- 🔍 **Cosine Similarity**: compute pairwise similarity scores efficiently
- 💬 **Interactive CLI**: explore recommendations with autocomplete and history
- 📓 **Jupyter Notebook**: analyze and visualize the recommendation pipeline
- 🚀 **Lightweight**: no external APIs or complex ML frameworks

### 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **NLP Engine** | scikit-learn (TfidfVectorizer) | Text feature extraction |
| **Data Processing** | pandas | CSV handling and DataFrame operations |
| **Similarity Metric** | cosine_similarity (sklearn) | Compute recommendation scores |
| **Interactive Shell** | Python REPL | User-friendly recommendation queries |
| **Analysis** | Jupyter Notebook | Exploratory data analysis and visualization |
| **Language** | Python 3.9+ | Core application logic |

### 📁 Project Structure

```
EFREI-NLP-Anime-Recommendation/
├── data_load.py           # CSV loading utilities
├── preprocess.py          # Text cleaning and normalization
├── vectorize.py           # TF-IDF computation and similarity matrix
├── recommend.py           # Recommendation logic
├── main.py                # Simple console demo
├── interactive.py         # Interactive CLI with menu
├── requirements.txt       # Python dependencies
├── data/
│   ├── Anime.csv          # Dataset (title, synopsis, genre, etc.)
│   └── anime_recommendation_tp2_3.ipynb  # Analysis notebook
└── README.md
```

### 🚀 Quick Start

#### Prerequisites

- Python 3.9 or higher
- pip package manager

#### Installation

```bash
# Clone the repository
cd EFREI-NLP-Anime-Recommendation

# Create virtual environment (recommended)
python -m venv .venv

# Activate environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Run Simple Demo

```bash
python main.py
```

#### Interactive Mode

```bash
python interactive.py
```

### 🎯 How It Works

1. **Preprocessing**: lowercase, punctuation removal, English stop words
2. **Vectorization**: TF-IDF with unigrams + bigrams
3. **Similarity Computation**: cosine similarity matrix
4. **Recommendation**: extract top N scores

### 🗺️ Roadmap

- [ ] Fuzzy title matching
- [ ] Hybrid filtering (content + collaborative)
- [ ] Genre weighting
- [ ] Web interface (Flask/Streamlit)
- [ ] Automatic model persistence
- [ ] Multi-language support

### 📄 License

This project is open source. See LICENSE file for details.

---

**Project**: EFREI NLP TP 2.3  
**Author**: Adam Beloucif  
**Repository**: [github.com/Adam-Blf/EFREI-NLP-Anime-Recommendation](https://github.com/Adam-Blf/EFREI-NLP-Anime-Recommendation)

For issues or feature requests, open an issue on GitHub.
