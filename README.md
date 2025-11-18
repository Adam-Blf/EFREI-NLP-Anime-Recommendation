# EFREI NLP - Anime Recommendation System

A **TF-IDF + cosine similarity** based recommendation engine that suggests anime titles similar to your favorites using natural language processing on synopses. Built as a practical NLP project demonstrating text vectorization, similarity metrics, and interactive CLI tools.

## ✨ Features

- 🎯 **Content-Based Filtering**: recommend anime based on synopsis similarity
- 📊 **TF-IDF Vectorization**: extract meaningful features from text descriptions
- 🔍 **Cosine Similarity**: compute pairwise similarity scores efficiently
- 💬 **Interactive CLI**: explore recommendations with autocomplete and history
- 📓 **Jupyter Notebook**: analyze and visualize the recommendation pipeline
- 🚀 **Lightweight**: no external APIs or complex ML frameworks

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **NLP Engine** | scikit-learn (TfidfVectorizer) | Text feature extraction |
| **Data Processing** | pandas | CSV handling and DataFrame operations |
| **Similarity Metric** | cosine_similarity (sklearn) | Compute recommendation scores |
| **Interactive Shell** | Python REPL | User-friendly recommendation queries |
| **Analysis** | Jupyter Notebook | Exploratory data analysis and visualization |
| **Language** | Python 3.9+ | Core application logic |

## 📁 Project Structure

```text
EFREI-NLP-Anime-Recommendation/
├── data_load.py           # CSV loading utilities
├── preprocess.py          # Text cleaning and normalization
├── vectorize.py           # TF-IDF computation and similarity matrix
├── recommend.py           # Recommendation logic
├── main.py                # Simple console demo
├── interactive.py         # Interactive CLI with menu
├── Anime.csv              # Dataset (title, synopsis, genre, etc.)
├── anime_recommendation_tp2_3.ipynb  # Jupyter analysis notebook
├── requirements.txt       # Python dependencies
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/Adam-Blf/EFREI-NLP-Anime-Recommendation.git
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

### Run Simple Demo

```bash
python main.py
```

Example output:

```text
Recommendations for "Death Note":
1. Code Geass: Hangyaku no Lelouch (Score: 0.42)
2. Monster (Score: 0.38)
3. Psycho-Pass (Score: 0.35)
...
```

### Run Interactive Mode

```bash
python interactive.py
```

Features:

- Search anime by title (exact match required)
- Adjust number of recommendations (default: 10)
- View anime details (synopsis, genre)
- Command history and session persistence

## 📋 Data Schema

### Anime.csv Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `Title` | Text | Anime title (English/Romaji) | "Steins;Gate" |
| `Synopsis` | Text | Plot summary (primary feature) | "A group of friends..." |
| `Genre` | Text | Comma-separated genres | "Sci-Fi, Thriller" |
| `Rating` | Float | Average user rating | 8.81 |
| `Episodes` | Integer | Total episode count | 24 |

> **Note**: Recommendations are based solely on `Synopsis` similarity. Genre and rating are not factored into scoring.

## 🎯 How It Works

### 1. Text Preprocessing (`preprocess.py`)

- Lowercase conversion
- Punctuation removal
- English stop words removal (NLTK stopwords)
- Whitespace normalization

### 2. Vectorization (`vectorize.py`)

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)  # Unigrams + bigrams
)
tfidf_matrix = vectorizer.fit_transform(synopses)
```

### 3. Similarity Computation

```python
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
```

### 4. Recommendation Retrieval (`recommend.py`)

- Find index of input anime title
- Extract similarity scores for that title
- Sort by score (descending) and return top N

## ⚙️ Configuration

### Adjust Recommendation Count

Edit `main.py` or pass parameter in `interactive.py`:

```python
recommendations = get_recommendations("Naruto", top_n=20)
```

### Modify TF-IDF Parameters

In `vectorize.py`:

```python
vectorizer = TfidfVectorizer(
    max_features=10000,      # Increase vocabulary size
    min_df=2,                 # Ignore rare terms
    max_df=0.8,               # Ignore overly common terms
    ngram_range=(1, 3)        # Add trigrams
)
```

### Cache Similarity Matrix

For faster repeated queries, serialize the matrix:

```python
import joblib

# After computing similarity_matrix
joblib.dump(similarity_matrix, 'similarity_cache.pkl')

# Load in subsequent runs
similarity_matrix = joblib.load('similarity_cache.pkl')
```

## 🔒 Best Practices

- **Title Matching**: titles must match dataset exactly (case-sensitive); consider adding fuzzy matching (e.g., `fuzzywuzzy`)
- **Dataset Updates**: re-run vectorization pipeline if `Anime.csv` is modified
- **Performance**: for large datasets (10,000+ anime), consider sparse matrix operations and incremental updates
- **Quality**: recommendation quality depends on synopsis richness; short/generic descriptions yield poor results

## 🧪 Testing

### Validate Recommendations

Test with well-known anime:

```bash
# Expected: similar psychological thrillers
python -c "from recommend import recommend_anime; from data_load import load_data; from preprocess import preprocess_synopsis; from vectorize import vectorize_synopsis; df=load_data('Anime.csv'); df=preprocess_synopsis(df); _,cs,idx=vectorize_synopsis(df); print(recommend_anime(['Death Note'], 5, df, cs, idx))"

# Expected: shonen action anime
python -c "from recommend import recommend_anime; from data_load import load_data; from preprocess import preprocess_synopsis; from vectorize import vectorize_synopsis; df=load_data('Anime.csv'); df=preprocess_synopsis(df); _,cs,idx=vectorize_synopsis(df); print(recommend_anime(['One Piece'], 5, df, cs, idx))"
```

### Jupyter Notebook

Open `anime_recommendation_tp2_3.ipynb` to:

- Explore dataset statistics
- Visualize TF-IDF feature importance
- Analyze similarity score distributions
- Compare different vectorization strategies

```bash
jupyter notebook anime_recommendation_tp2_3.ipynb
```

## 🗺️ Roadmap

- [ ] **Fuzzy Title Matching**: handle typos and partial matches
- [ ] **Hybrid Filtering**: combine content-based with collaborative filtering (user ratings)
- [ ] **Genre Weighting**: boost scores for genre overlap
- [ ] **Web Interface**: Flask/Streamlit UI for non-technical users
- [ ] **Model Persistence**: save/load vectorizer and matrix automatically
- [ ] **Multi-Language Support**: process synopses in Japanese, French, etc.
- [ ] **Performance Profiling**: optimize for 50,000+ anime datasets

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

**Project**: EFREI NLP TP 2.3  
**Author**: Adam Beloucif  
**Repository**: [github.com/Adam-Blf/EFREI-NLP-Anime-Recommendation](https://github.com/Adam-Blf/EFREI-NLP-Anime-Recommendation)

For issues or feature requests, open an issue on GitHub.
