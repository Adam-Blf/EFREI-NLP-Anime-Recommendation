import streamlit as st
import pandas as pd
from data_load import load_data
from preprocess import preprocess_data
from vectorize import vectorize_synopsis
from recommend import recommend_anime

# Configuration de la page
st.set_page_config(
    page_title="Anime Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS pour un look moderne
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #1a1a2e, #16213e);
        color: #ffffff;
    }
    .stButton>button {
        background-color: #e94560;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ff6b6b;
    }
    .anime-card {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        border-left: 5px solid #e94560;
    }
    h1 {
        color: #e94560;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h3 {
        color: #4db5ff;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process():
    # Chargement des données
    try:
        df = load_data('Anime.csv')
        df = preprocess_data(df)
        tfidf_matrix, cosine_sim, indices = vectorize_synopsis(df)
        return df, cosine_sim, indices
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None, None, None

def main():
    st.title("🎬 Anime Recommendation Engine")
    st.markdown("### Découvrez votre prochain anime favori grâce à l'IA")

    with st.spinner('Chargement du modèle NLP...'):
        df, cosine_sim, indices = load_and_process()

    if df is not None:
        # Sidebar pour les contrôles
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Sélection des animes favoris
            all_titles = sorted(df['Title'].unique().tolist())
            favorites = st.multiselect(
                "Quels animes avez-vous aimés ?",
                options=all_titles,
                placeholder="Rechercher un anime..."
            )
            
            top_n = st.slider("Nombre de recommandations", 1, 20, 5)
            
            st.markdown("---")
            st.markdown("### 📊 Stats du Dataset")
            st.metric("Total Animes", len(df))
            st.metric("Genres Uniques", len(set(",".join(df['Genre'].dropna()).split(","))))

        # Zone principale
        if favorites:
            if st.button("🔍 Générer les recommandations"):
                recommendations = recommend_anime(favorites, top_n, df, cosine_sim, indices)
                
                st.markdown(f"### 🎯 Recommandations basées sur : {', '.join(favorites)}")
                
                # Affichage en grille
                cols = st.columns(2)
                for idx, row in recommendations.iterrows():
                    with cols[idx % 2]:
                        st.markdown(f"""
                        <div class="anime-card">
                            <h3>{row['Title']}</h3>
                            <p><strong>Genre:</strong> {row['Genre']}</p>
                            <p><strong>Type:</strong> {row['Type']} | <strong>Episodes:</strong> {row['Episodes']}</p>
                            <p><em>{row['Synopsis'][:200]}...</em></p>
                        </div>
                        """, unsafe_allow_html=True)
                        with st.expander("Voir le synopsis complet"):
                            st.write(row['Synopsis'])
        else:
            st.info("👈 Commencez par sélectionner vos animes préférés dans la barre latérale !")
            
            # Affichage aléatoire pour inspiration
            st.markdown("### 🎲 Quelques suggestions populaires")
            sample = df.sample(3)
            cols = st.columns(3)
            for i, (_, row) in enumerate(sample.iterrows()):
                with cols[i]:
                    st.markdown(f"**{row['Title']}**")
                    st.caption(row['Genre'])

if __name__ == "__main__":
    main()
