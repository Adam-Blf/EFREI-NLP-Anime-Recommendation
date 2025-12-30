import streamlit as st
import pandas as pd
import os
from data_load import load_data
from preprocess import preprocess_data
from vectorize import vectorize_synopsis
from recommend import recommend_anime

# Import LLM engine
try:
    from llm_engine import LLMEngine, load_embeddings_cache, create_embeddings_cache
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

st.set_page_config(page_title="Anime Recommender", page_icon="🎬", layout="wide")

st.markdown("""
<style>
.stApp { background: linear-gradient(to right, #1a1a2e, #16213e); color: #ffffff; }
.stButton>button { background-color: #e94560; color: white; border-radius: 20px; }
.anime-card { background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; margin-bottom: 20px; border-left: 5px solid #e94560; }
h1 { color: #e94560; }
h3 { color: #4db5ff; }
.llm-explanation { background: rgba(78,205,196,0.2); padding: 10px; border-radius: 10px; margin-top: 10px; font-style: italic; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process():
    try:
        df = load_data('Anime.csv')
        df = preprocess_data(df)
        tfidf_matrix, cosine_sim, indices = vectorize_synopsis(df)
        return df, cosine_sim, indices
    except Exception as e:
        st.error(f"Erreur: {e}")
        return None, None, None

@st.cache_resource
def load_llm_engine():
    if LLM_AVAILABLE:
        return LLMEngine()
    return None

@st.cache_data
def get_semantic_embeddings(_df):
    if not LLM_AVAILABLE:
        return None
    cache_path = "embeddings_cache.npy"
    embeddings = load_embeddings_cache(cache_path)
    if embeddings is None:
        embeddings = create_embeddings_cache(_df, cache_path)
    return embeddings

def main():
    st.title("🎬 Anime Recommendation Engine")
    st.markdown("### Découvrez votre prochain anime favori grâce à l'IA")

    with st.spinner('Chargement...'):
        df, cosine_sim, indices = load_and_process()
        llm = load_llm_engine()
        embeddings = get_semantic_embeddings(df) if llm else None

    if df is None:
        return

    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Mode de recherche
        search_mode = st.radio(
            "Mode de recherche",
            ["🎯 Par favoris", "💬 Langage naturel"] if LLM_AVAILABLE and embeddings is not None else ["🎯 Par favoris"]
        )
        
        if search_mode == "🎯 Par favoris":
            all_titles = sorted(df['Title'].unique().tolist())
            favorites = st.multiselect("Vos animes préférés", options=all_titles)
            top_n = st.slider("Nombre de recommandations", 1, 20, 5)
        else:
            favorites = []
            top_n = st.slider("Nombre de résultats", 1, 20, 10)
        
        use_llm_explain = st.checkbox("✨ Explications IA", value=True) if LLM_AVAILABLE else False
        
        st.markdown("---")
        st.metric("Total Animes", len(df))
        if LLM_AVAILABLE:
            st.success("🤖 LLM: Actif")
        else:
            st.warning("🤖 LLM: pip install sentence-transformers")

    # Mode langage naturel
    if search_mode == "💬 Langage naturel":
        query = st.text_input("🔍 Décrivez l'anime que vous cherchez", 
                             placeholder="Ex: Un anime sombre avec des combats épiques comme Attack on Titan")
        
        if query and st.button("🚀 Rechercher"):
            with st.spinner("Recherche sémantique..."):
                results = llm.semantic_search(query, df, embeddings, top_k=top_n)
            
            st.markdown(f"### 🎯 Résultats pour: *{query}*")
            cols = st.columns(2)
            for idx, (_, row) in enumerate(results.iterrows()):
                with cols[idx % 2]:
                    st.markdown(f"""
                    <div class="anime-card">
                        <h3>{row['Title']}</h3>
                        <p><strong>Score:</strong> {row['similarity_score']:.2%}</p>
                        <p><strong>Genre:</strong> {row.get('Genre', 'N/A')}</p>
                        <p><em>{row['Synopsis'][:200]}...</em></p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Mode favoris classique
    elif favorites:
        if st.button("🔍 Générer les recommandations"):
            recommendations = recommend_anime(favorites, top_n, df, cosine_sim, indices)
            
            st.markdown(f"### 🎯 Basé sur: {', '.join(favorites)}")
            cols = st.columns(2)
            
            for idx, (_, row) in enumerate(recommendations.iterrows()):
                with cols[idx % 2]:
                    explanation = ""
                    if use_llm_explain and llm:
                        with st.spinner(f"Analyse de {row['Title']}..."):
                            explanation = llm.explain_recommendation(
                                row['Title'], row['Synopsis'], favorites
                            )
                    
                    st.markdown(f"""
                    <div class="anime-card">
                        <h3>{row['Title']}</h3>
                        <p><strong>Genre:</strong> {row.get('Genre', 'N/A')}</p>
                        <p><em>{row['Synopsis'][:200]}...</em></p>
                        {f'<div class="llm-explanation">💡 {explanation}</div>' if explanation else ''}
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("👈 Sélectionnez vos animes préférés ou utilisez la recherche en langage naturel!")
        st.markdown("### 🎲 Suggestions")
        for _, row in df.sample(3).iterrows():
            st.markdown(f"**{row['Title']}** - {row.get('Genre', '')}")

if __name__ == "__main__":
    main()
