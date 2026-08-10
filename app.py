"""
Streamlit UI for the Content-Based Movie Recommender.
Run with: streamlit run app.py
"""

import streamlit as st
from recommender import MovieRecommender

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

st.title("🎬 Movie Recommendation System")
st.caption("Content-based filtering using TF-IDF + Cosine Similarity")


@st.cache_resource
def load_engine():
    return MovieRecommender("movies.csv")


engine = load_engine()

movie_list = sorted(engine.df["title"].tolist())
selected_movie = st.selectbox("Pick a movie you like:", movie_list)
top_n = st.slider("Number of recommendations", 3, 10, 5)

if st.button("Recommend"):
    results, matched_title = engine.recommend(selected_movie, top_n)

    if results is None:
        st.error(matched_title)
    else:
        st.subheader(f"Because you watched: {matched_title}")
        for r in results:
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{r['title']}**")
                    st.caption(r["genres"])
                with col2:
                    st.metric("Match", f"{r['similarity'] * 100:.0f}%")

st.divider()
st.caption("Dataset: curated sample of 90 popular movies. Swap in the full TMDB 5000 dataset by replacing movies.csv with matching columns.")
