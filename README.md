# Movie Recommendation System (Content-Based)

A content-based movie recommender using TF-IDF vectorization and cosine similarity.

## Files
- `movies.csv` — dataset of 93 movies (title, genres, overview, cast, director)
- `recommender.py` — core recommendation engine (CLI usable)
- `app.py` — Streamlit web UI
- `requirements.txt` — dependencies

## Setup
```bash
pip install -r requirements.txt
```

## Run from command line
```bash
python recommender.py "The Dark Knight" --top 5
```

## Run the web app
```bash
streamlit run app.py
```

## How it works
1. Combine each movie's genres, overview, cast, and director into one text string (`tags`), weighting genres and director more heavily.
2. Convert all `tags` into TF-IDF vectors (removes common English stop words, weights rare/distinctive terms higher).
3. Compute pairwise cosine similarity between every pair of movies.
4. For a queried movie, return the N movies with the highest similarity score (excluding itself).

## Extending this project
- Swap `movies.csv` for the full TMDB 5000 or MovieLens dataset for broader coverage.
- Add collaborative filtering (user ratings + matrix factorization) for a hybrid recommender.
- Deploy the Streamlit app to Streamlit Community Cloud or Hugging Face Spaces for a live demo link.
