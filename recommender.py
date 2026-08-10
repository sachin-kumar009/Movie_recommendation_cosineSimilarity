"""
Content-Based Movie Recommendation System
--------------------------------------------
Approach: TF-IDF vectorization of movie metadata (genres + overview + cast +
director) followed by cosine similarity to find the most similar movies to
a given title.

Usage:
    python recommender.py "The Dark Knight"
    python recommender.py "Inception" --top 10
"""

import sys
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self._prepare_data()
        self._vectorize()

    def _prepare_data(self):
        """Combine relevant text columns into one 'tags' column."""
        for col in ["genres", "overview", "cast", "director"]:
            self.df[col] = self.df[col].fillna("")

        # Give genres and director extra weight by repeating them —
        # this nudges the vectorizer to treat them as stronger signals.
        self.df["tags"] = (
            (self.df["genres"] + " ") * 2
            + self.df["overview"] + " "
            + self.df["cast"] + " "
            + (self.df["director"] + " ") * 2
        ).str.lower()

        # Normalize title for lookups (case-insensitive, trimmed)
        self.df["title_lower"] = self.df["title"].str.lower().str.strip()

    def _vectorize(self):
        """Fit TF-IDF on the tags column and precompute the similarity matrix."""
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["tags"])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def _find_index(self, title: str):
        title_lower = title.lower().strip()
        matches = self.df[self.df["title_lower"] == title_lower]
        if not matches.empty:
            return matches.index[0]

        # fallback: partial match
        partial = self.df[self.df["title_lower"].str.contains(title_lower, na=False)]
        if not partial.empty:
            return partial.index[0]

        return None

    def recommend(self, title: str, top_n: int = 5):
        idx = self._find_index(title)
        if idx is None:
            return None, f"Movie '{title}' not found in the dataset."

        scores = list(enumerate(self.similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        # skip index 0 result — it's always the movie itself (similarity = 1.0)
        scores = [s for s in scores if s[0] != idx][:top_n]

        results = []
        for i, score in scores:
            row = self.df.iloc[i]
            results.append({
                "title": row["title"],
                "genres": row["genres"],
                "similarity": round(float(score), 3),
            })

        matched_title = self.df.iloc[idx]["title"]
        return results, matched_title


def main():
    parser = argparse.ArgumentParser(description="Content-based movie recommender")
    parser.add_argument("title", type=str, help="Movie title to get recommendations for")
    parser.add_argument("--top", type=int, default=5, help="Number of recommendations")
    parser.add_argument("--csv", type=str, default="movies.csv", help="Path to movies CSV")
    args = parser.parse_args()

    engine = MovieRecommender(args.csv)
    results, matched_title = engine.recommend(args.title, args.top)

    if results is None:
        print(matched_title)  # error message
        sys.exit(1)

    print(f"\nBecause you watched: {matched_title}\n")
    print(f"{'Rank':<5}{'Movie':<45}{'Genres':<35}{'Similarity'}")
    print("-" * 100)
    for i, r in enumerate(results, 1):
        print(f"{i:<5}{r['title']:<45}{r['genres']:<35}{r['similarity']}")


if __name__ == "__main__":
    main()
