import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os

# --- 1. Load Data (Adjust paths if necessary) ---
# NOTE: Using a subset of files from the MovieLens dataset (e.g., 20M dataset uses 'movies.csv', 'ratings.csv')
MOVIE_FILE = 'movies.csv' 
RATING_FILE = 'ratings.csv' 
SAMPLE_SIZE = 100000 # Limit ratings for performance on large datasets

if not all(os.path.exists(f) for f in [MOVIE_FILE, RATING_FILE]):
    print("Error: Required files ('movies.csv' and 'ratings.csv') not found.")
    print("Please download and extract the MovieLens dataset into your working directory.")
    sys.exit()

# Load movies metadata
movies_df = pd.read_csv(MOVIE_FILE)
# Load a sample of ratings data for faster processing
ratings_df = pd.read_csv(RATING_FILE).sample(n=SAMPLE_SIZE, random_state=42)

print(f"Movies loaded: {movies_df.shape}")
print(f"Sample Ratings loaded: {ratings_df.shape}")

# --- 2. Data Cleaning and Preprocessing ---
# Remove duplicates (if a user somehow rated the same movie multiple times)
ratings_df.drop_duplicates(subset=['userId', 'movieId'], keep='first', inplace=True)

# Merge movie titles/genres with ratings data
merged_df = pd.merge(ratings_df, movies_df, on='movieId')
print(f"Data merged: {merged_df.shape}")

# --- 3. Content-Based Filtering Preparation (using Genres) ---

# Convert the pipe-separated genre string into a suitable format.
# We will use TFIDF to create a feature matrix where each genre is a feature.
# NOTE: The TFIDF is applied to the genres, which is equivalent to One-Hot Encoding here,
# but can handle weighted importance if genres were more complex.

tfidf = TfidfVectorizer(stop_words='english')

# Apply TFIDF to the genre strings. NaN values are replaced with an empty string.
tfidf_matrix = tfidf.fit_transform(movies_df['genres'].fillna(''))

print(f"TFIDF Matrix shape (Movies x Genres): {tfidf_matrix.shape}")

# --- 4. Calculate Cosine Similarity ---
# Calculate the cosine similarity matrix (a measure of distance/similarity between every pair of movies)
print("\nCalculating Cosine Similarity matrix (This may take a moment)...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a mapping of movie titles to their index in the DataFrame
indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()

# --- 5. Recommendation Function ---
def get_recommendations(title, cosine_sim=cosine_sim, movies_df=movies_df, indices=indices):
    # Get the index of the movie that matches the title
    if title not in indices:
        print(f"Movie '{title}' not found in the index. Skipping recommendation.")
        return pd.DataFrame() # Return empty DataFrame
        
    idx = indices[title]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores (descending)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 11 most similar movies (excluding the movie itself at index 0)
    sim_scores = sim_scores[1:11] 

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    recommendations = movies_df['title'].iloc[movie_indices]
    
    # Return recommendations along with their similarity scores (optional but insightful)
    scores = [i[1] for i in sim_scores]
    return pd.DataFrame({'Movie Title': recommendations.values, 'Similarity Score': scores})


# --- 6. Generate Recommendations Example ---
TEST_MOVIE = 'Toy Story (1995)'
print(f"\n--- Recommendations for: {TEST_MOVIE} ---")
recommendations = get_recommendations(TEST_MOVIE)

if not recommendations.empty:
    print(recommendations.to_markdown(index=False))
else:
    # Try another popular title if the first one failed (common issue with large datasets and titles)
    TEST_MOVIE = 'Braveheart (1995)'
    recommendations = get_recommendations(TEST_MOVIE)
    if not recommendations.empty:
        print(f"\n--- Recommendations for: {TEST_MOVIE} (Fallback) ---")
        print(recommendations.to_markdown(index=False))

# --- 7. Visualization (EDA) ---
# Extract and count all genres
all_genres = movies_df['genres'].str.split('|', expand=True).stack()
genre_counts = all_genres.value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=genre_counts.index, y=genre_counts.values, palette="viridis")
plt.title('Top 10 Most Popular Genres (by Movie Count)')
plt.xlabel('Genre')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('popular_genres_bar_chart.png')
print("\nVisualization of popular genres saved as 'popular_genres_bar_chart.png'.")

# --- 8. Evaluation Discussion ---
print("\n--- Evaluation Discussion ---")
print("Content-Based Filtering is primarily evaluated subjectively (e.g., 'Do the recommendations make sense?').")
print("For quantitative evaluation, metrics like **Precision@K** (what percentage of the top K recommendations are relevant to the user?) are preferred over RMSE, which is suitable for predicting specific ratings.")