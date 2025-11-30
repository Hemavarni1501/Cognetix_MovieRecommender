import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os

# --- 1. Load Data and Pre-calculate Similarity (Caching and Sampling) ---
# We use @st.cache_data to load the data and calculate the large matrix only once, 
# but we sample the movies to prevent MemoryError.
@st.cache_data
def load_data_and_calculate_similarity():
    # Load movies data (required for titles and genres)
    MOVIE_FILE = 'movies.csv'
    
    if not os.path.exists(MOVIE_FILE):
        st.error(f"Error: {MOVIE_FILE} not found. Ensure it is in the directory.")
        return None, None, None

    try:
        movies_df = pd.read_csv(MOVIE_FILE)
    except Exception as e:
        st.error(f"Error loading movies.csv: {e}")
        return None, None, None

    # *** FIX: Sample a small number of movies to reduce matrix size ***
    # This prevents the MemoryError caused by the massive N x N similarity matrix.
    SAMPLE_MOVIES = 5000 
    if len(movies_df) > SAMPLE_MOVIES:
        movies_df = movies_df.sample(n=SAMPLE_MOVIES, random_state=42).reset_index(drop=True)
    
    st.sidebar.info(f"Loaded and sampled {len(movies_df)} movies for the recommender matrix.")
    # *******************************************************************
    
    # Prepare for Content-Based Filtering
    movies_df['genres'] = movies_df['genres'].fillna('')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres'])
    
    # Calculate Cosine Similarity
    # This is the most memory-intensive step
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Create title-to-index mapping
    indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()
    
    return movies_df, cosine_sim, indices

# Load the resources globally
movies_df, cosine_sim, indices = load_data_and_calculate_similarity()

# --- 2. Recommendation Function ---
def get_recommendations(title, cosine_sim=cosine_sim, movies_df=movies_df, indices=indices):
    """Generates the top 10 recommendations based on Cosine Similarity."""
    
    if movies_df is None:
        return pd.DataFrame() 

    # Find the index of the selected movie
    if title not in indices:
        # This should ideally not happen if using the dropdown, but is a safety check.
        return pd.DataFrame()
        
    idx = indices[title]

    # Get the pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11] # Top 10 recommendations (excluding the movie itself at index 0)

    # Get the movie indices and corresponding similarity scores
    movie_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]

    recommendations = movies_df['title'].iloc[movie_indices]
    
    return pd.DataFrame({
        'Rank': range(1, 11), 
        'Movie Title': recommendations.values, 
        'Similarity Score': [f"{s:.4f}" for s in scores]
    })

# --- 3. Streamlit UI ---
st.title('🎬 Movie Recommender System (Content-Based)')
st.markdown("Developed using **Cosine Similarity** based on movie genres from a sample of the MovieLens dataset.")

if movies_df is not None:
    # Get a list of movie titles for the dropdown
    movie_titles = sorted(movies_df['title'].tolist())
    
    # User selects a movie
    selected_movie = st.selectbox(
        'Select a movie you like:',
        movie_titles,
        # Default selection is 'Toy Story (1995)' if it exists in the sampled data, otherwise the first movie.
        index=movie_titles.index('Toy Story (1995)') if 'Toy Story (1995)' in movie_titles else 0
    )

    if st.button('Get Recommendations'):
        with st.spinner('Generating recommendations...'):
            results = get_recommendations(selected_movie)
            
            if not results.empty:
                st.subheader(f'Top 10 Recommendations for "{selected_movie}"')
                st.dataframe(results.set_index('Rank'), use_container_width=True)
            
            # Display genre information for context
            original_genres = movies_df[movies_df['title'] == selected_movie]['genres'].iloc[0]
            st.info(f"Original Movie Genres: **{original_genres}**")

else:
    st.error("Cannot proceed. Data loading or similarity matrix calculation failed.")