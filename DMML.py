import streamlit as st
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess data
@st.cache
def load_data():
    movies_data = pd.read_csv('movies.csv')
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')
    combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']
    return movies_data, combined_features

movies_data, combined_features = load_data()

# Generate feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Compute similarity matrix
similarity = cosine_similarity(feature_vectors)

# Streamlit app
st.title('Movie Recommendation System')
st.markdown("Enter your favorite movie and get recommendations based on its content!")

# User input
movie_name = st.text_input("Enter your favorite movie:")

if movie_name:
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    if find_close_match:
        close_match = find_close_match[0]
        st.write(f"We found a close match: **{close_match}**")

        index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
        similarity_score = list(enumerate(similarity[index_of_the_movie]))
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        st.subheader("Movies suggested for you:")
        for i, movie in enumerate(sorted_similar_movies[1:31], start=1):  # Skip the first movie as it is the input movie
            index = movie[0]
            title_from_index = movies_data[movies_data.index == index]['title'].values[0]
            st.write(f"{i}. {title_from_index}")
    else:
        st.error("No matching movie found. Please try a different title.")