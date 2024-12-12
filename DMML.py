import streamlit as st
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to load and process the movie data
def load_data():
    # Load data (ensure the correct path to your CSV file)
    movies_data = pd.read_csv('movies.csv')
    
    # Select relevant features for recommendation
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    
    # Replace null values with empty strings
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')
    
    # Combine all selected features into one string
    combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']
    
    # Convert the text data to feature vectors using TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    
    # Get the cosine similarity matrix
    similarity = cosine_similarity(feature_vectors)
    
    return movies_data, similarity

# Function to get movie recommendations
def get_movie_recommendations(movie_name, movies_data, similarity):
    # List of all movie titles
    list_of_all_titles = movies_data['title'].tolist()
    
    # Find the close match for the movie title
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    
    if len(find_close_match) == 0:
        return []

    close_match = find_close_match[0]
    
    # Get the index of the movie
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    
    # Get the similarity score for the movie
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    
    # Sort the movies based on their similarity score
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    
    # Get the top 30 similar movies
    recommended_movies = []
    for i, movie in enumerate(sorted_similar_movies):
        index = movie[0]
        title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        if i < 30:
            recommended_movies.append(title_from_index)
    
    return recommended_movies

# Streamlit UI
def main():
    st.title('Movie Recommendation System')
    
    # Load data and similarity matrix
    movies_data, similarity = load_data()
    
    # User input for movie name
    movie_name = st.text_input('Enter your favourite movie name:')
    
    if movie_name:
        # Get recommendations
        recommendations = get_movie_recommendations(movie_name, movies_data, similarity)
        
        if recommendations:
            st.write('Movies suggested for you:')
            for i, title in enumerate(recommendations, start=1):
                st.write(f'{i}. {title}')
        else:
            st.write("Sorry, no close match found for your movie.")

if __name__ == "__main__":
    main()
