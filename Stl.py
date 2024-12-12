def main():
    st.title('Movie Recommendation System')
    
    movies_data, similarity = load_data()
    movie_name = st.text_input('Enter your favourite movie name:')
    
    if movie_name:
        recommendations = get_movie_recommendations(movie_name, movies_data, similarity)
        if recommendations:
            st.write('Movies suggested for you:')
            for i, title in enumerate(recommendations, start=1):
                st.write(f'{i}. {title}')
        else:
            st.write("Sorry, no close match found for your movie.")
