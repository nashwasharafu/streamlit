# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import hashlib
import time

st.set_page_config(
    page_title="Cinema Insights",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_data():
    
    movies = pd.DataFrame({
        'title': ['The Shawshank Redemption', 'The Godfather', 'The Dark Knight', 
                 'Pulp Fiction', 'Forrest Gump', 'Inception', 'The Matrix',
                 'Goodfellas', 'The Silence of the Lambs', 'Star Wars: Episode V'],
        'genre': ['Drama', 'Crime', 'Action', 'Crime', 'Drama', 'Action', 'Sci-Fi',
                 'Crime', 'Thriller', 'Sci-Fi'],
        'year': [1994, 1972, 2008, 1994, 1994, 2010, 1999, 1990, 1991, 1980],
        'rating': [9.3, 9.2, 9.0, 8.9, 8.8, 8.8, 8.7, 8.7, 8.6, 8.7],
        'votes': [2500000, 1700000, 2300000, 1900000, 1800000, 2100000, 1700000, 1000000, 1300000, 1200000],
        'director': ['Frank Darabont', 'Francis Ford Coppola', 'Christopher Nolan',
                    'Quentin Tarantino', 'Robert Zemeckis', 'Christopher Nolan',
                    'Lana Wachowski', 'Martin Scorsese', 'Jonathan Demme', 'Irvin Kershner'],
        'runtime': [142, 175, 152, 154, 142, 148, 136, 146, 118, 124],
        'revenue': [58.3, 245.1, 1004.6, 213.9, 677.9, 836.8, 463.5, 47.1, 272.7, 538.4]
    })
    return movies


def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text


def load_users():
    try:
        with open('users.pkl', 'rb') as f:
            users = pickle.load(f)
    except FileNotFoundError:
        users = {}
    return users

def save_users(users):
    with open('users.pkl', 'wb') as f:
        pickle.dump(users, f)

def get_recommendations(movies, favorite_genre, min_rating, years_range):
    filtered_movies = movies[
        (movies['genre'].str.contains(favorite_genre, case=False)) &
        (movies['rating'] >= min_rating) &
        (movies['year'] >= years_range[0]) &
        (movies['year'] <= years_range[1])
    ]
    return filtered_movies.sort_values('rating', ascending=False).head(5)


if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'users' not in st.session_state:
    st.session_state.users = load_users()


def login_page():
    st.title("🎬 Cinema Insights - Movie Data Analysis")
    st.markdown("### Login to access exclusive movie insights and recommendations")
    
    login_form = st.form(key='login_form')
    username = login_form.text_input("Username")
    password = login_form.text_input("Password", type='password')
    submit = login_form.form_submit_button("Login")
    
    if submit:
        if username in st.session_state.users:
            if check_hashes(password, st.session_state.users[username]):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success("Login successful!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Incorrect password")
        else:
            st.error("Username not found")
    
    st.markdown("---")
    st.markdown("Don't have an account? Register below.")
    
    register_form = st.form(key='register_form')
    new_username = register_form.text_input("New Username")
    new_password = register_form.text_input("New Password", type='password')
    register = register_form.form_submit_button("Register")
    
    if register:
        if new_username in st.session_state.users:
            st.error("Username already exists")
        else:
            st.session_state.users[new_username] = make_hashes(new_password)
            save_users(st.session_state.users)
            st.success("Registration successful! Please login.")


def main_app():
    st.sidebar.title(f"Welcome, {st.session_state.username}!")
    
    
    app_mode = st.sidebar.selectbox("Choose a section", 
                                   ["Dashboard", "Movie Explorer", "Recommendations", "Your Ratings"])
    

    movies = load_data()
    
    if app_mode == "Dashboard":
        st.title("🎬 Movie Data Dashboard")
        

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Movies", len(movies))
        col2.metric("Average Rating", f"{movies['rating'].mean():.1f}")
        col3.metric("Earliest Year", movies['year'].min())
        col4.metric("Latest Year", movies['year'].max())
        
  
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Movies by Genre")
            genre_counts = movies['genre'].value_counts()
            fig = px.pie(values=genre_counts.values, names=genre_counts.index)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Ratings Distribution")
            fig = px.histogram(movies, x='rating', nbins=10)
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Movies by Year and Rating")
        fig = px.scatter(movies, x='year', y='rating', size='votes', color='genre',
                         hover_name='title', log_x=False, size_max=60)
        st.plotly_chart(fig, use_container_width=True)
    
    elif app_mode == "Movie Explorer":
        st.title("🔍 Movie Explorer")
        

        col1, col2, col3 = st.columns(3)
        with col1:
            selected_genre = st.selectbox("Filter by Genre", ["All"] + list(movies['genre'].unique()))
        with col2:
            year_range = st.slider("Year Range", 
                                  min_value=int(movies['year'].min()), 
                                  max_value=int(movies['year'].max()),
                                  value=(int(movies['year'].min()), int(movies['year'].max())))
        with col3:
            min_rating = st.slider("Minimum Rating", 0.0, 10.0, 7.0, 0.1)
        
        
        filtered_movies = movies.copy()
        if selected_genre != "All":
            filtered_movies = filtered_movies[filtered_movies['genre'] == selected_genre]
        filtered_movies = filtered_movies[
            (filtered_movies['year'] >= year_range[0]) & 
            (filtered_movies['year'] <= year_range[1]) &
            (filtered_movies['rating'] >= min_rating)
        ]
        
        st.dataframe(filtered_movies, use_container_width=True)
    
    elif app_mode == "Recommendations":
        st.title("🎯 Personalized Recommendations")
        
        st.subheader("Tell us your preferences")
        col1, col2 = st.columns(2)
        
        with col1:
            favorite_genre = st.selectbox("Favorite Genre", movies['genre'].unique())
            min_rating = st.slider("Minimum Rating", 0.0, 10.0, 7.5, 0.1)
        
        with col2:
            years_range = st.slider("Release Year Range", 
                                   min_value=int(movies['year'].min()), 
                                   max_value=int(movies['year'].max()),
                                   value=(1990, 2010))
        
        if st.button("Get Recommendations"):
            recommendations = get_recommendations(movies, favorite_genre, min_rating, years_range)
            
            if len(recommendations) > 0:
                st.success("Here are movies you might enjoy:")
                for idx, row in recommendations.iterrows():
                    with st.expander(f"{row['title']} ({row['year']}) - Rating: {row['rating']}"):
                        st.write(f"**Genre:** {row['genre']}")
                        st.write(f"**Director:** {row['director']}")
                        st.write(f"**Runtime:** {row['runtime']} minutes")
                        st.write(f"**Votes:** {row['votes']:,}")
            else:
                st.warning("No movies match your criteria. Try broadening your search.")
    
    elif app_mode == "Your Ratings":
        st.title("⭐ Your Ratings")
        
        if 'user_ratings' not in st.session_state:
            st.session_state.user_ratings = {}
        
        st.subheader("Rate Movies")
        selected_movie = st.selectbox("Select a movie to rate", movies['title'])
        rating = st.slider("Your Rating", 1, 10, 5)
        review = st.text_area("Your Review (optional)")
        
        if st.button("Submit Rating"):
            st.session_state.user_ratings[selected_movie] = {
                'rating': rating,
                'review': review,
                'timestamp': time.time()
            }
            st.success(f"Thanks for rating {selected_movie}!")
        
        st.subheader("Your Rating History")
        if st.session_state.user_ratings:
            user_ratings_df = pd.DataFrame.from_dict(st.session_state.user_ratings, orient='index')
            user_ratings_df['movie'] = user_ratings_df.index
            st.dataframe(user_ratings_df[['movie', 'rating', 'review']], use_container_width=True)
        else:
            st.info("You haven't rated any movies yet.")
    

    st.sidebar.markdown("---")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()


if st.session_state.authenticated:
    main_app()
else:
    login_page()