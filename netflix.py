# Netflix Movie & TV Show Analysis with Recommendation and Streamlit Dashboard

import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load dataset
df = pd.read_csv('netflix_titles.csv')

# Data Cleaning
df['date_added'] = pd.to_datetime(df['date_added'])
df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month
df['rating'].fillna('Not Rated', inplace=True)
df['country'].fillna('Unknown', inplace=True)
df['description'].fillna('', inplace=True)

# --- Streamlit Dashboard ---
st.title("📺 Netflix Movies & TV Shows Analysis")

# Content Type Distribution
st.subheader("Distribution of Content Types")
type_counts = df['type'].value_counts()
st.bar_chart(type_counts)

# Top Genres
st.subheader("Top 10 Genres on Netflix")
genres = df['listed_in'].str.split(',').explode().str.strip().value_counts().head(10)
st.bar_chart(genres)

# Country Distribution
st.subheader("Top 10 Content-Producing Countries")
top_countries = df['country'].value_counts().head(10)
st.bar_chart(top_countries)

# Rating Distribution
st.subheader("Top Ratings on Netflix")
top_ratings = df['rating'].value_counts().head(10)
st.bar_chart(top_ratings)

# --- Recommendation System ---
st.subheader("🔍 Content-Based Movie Recommendation")
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Build title to index mapping
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices.get(title)
    if pd.isna(idx):
        return ["Title not found in dataset."]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5 recommendations
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

selected_title = st.selectbox("Select a Movie or TV Show", df['title'].dropna().unique())
if st.button("Recommend Similar Titles"):
    recommendations = get_recommendations(selected_title)
    st.write("### Recommended Titles:")
    for rec in recommendations:
        st.write(f"- {rec}")
