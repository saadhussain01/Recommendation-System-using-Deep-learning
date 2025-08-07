import pandas as pd
import streamlit as st
import numpy as np
import pickle


df = pd.read_csv("netflix_content.csv")
model = pickle.load(open("model.pkl",'rb'))


def recommend_similar(content_title, top_k=5):
    content_row = df[df['Title'].str.contains(content_title, case=False, na=False)].iloc[0]
    content_id = content_row['Content_ID']
    language_id = content_row['Language_ID']
    content_type_id = content_row['ContentType_ID']

    predictions = model.predict({
        'content_id': np.array([content_id]),
        'language_id': np.array([language_id]),
        'content_type': np.array([content_type_id])
    })

    top_indices = predictions[0].argsort()[-top_k-1:][::-1]
    recommendations = df[df['Content_ID'].isin(top_indices)]
    return recommendations[['Title', 'Language Indicator', 'Content Type', 'Hours Viewed']]



st.set_page_config(page_title="Movie Recommender",layout='centered')
st.title("Movie Recommender System")


movie_titles = sorted(df['Title'].unique())
selected_movie = st.selectbox("Select a movie",movie_titles)


if st.button("Recommed"):
    with st.spinner("Generating recommendations..."):
        recommendations = recommend_similar(selected_movie)
        st.subheader("Recommend Titles:")
        st.dataframe(recommendations.reset_index(drop=True))
