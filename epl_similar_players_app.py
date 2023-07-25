import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

@st.cache_data
def load_data():
    # Load the dataset
    df = pd.read_csv('pl_20-21.csv')

    # Replace NaN values with 0
    df.fillna(0, inplace=True)

    # Remove '%' symbol from percentage columns and convert to numeric values
    percentage_cols = [col for col in df.columns if df[col].astype(str).str.contains('%').any()]
    for col in percentage_cols:
        df[col] = df[col].str.replace('%', '').astype(float)

    # Replace infinity values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaN values with 0 (or use another appropriate fill method)
    df.fillna(0, inplace=True)

    # Select the numerical columns for generating vector embeddings
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numerical_cols = [col for col in numerical_cols if col != 'Unnamed: 0']

    # Standardize the numerical columns
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Generate vector embeddings (as numpy array)
    vector_embeddings = df[numerical_cols].values

    # Create a similarity matrix based on cosine similarity
    similarity_matrix = cosine_similarity(vector_embeddings)

    # Strip leading/trailing spaces from player names
    df['Name'] = df['Name'].str.strip()

    # Convert the numpy array to a DataFrame for easier handling
    similarity_df = pd.DataFrame(similarity_matrix, index=df['Name'], columns=df['Name'])

    return df, similarity_df

def get_similar_players(df, similarity_df, player_name, top_n=10):
    # Get the similarity scores for the given player
    similarity_scores = similarity_df[player_name]

    # Sort the scores in descending order and take the top_n players
    most_similar_players = similarity_scores.sort_values(ascending=False).head(top_n + 1)

    # Exclude the player themselves from the list
    most_similar_players = most_similar_players[most_similar_players.index != player_name]

    # Create a DataFrame with names and positions of the similar players
    similar_players_df = df[df['Name'].isin(most_similar_players.index)][['Name', 'Position']]

    # Reset index
    similar_players_df.reset_index(drop=True, inplace=True)

    # Add the similarity scores to the DataFrame
    similar_players_df['Similarity'] = most_similar_players.values

    return similar_players_df

# Add title
st.title("EPL Player Similarity Finder (2020/2021 Season)")

# Load the data
df, similarity_df = load_data()

# Select a player
player_name = st.selectbox('Select a player', df['Name'].unique())

# Select number of similar players to display
top_n = st.slider('Select number of similar players to display', min_value=1, max_value=50, value=10)

# Button to get similar players
if st.button('Get Similar Players'):
    # Get the most similar players
    similar_players = get_similar_players(df, similarity_df, player_name, top_n)

    # Display the similar players
    st.write(similar_players)
