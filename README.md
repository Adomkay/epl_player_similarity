# EPL Player Similarity
An interactive Streamlit app that uses player statistics from the English Premier League 2020/2021 season to find and display players with similar performance profiles, leveraging cosine similarity for similarity measurement.
# Dataset
The dataset used for this app is the English Premier League Player Statistics 2020/2021, which is publicly available on Kaggle. The dataset can be downloaded here.

# Steps taken to create the app:
Data Ingestion: The player stats dataset (pl_20-21.csv) is loaded using pandas. The dataset contains various statistics for each player in the English Premier League for the 2020/2021 season.

# Data Preprocessing: 
NaN values in the dataset are replaced with zeros. For columns containing percentage values, the '%' symbol is removed and the values are converted to numeric format. Then, the player names are stripped of any leading or trailing whitespaces.

# Vector Embedding Generation: 
For each player, a vector embedding is generated using the standardized numerical statistics. Standardization is performed using Scikit-learn's StandardScaler.

# Similarity Matrix Calculation: 
A similarity matrix is calculated based on cosine similarity of the vector embeddings. Each entry in this matrix represents the cosine similarity between a pair of players.

# Player Selection: 
The Streamlit app allows users to select a player from a dropdown list, which includes all players in the dataset.

# Similar Player Retrieval: 
Upon selection of a player, the 10 most similar players to the selected player are retrieved based on the similarity matrix. The players are presented along with their positions and similarity scores.

# How to run the app:
Clone this repository or download the files to your local machine.

# Install the necessary Python packages. 
This project requires pandas, numpy, sklearn, and streamlit. You can install them using pip:

pip install pandas numpy scikit-learn streamlit

Navigate to the directory containing the files in a terminal or command prompt.
Run the Streamlit app using the command:

streamlit run app.py

Visit the URL displayed in the terminal (usually http://localhost:8501) to interact with the app.
