import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Function to read parquet files from a directory
def read_parquet_files(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.parquet')]
    dataframes = [pd.read_parquet(os.path.join(directory, f)) for f in files]
    return pd.concat(dataframes, ignore_index=True)

# Path to your parquet files (adjust as necessary)
directory_path = 'starcoderdata/javascript/'

# Read the parquet files
data = read_parquet_files(directory_path)

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the 'content' column
tfidf_matrix = tfidf_vectorizer.fit_transform(data['content'])

# Example: Display the shape of the TF-IDF matrix
print("TF-IDF Matrix shape:", tfidf_matrix.shape)
