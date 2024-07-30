import os
import pandas as pd
import pyarrow.parquet as pq
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

train_file = 'output/train--.parquet'
output_path = 'output/'

# Create output directory if it doesn't exist
Path(output_path).mkdir(parents=True, exist_ok=True)

def custom_code_tokenizer(code):
    # Define a regular expression to match code tokens
    token_pattern = re.compile(r'\w+|[{}()\[\];,]')
    return token_pattern.findall(code)

def process_file(file_path, output_path):
    vectorizer = TfidfVectorizer(tokenizer=custom_code_tokenizer)

    # Read the specific parquet file
    print(f"Reading Parquet file: {file_path}")
    df = pd.read_parquet(file_path)

    # Extract documents
    documents = df['content'].tolist()

    # Fit TF-IDF vectorizer
    print("Fitting TF-IDF vectorizer...")
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    # Create a DataFrame of features and their scores
    # Sum TF-IDF scores across all documents
    scores = tfidf_matrix.sum(axis=0).A1
    score_data = {'Token': feature_names, 'Score': scores}
    df_scores = pd.DataFrame(score_data)

    # Sort by scores in descending order and save top tokens
    df_scores_sorted = df_scores.sort_values(by='Score', ascending=False)
    df_scores_sorted.head(2000).to_parquet(os.path.join(output_path, 'top_tokens.parquet'), index=False)

    print(f"Saved top 2000 tokens to {output_path}/top_tokens.parquet")

# Run TF-IDF
process_file(train_file, output_path)
