import os
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

train_file = 'output/data/train.parquet'
token_file = 'output/model-large/tokens.jsonl'
token_count = 1000

# Create output directory
Path(token_file).parent.mkdir(parents=True, exist_ok=True)

def custom_code_tokenizer(code):
    # Define a regular expression to match code tokens
    token_pattern = re.compile(r'\w+|[{}()\[\];,]')
    return token_pattern.findall(code)

def process_file(train_file, token_file):
    vectorizer = TfidfVectorizer(tokenizer=custom_code_tokenizer)

    # Read the specific parquet file
    print(f"Reading Parquet file: {train_file}")
    df = pd.read_parquet(train_file)

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
    df_scores_sorted.head(token_count).to_json(token_file, orient='records', lines=True)

    print(f"Saved top {token_count} tokens to {token_file}")

# Run TF-IDF
process_file(train_file, token_file)
