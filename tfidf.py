import os
import pandas as pd
import pyarrow.parquet as pq
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# Directory containing parquet files and output directory
directory_path = 'starcoderdata/javascript/'
output_path = 'output/'

def custom_code_tokenizer(code):
    # Define a regular expression to match code tokens
    token_pattern = re.compile(r'\w+|[{}()\[\];,]')
    return token_pattern.findall(code)

def process_files(directory, output_path):
    vectorizer = TfidfVectorizer(tokenizer=custom_code_tokenizer)

    # List all files
    all_files = [f for f in os.listdir(directory) if f.endswith('.parquet')]

    # Collect all documents to compute global TF-IDF
    documents = []
    for filename in tqdm(all_files, desc="Processing files", unit="file"):
        filepath = os.path.join(directory, filename)
        parquet_file = pq.ParquetFile(filepath)

        for batch in parquet_file.iter_batches(batch_size=1000, columns=['content']):
            df = batch.to_pandas()
            documents.extend(df['content'].tolist())

    # Fit TF-IDF vectorizer
    print("Fitting TF-IDF vectorizer...")
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    # Create a DataFrame of features and their scores
    # Sum TF-IDF scores across all documents
    scores = tfidf_matrix.sum(axis=0).A1
    score_data = {'Token': feature_names, 'Score': scores}
    df_scores = pd.DataFrame(score_data)

    # Sort the DataFrame by scores in descending order
    print("Sorting tokens by TF-IDF scores...")
    df_scores_sorted = df_scores.sort_values(by='Score', ascending=False)

    # Save to a Parquet file
    # df_scores_sorted.to_parquet(os.path.join(output_path, 'tfidf_scores_sorted.parquet'))

    # Save top 1000 tokens to Parquet file
    df_scores_sorted.head(1000).to_parquet(os.path.join(output_path, 'top_tokens.parquet'))

# Run TF-IDF
process_files(directory_path, output_path)
