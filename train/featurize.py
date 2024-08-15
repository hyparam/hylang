import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import CountVectorizer

# Paths
train_path = 'output/data/train.parquet'
token_path = 'output/model-large/tokens.jsonl'
output_path = 'output/model-large/featurized_data.parquet'

def load_tokens(token_path):
    tokens_df = pd.read_json(token_path, lines=True)
    return tokens_df['Token'].tolist()

def create_feature_matrix(train_parquet_path, tokens):
    print(f"Creating feature matrix from {len(tokens)} tokens...")
    vectorizer = CountVectorizer(vocabulary=tokens, binary=True)
    
    # Read the train.parquet file
    df = pd.read_parquet(train_parquet_path)
    
    # Transform content to feature vectors
    features = vectorizer.transform(df['content']).toarray()
    
    # Create DataFrame with features
    feature_df = pd.DataFrame(features, columns=tokens, index=df.index)
    
    # Rename 'language' column to 'programming_language' to fix conflict with feature name
    df = df.rename(columns={'language': 'programming_language'})

    # Combine features with programming language labels
    result_df = pd.concat([df[['programming_language']], feature_df], axis=1)
    
    # Write the result to parquet file
    table = pa.Table.from_pandas(result_df)
    pq.write_table(table, output_path, compression='snappy')

# Load top TF-IDF words and create feature matrix
tokens = load_tokens(token_path)
create_feature_matrix(train_path, tokens)

print(f"Featurized data saved to {output_path}")