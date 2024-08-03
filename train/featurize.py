import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

# Paths
token_parquet_path = 'output/top_tokens.parquet'
train_parquet_path = 'output/train.parquet'
output_path = 'output/featurized_data.parquet'

def load_top_tfidf_words(tfidf_parquet_path, top_n=1000):
    df_tfidf = pd.read_parquet(tfidf_parquet_path)
    top_words = df_tfidf.nlargest(top_n, 'Score')['Token'].tolist()
    return top_words

def create_feature_matrix(train_parquet_path, top_words):
    vectorizer = CountVectorizer(vocabulary=top_words, binary=True)
    
    # Read the train.parquet file
    df = pd.read_parquet(train_parquet_path)
    
    # Transform content to feature vectors
    features = vectorizer.transform(df['content']).toarray()
    
    # Create DataFrame with features
    feature_df = pd.DataFrame(features, columns=top_words, index=df.index)
    
    # Rename 'language' column to 'programming_language' to fix conflict with feature name
    df = df.rename(columns={'language': 'programming_language'})

    # Combine features with programming language labels
    result_df = pd.concat([df[['programming_language']], feature_df], axis=1)
    
    # Write the result to parquet file
    table = pa.Table.from_pandas(result_df)
    pq.write_table(table, output_path, compression='snappy')

# Load top TF-IDF words and create feature matrix
top_words = load_top_tfidf_words(token_parquet_path)
create_feature_matrix(train_parquet_path, top_words)

print(f"Featurized data saved to {output_path}")