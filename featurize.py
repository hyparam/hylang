import os
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

# Paths
tfidf_parquet_path = 'output/tfidf_scores_sorted.parquet'
data_directory = 'starcoderdata/'
output_path = 'output/featurized_data.parquet'

def load_top_tfidf_words(tfidf_parquet_path, top_n=1000):
    df_tfidf = pd.read_parquet(tfidf_parquet_path)
    top_words = df_tfidf.nlargest(top_n, 'Score')['Token'].tolist()
    return top_words

def create_feature_matrix(data_directory, top_words):
    vectorizer = CountVectorizer(vocabulary=top_words, binary=True)
    
    all_features = []
    for language in tqdm(os.listdir(data_directory), desc="Processing Languages"):
        language_dir = os.path.join(data_directory, language)
        files = [f for f in os.listdir(language_dir) if f.endswith('.parquet')]
        
        for filename in tqdm(files, desc=f"Files in {language}", leave=False):
            filepath = os.path.join(language_dir, filename)
            df = pd.read_parquet(filepath)
            if 'content' in df.columns:
                features = vectorizer.transform(df['content']).toarray()
                labels = pd.DataFrame({'language': [language]*len(df)})
                feature_df = pd.DataFrame(features, columns=top_words)
                result_df = pd.concat([labels, feature_df], axis=1)
                all_features.append(result_df)
    
    if all_features:
        final_df = pd.concat(all_features)
        final_df.to_parquet(output_path)

top_words = load_top_tfidf_words(tfidf_parquet_path)
create_feature_matrix(data_directory, top_words)
