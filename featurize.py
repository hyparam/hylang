import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

# Paths
tfidf_parquet_path = 'output/tfidf_scores_sorted_1000.parquet'
data_directory = 'starcoderdata/'
output_path = 'output/featurized_data.parquet'

def load_top_tfidf_words(tfidf_parquet_path, top_n=1000):
    df_tfidf = pd.read_parquet(tfidf_parquet_path)
    top_words = df_tfidf.nlargest(top_n, 'Score')['Token'].tolist()
    return top_words

def create_feature_matrix(data_directory, top_words):
    vectorizer = CountVectorizer(vocabulary=top_words, binary=True)
    schema = None
    writer = None
    
    try:
        for language in tqdm(os.listdir(data_directory), desc="Processing Languages", unit="lang"):
            language_dir = os.path.join(data_directory, language)
            files = [f for f in os.listdir(language_dir) if f.endswith('.parquet')]

            for filename in tqdm(files, desc=f"Files in {language}", leave=False, unit="file"):
                filepath = os.path.join(language_dir, filename)
                df = pd.read_parquet(filepath)
                # Sample 1% of the rows
                df_sampled = df.sample(frac=0.01, random_state=1) if len(df) > 100 else df  # Ensure there's enough rows to sample
                if 'content' in df_sampled.columns:
                    features = vectorizer.transform(df_sampled['content']).toarray()
                    labels = pd.DataFrame({'programming_language': [language]*len(df_sampled)}, index=df_sampled.index)
                    feature_df = pd.DataFrame(features, columns=top_words, index=df_sampled.index)
                    result_df = pd.concat([labels, feature_df], axis=1)

                    table = pa.Table.from_pandas(result_df, schema=schema, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(output_path, table.schema, compression='snappy')
                        schema = table.schema
                    writer.write_table(table)
    finally:
        if writer:
            writer.close()

top_words = load_top_tfidf_words(tfidf_parquet_path)
create_feature_matrix(data_directory, top_words)
