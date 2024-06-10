import os
import random
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm

def process_files(directory, sample_percentage=10):
    vectorizer = HashingVectorizer(n_features=2**20, alternate_sign=False)
    classifier = SGDClassifier()  # Example classifier

    # List all files and sample from them
    all_files = [f for f in os.listdir(directory) if f.endswith('.parquet')]
    sampled_files = random.sample(all_files, k=int(len(all_files) * (sample_percentage / 100)))

    # Loop through the sampled files
    for filename in tqdm(sampled_files, desc="Processing files"):
        filepath = os.path.join(directory, filename)
        parquet_file = pq.ParquetFile(filepath)

        # Progress bar for batches within each file
        total_batches = parquet_file.metadata.num_row_groups
        progress = tqdm(total=total_batches, desc=f"Reading {filename}", leave=False)

        # Read in batches
        for batch in parquet_file.iter_batches(batch_size=1000, columns=['content']):
            df = batch.to_pandas()
            X = vectorizer.transform(df['content'])
            # Example: training a model with dummy labels (uncomment and replace as necessary)
            # Y = df['label']
            # classifier.partial_fit(X, Y, classes=np.unique(Y))
            progress.update(1)

        progress.close()

    # Optionally save the model or do additional processing
    # joblib.dump(classifier, 'model.pkl')

# Directory containing parquet files
directory_path = 'starcoderdata/javascript/'
process_files(directory_path, sample_percentage=10)
