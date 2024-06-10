import os
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier  # Example classifier
from tqdm import tqdm  # Import tqdm for the progress bar

def process_files(directory):
    vectorizer = HashingVectorizer(n_features=2**20, alternate_sign=False)
    classifier = SGDClassifier()  # Just an example classifier

    # Loop through all files in the directory
    for filename in tqdm(os.listdir(directory), desc="Processing files"):
        if filename.endswith('.parquet'):
            filepath = os.path.join(directory, filename)
            parquet_file = pq.ParquetFile(filepath)

            # Prepare to show progress for batches within each file
            total_batches = parquet_file.metadata.num_row_groups
            progress = tqdm(total=total_batches, desc=f"Reading {filename}", leave=False)

            # Read in batches
            for batch in parquet_file.iter_batches(batch_size=1000, columns=['content']):
                df = batch.to_pandas()
                X = vectorizer.transform(df['content'])
                # Example: training a model with dummy labels (uncomment and replace as necessary)
                # Y = df['label']  # Uncomment and adjust if you have a label column
                # classifier.partial_fit(X, Y, classes=np.unique(Y))  # Uncomment for supervised learning
                # For unsupervised or other processing, just handle X
                progress.update(1)  # Update the inner progress bar after each batch

            progress.close()  # Ensure the inner progress bar is properly closed after finishing a file

    # Save the model or do additional processing if needed
    # joblib.dump(classifier, 'model.pkl')  # Example to save the trained model

# Directory containing parquet files
directory_path = 'starcoderdata/javascript/'
process_files(directory_path)
