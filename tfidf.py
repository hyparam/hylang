import os
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier  # Example classifier

def process_files(directory):
    vectorizer = HashingVectorizer(n_features=2**20, alternate_sign=False)
    classifier = SGDClassifier()  # Just an example classifier

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.parquet'):
            filepath = os.path.join(directory, filename)
            parquet_file = pq.ParquetFile(filepath)

            # Read in batches
            for batch in parquet_file.iter_batches(batch_size=1000, columns=['content']):  # Adjust batch_size based on memory
                df = batch.to_pandas()
                X = vectorizer.transform(df['content'])
                # Example: training a model with dummy labels (uncomment and replace as necessary)
                # Y = df['label']  # Uncomment and adjust if you have a label column
                # classifier.partial_fit(X, Y, classes=np.unique(Y))  # Uncomment for supervised learning
                # For unsupervised or other processing, just handle X

    # Save the model or do additional processing if needed
    # joblib.dump(classifier, 'model.pkl')  # Example to save the trained model

# Directory containing parquet files
directory_path = 'starcoderdata/javascript/'
process_files(directory_path)
