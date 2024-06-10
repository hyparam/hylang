import os
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier  # Example classifier

def process_files(directory):
    vectorizer = HashingVectorizer(n_features=2**20, alternate_sign=False)  # Large number of features to reduce hash collisions
    classifier = SGDClassifier()  # Just an example classifier
    
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.parquet'):
            filepath = os.path.join(directory, filename)
            for chunk in pd.read_parquet(filepath, chunksize=1000):  # Adjust chunksize based on memory availability
                X = vectorizer.transform(chunk['content'])
                # Suppose we have labels in chunk['label'] for supervised learning
                # Y = chunk['label']  # Uncomment and adjust if you have labels
                # classifier.partial_fit(X, Y, classes=np.unique(Y))  # Uncomment for supervised learning
                # For unsupervised or other processing, just handle X

    # You can save the model or do additional processing
    # joblib.dump(classifier, 'model.pkl')  # Example to save the trained model

# Directory containing parquet files
directory_path = 'starcoderdata/javascript/'
process_files(directory_path)
