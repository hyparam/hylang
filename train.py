import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pyarrow.parquet as pq
import joblib

def load_top_tfidf_words(tfidf_parquet_path, top_n=1000):
    # Load TF-IDF scores from a Parquet file
    df_tfidf = pd.read_parquet(tfidf_parquet_path)
    # Sort and select the top N words
    top_words = df_tfidf.nlargest(top_n, 'Score')['Token'].tolist()
    return top_words

def create_feature_matrix(data_directory, top_words):
    all_features = []
    all_labels = []
    
    # Iterate over language directories
    for language in os.listdir(data_directory):
        language_dir = os.path.join(data_directory, language)
        if os.path.isdir(language_dir):
            # Process each Parquet file in the language directory
            for filename in os.listdir(language_dir):
                if filename.endswith('.parquet'):
                    filepath = os.path.join(language_dir, filename)
                    parquet_file = pq.ParquetFile(filepath)
                    for batch in parquet_file.iter_batches(batch_size=1000, columns=['content']):
                        df = batch.to_pandas()
                        # Create feature vector for each document
                        features = df['content'].apply(lambda x: [1 if word in x else 0 for word in top_words])
                        labels = [language] * len(df)
                        all_features.extend(features.tolist())
                        all_labels.extend(labels)
    return all_features, all_labels

def train_decision_tree(features, labels):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    # Train a Decision Tree classifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    # Predict and calculate accuracy
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy: {accuracy:.2%}')
    return clf

def save_classifier(classifier, filename):
    joblib.dump(classifier, filename)

def load_classifier(filename):
    return joblib.load(filename)

# Paths
tfidf_parquet_path = 'output/tfidf_scores_sorted.parquet'
data_directory = 'starcoderdata/'
classifier_path = 'saved_model/classifier.joblib'

# Load and prepare data
top_words = load_top_tfidf_words(tfidf_parquet_path, 1000)
features, labels = create_feature_matrix(data_directory, top_words)

# Train and save the classifier
classifier = train_decision_tree(features, labels)
save_classifier(classifier, classifier_path)

# Optionally load the classifier
# loaded_classifier = load_classifier(classifier_path)
# Use loaded_classifier for further evaluation or prediction
