import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pyarrow.parquet as pq
import joblib
from tqdm import tqdm

# Paths for data and model storage
tfidf_parquet_path = 'output/tfidf_scores_sorted_1000.parquet'
data_directory = 'starcoderdata/'
classifier_path = 'output/classifier.joblib'

def load_top_tfidf_words(tfidf_parquet_path, top_n=1000):
    df_tfidf = pd.read_parquet(tfidf_parquet_path)
    top_words = df_tfidf.nlargest(top_n, 'Score')['Token'].tolist()
    return top_words

def create_feature_matrix(data_directory, top_words):
    all_features = []
    all_labels = []

    # Subdirectories for each programming language
    languages = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]
    for language in tqdm(languages, desc="Processing Languages"):
        language_dir = os.path.join(data_directory, language)
        files = [f for f in os.listdir(language_dir) if f.endswith('.parquet')]

        # Process each file in the language directory
        for filename in tqdm(files, desc=f"Files in {language}", leave=False):
            filepath = os.path.join(language_dir, filename)
            parquet_file = pq.ParquetFile(filepath)
            
            for batch in parquet_file.iter_batches(batch_size=1000, columns=['content']):
                df = batch.to_pandas()
                features = df['content'].apply(lambda x: [1 if word in x else 0 for word in top_words])
                labels = [language] * len(df)
                all_features.extend(features.tolist())
                all_labels.extend(labels)
    return all_features, all_labels

def train_decision_tree(features, labels):
    print("Training the classifier...")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy: {accuracy:.2%}')
    return clf

# Load and prepare data
top_words = load_top_tfidf_words(tfidf_parquet_path, 1000)
features, labels = create_feature_matrix(data_directory, top_words)

# Train and save the classifier
classifier = train_decision_tree(features, labels)
joblib.dump(classifier, classifier_path)
