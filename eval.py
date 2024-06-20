import os
import pyarrow.parquet as pq
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import joblib

classifier = joblib.load(classifier_path)

def model(input_text):
    """
    Naive language classification model that always predicts 'python'.
    """
    return 'markdown'

def evaluate_model(base_directory):
    all_predictions = []
    all_true_labels = []

    # Get all language directories
    language_dirs = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]

    # Progress bar for languages
    for language in tqdm(language_dirs, desc="Languages", unit="lang"):
        language_dir = os.path.join(base_directory, language)

        # Collect all files for the current language
        files = [f for f in os.listdir(language_dir) if f.endswith('.parquet')]

        # Progress bar for files in the current language directory
        for filename in tqdm(files, desc=f"Processing {language}", leave=False):
            filepath = os.path.join(language_dir, filename)
            parquet_file = pq.ParquetFile(filepath)

            # Read batches of data
            for batch in parquet_file.iter_batches(batch_size=1000, columns=['content']):
                df = batch.to_pandas()
                predictions = [model(text) for text in df['content']]
                true_labels = [language] * len(df)

                all_predictions.extend(predictions)
                all_true_labels.extend(true_labels)

    # Calculate and print the overall accuracy
    accuracy = accuracy_score(all_true_labels, all_predictions)
    print(f"Overall Accuracy: {accuracy:.2%}")

# Base directory containing language-specific subdirectories
base_directory = 'starcoderdata/'

# Evaluate the model
evaluate_model(base_directory)
