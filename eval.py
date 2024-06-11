import os
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score

def model(input_text):
    """
    Naive language classification model that always predicts 'python'.
    """
    return 'python'

def evaluate_model(base_directory):
    all_predictions = []
    all_true_labels = []

    # Loop through each language directory
    for language in os.listdir(base_directory):
        language_dir = os.path.join(base_directory, language)
        if os.path.isdir(language_dir):
            # Loop through all Parquet files in the language directory
            for filename in os.listdir(language_dir):
                if filename.endswith('.parquet'):
                    filepath = os.path.join(language_dir, filename)
                    parquet_file = pq.ParquetFile(filepath)

                    # Read batches of data
                    for batch in parquet_file.iter_batches(batch_size=1000, columns=['content']):
                        df = batch.to_pandas()
                        predictions = [model(text) for text in df['content']]
                        true_labels = [language] * len(df)  # All samples in this file are assumed to be in 'language'

                        all_predictions.extend(predictions)
                        all_true_labels.extend(true_labels)

    # Calculate and print the overall accuracy
    accuracy = accuracy_score(all_true_labels, all_predictions)
    print(f"Accuracy: {accuracy:.2%}")

# Base directory containing language-specific subdirectories
base_directory = 'starcoderdata/'

# Evaluate the model
evaluate_model(base_directory)
