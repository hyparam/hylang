import os
import pyarrow.parquet as pq

def model(input_text):
    """
    Naive language classification model that always predicts 'python'.
    """
    return 'python'

def evaluate_model(directory):
    correct_predictions = 0
    total_predictions = 0

    # Loop through all Parquet files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.parquet'):
            filepath = os.path.join(directory, filename)
            parquet_file = pq.ParquetFile(filepath)

            # Read batches of data
            for batch in parquet_file.iter_batches(batch_size=1000, columns=['content', 'language']):
                df = batch.to_pandas()

                # Apply the model to each row in the DataFrame
                predictions = [model(text) for text in df['content']]
                true_labels = df['language'].tolist()

                # Calculate accuracy
                correct_predictions += sum(1 for i in range(len(predictions)) if predictions[i] == true_labels[i])
                total_predictions += len(predictions)

    # Print the results
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})")

# Directory containing Parquet files
directory_path = 'starcoderdata/javascript/'

# Evaluate the model
evaluate_model(directory_path)
