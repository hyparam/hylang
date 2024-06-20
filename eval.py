import os
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import joblib
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer

# Paths for data and model storage
classifier_path = 'output/classifier.pth'
label_encoder_path = 'output/label_encoder.joblib'
tfidf_parquet_path = 'output/tfidf_scores_top1000.parquet'

# Load the label encoder
label_encoder = joblib.load(label_encoder_path)

# Define the model architecture
class SimpleLinearNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleLinearNN, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)

# Load the model
input_dim = 1000  # Assuming 1000 TF-IDF features were used
output_dim = len(label_encoder.classes_)
model = SimpleLinearNN(input_dim, output_dim)
model.load_state_dict(torch.load(classifier_path))
model.eval()  # Set the model to evaluation mode

# Load the top TF-IDF words to initialize the vectorizer
df_tfidf = pd.read_parquet(tfidf_parquet_path)
top_words = df_tfidf.nlargest(1000, 'Score')['Token'].tolist()
vectorizer = CountVectorizer(vocabulary=top_words, binary=True)

def model_inference(input_text):
    """
    Language classification model that predicts the programming language based on the input text.
    """
    features = vectorizer.transform([input_text]).toarray()
    with torch.no_grad():
        outputs = model(torch.tensor(features, dtype=torch.float32))
        _, predicted = torch.max(outputs, 1)
        predicted_label = label_encoder.inverse_transform(predicted.cpu().numpy())[0]
    return predicted_label

def evaluate_model(base_directory):
    all_predictions = []
    all_true_labels = []

    # Get all language directories
    language_dirs = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]

    # Subdirectories for each programming language
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
                predictions = [model_inference(text) for text in df['content']]
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
