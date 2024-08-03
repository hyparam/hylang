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
token_parquet_path = 'output/top_tokens.parquet'
eval_parquet_path = 'output/eval.parquet'

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
df_tfidf = pd.read_parquet(token_parquet_path)
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

def evaluate_model(eval_parquet_path):
    all_predictions = []
    all_true_labels = []

    # Read the eval.parquet file
    df = pd.read_parquet(eval_parquet_path)

    # Process the data in batches
    batch_size = 1000
    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch = df.iloc[i:i+batch_size]
        predictions = [model_inference(text) for text in batch['content']]
        true_labels = batch['language'].tolist()

        all_predictions.extend(predictions)
        all_true_labels.extend(true_labels)

    # Calculate and print the overall accuracy
    accuracy = accuracy_score(all_true_labels, all_predictions)
    print(f"Overall Accuracy: {accuracy:.2%}")

# Evaluate the model
evaluate_model(eval_parquet_path)
