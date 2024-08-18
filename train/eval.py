import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import joblib
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
import torch.nn.functional as F

# Paths for data and model storage
classifier_path = 'output/model-large/classifier.pth'
label_encoder_path = 'output/model-large/label_encoder.joblib'
token_path = 'output/model-large/tokens.jsonl'
eval_parquet_path = 'output/data/eval.parquet'
output_parquet_path = 'output/data/eval_results.parquet'

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

# Load the top TF-IDF words to initialize the vectorizer
df_tokens = pd.read_json(token_path, lines=True)
tokens = df_tokens['Token'].tolist()
vectorizer = CountVectorizer(vocabulary=tokens, binary=True)

# Load the model
input_dim = len(tokens)
output_dim = len(label_encoder.classes_)
model = SimpleLinearNN(input_dim, output_dim)
model.load_state_dict(torch.load(classifier_path))
model.eval()  # Set the model to evaluation mode

def model_inference(input_text):
    """
    Language classification model that predicts the programming language based on the input text.
    Returns the predicted label and confidence score.
    """
    features = vectorizer.transform([input_text]).toarray()
    with torch.no_grad():
        outputs = model(torch.tensor(features, dtype=torch.float32))
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_label = label_encoder.inverse_transform(predicted.cpu().numpy())[0]
        confidence_score = confidence.item()
    return predicted_label, confidence_score

def evaluate_model(eval_parquet_path, output_parquet_path):
    all_predictions = []
    all_true_labels = []
    all_inputs = []
    all_confidences = []

    # Read the eval.parquet file
    df = pd.read_parquet(eval_parquet_path)

    # Process the data in batches
    batch_size = 1000
    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch = df.iloc[i:i+batch_size]
        predictions_and_confidences = [model_inference(text) for text in batch['content']]
        predictions, confidences = zip(*predictions_and_confidences)
        true_labels = batch['language'].tolist()
        inputs = batch['content'].tolist()

        all_predictions.extend(predictions)
        all_true_labels.extend(true_labels)
        all_inputs.extend(inputs)
        all_confidences.extend(confidences)

    # Calculate the overall accuracy
    accuracy = accuracy_score(all_true_labels, all_predictions)
    print(f"Overall Accuracy: {accuracy:.2%}")

    # Create a DataFrame with the results
    results_df = pd.DataFrame({
        'input': all_inputs,
        'gold_output': all_true_labels,
        'predicted_output': all_predictions,
        'confidence': all_confidences,
        'score': [1 if pred == true else 0 for pred, true in zip(all_predictions, all_true_labels)]
    })

    # Save the results to a parquet file
    results_df.to_parquet(output_parquet_path, index=False)
    print(f"Results saved to {output_parquet_path}")

# Evaluate the model and save results
evaluate_model(eval_parquet_path, output_parquet_path)
