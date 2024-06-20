import os
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pyarrow.parquet as pq
from tqdm import tqdm

# Paths for data and model storage
tfidf_parquet_path = 'output/tfidf_scores_top1000.parquet'
data_directory = 'starcoderdata/'
classifier_path = 'output/classifier.pth'

class SimpleLinearNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleLinearNN, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)

def train_model(features, labels, input_dim, output_dim, epochs=10, batch_size=32, learning_rate=0.01):
    print("Training the neural network...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert features and labels to tensors
    X = torch.tensor(features.values, dtype=torch.float32).to(device) # Convert DataFrame to NumPy array
    le = LabelEncoder()
    y = torch.tensor(le.fit_transform(labels), dtype=torch.long).to(device)
    
    # Create train and test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the model, loss function, and optimizer
    model = SimpleLinearNN(input_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Training loop with batch processing
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    # Evaluation on test set
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = accuracy_score(y_test.cpu(), predicted.cpu())
        print(f'Accuracy: {accuracy:.2%}')
    
    return model, le

# Load and prepare data
def load_data(parquet_path):
    df = pd.read_parquet(parquet_path)
    labels = df['programming_language']
    features = df.drop(columns=['programming_language'])
    return features, labels

# Load features and labels from parquet file
features, labels = load_data('output/featurized_data.parquet')

# Determine input and output dimensions
input_dim = features.shape[1]
output_dim = len(labels.unique())

# Train and save the classifier model and label encoder
model, label_encoder = train_model(features, labels, input_dim, output_dim)

torch.save(model.state_dict(), classifier_path)
joblib.dump(label_encoder, 'output/label_encoder.joblib')
