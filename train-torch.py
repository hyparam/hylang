import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pyarrow.parquet as pq
from tqdm import tqdm

# Paths
tfidf_parquet_path = 'output/tfidf_scores_sorted_1000.parquet'
data_directory = 'starcoderdata/'
classifier_path = 'output/classifier.pth'

def load_top_tfidf_words(tfidf_parquet_path, top_n=1000):
    df_tfidf = pd.read_parquet(tfidf_parquet_path)
    top_words = df_tfidf.nlargest(top_n, 'Score')['Token'].tolist()
    return top_words

def create_feature_matrix(data_directory, top_words):
    all_features = []
    all_labels = []
    
    languages = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]
    for language in tqdm(languages, desc="Processing Languages"):
        language_dir = os.path.join(data_directory, language)
        files = [f for f in os.listdir(language_dir) if f.endswith('.parquet')]
        
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

class SimpleLinearNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleLinearNN, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)

def train_neural_network(features, labels, input_dim, output_dim, epochs=10, batch_size=32, learning_rate=0.01):
    print("Training the neural network...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert features and labels to tensors
    X = torch.tensor(features, dtype=torch.float32).to(device)
    le = LabelEncoder()
    y = torch.tensor(le.fit_transform(labels), dtype=torch.long).to(device)
    
    # Create train and test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the model, loss function, and optimizer
    model = SimpleLinearNN(input_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Training loop
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
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = accuracy_score(y_test.cpu(), predicted.cpu())
        print(f'Accuracy: {accuracy:.2%}')
    
    return model, le

# Load and prepare data
top_words = load_top_tfidf_words(tfidf_parquet_path, 1000)
features, labels = create_feature_matrix(data_directory, top_words)

# Train and save the classifier
input_dim = len(top_words)
output_dim = len(set(labels))
model, label_encoder = train_neural_network(features, labels, input_dim, output_dim)
torch.save(model.state_dict(), classifier_path)
joblib.dump(label_encoder, 'output/label_encoder.joblib')
