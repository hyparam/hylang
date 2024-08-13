import torch
import json
import numpy as np

# Paths for model storage
classifier_path = 'output/classifier.pth'
params_path = 'output/params.json'
reduced_model_path = '../output/reduced_model.pth'

class SimpleLinearNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleLinearNN, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

def select_top_features(model, n_features):
    weights = model.linear.weight.data.numpy()
    feature_importance = np.sum(np.abs(weights), axis=0)
    top_feature_indices = np.argsort(feature_importance)[-n_features:]
    return top_feature_indices

def load_model_and_select_features(n_features):
    with open(params_path, 'r') as json_file:
        params = json.load(json_file)
    
    input_dim = len(params['tokens'])
    output_dim = len(params['languages'])

    model = SimpleLinearNN(input_dim, output_dim)
    model.load_state_dict(torch.load(classifier_path))

    top_feature_indices = select_top_features(model, n_features)
    top_feature_names = [params['tokens'][i] for i in top_feature_indices]
    
    return model, params, top_feature_indices, top_feature_names

def create_reduced_model(original_model, top_feature_indices):
    n_features = len(top_feature_indices)
    output_dim = original_model.linear.out_features

    reduced_model = SimpleLinearNN(n_features, output_dim)
    
    # Copy weights and biases for selected features
    reduced_model.linear.weight.data = original_model.linear.weight.data[:, top_feature_indices].clone()
    reduced_model.linear.bias.data = original_model.linear.bias.data.clone()

    return reduced_model

def save_reduced_model(reduced_model, top_feature_names, params):
    model_dict = {
        'state_dict': reduced_model.state_dict(),
        'top_features': top_feature_names,
        'languages': params['languages']
    }
    torch.save(model_dict, reduced_model_path)

def main():
    n_features = 100
    model, params, top_feature_indices, top_feature_names = load_model_and_select_features(n_features)
    reduced_model = create_reduced_model(model, top_feature_indices)
    save_reduced_model(reduced_model, top_feature_names, params)
    
    print(f"Top {n_features} features:")
    for i, feature in enumerate(top_feature_names, 1):
        print(f"{i}. {feature}")
    print(f"\nReduced model saved to {reduced_model_path}")

if __name__ == "__main__":
    main()
