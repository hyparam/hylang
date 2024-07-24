import torch
import json
import numpy as np

# Paths for model storage
classifier_path = 'output/classifier.pth'
params_path = 'output/params.json'
params_lite_path = '../src/params.json'

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

def create_lite_params(model, params, top_feature_indices, top_feature_names):
    lite_params = {
        'tokens': top_feature_names,
        'languages': params['languages'],
        'weights': model.linear.weight.data.numpy()[:, top_feature_indices].tolist(),
        'biases': model.linear.bias.data.numpy().tolist()
    }
    return lite_params

def save_lite_params(lite_params):
    with open(params_lite_path, 'w') as json_file:
        json.dump(lite_params, json_file, indent=2)

def main():
    n_features = 100
    model, params, top_feature_indices, top_feature_names = load_model_and_select_features(n_features)
    lite_params = create_lite_params(model, params, top_feature_indices, top_feature_names)
    save_lite_params(lite_params)
    
    print(f"Top {n_features} features:")
    for i, feature in enumerate(top_feature_names, 1):
        print(f"{i}. {feature}")
    print(f"\nReduced model parameters saved to {params_lite_path}")

if __name__ == "__main__":
    main()
