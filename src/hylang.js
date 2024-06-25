// Example tokens and model parameters (weights and biases) loaded from the JSON file
const tokens = ["function", "var", "const", "let", "if", "else", "for", "while"]; // Example tokens
const modelParams = {
    "weights": [
        [0.2, 0.5, -0.3, 0.8],
        [-0.6, 0.1, 0.3, 0.5],
        [0.7, -0.2, 0.1, -0.4]
    ],
    "biases": [0.1, -0.1, 0.05]
};

// Function to convert input text to features
function textToFeatures(text, tokens) {
    const features = new Array(tokens.length).fill(0);
    tokens.forEach((token, index) => {
        const regex = new RegExp(`\\b${token}\\b`, 'g');
        features[index] = (text.match(regex) || []).length;
    });
    return features;
}

// Function to classify the input text based on features
function classifyText(text, tokens, modelParams) {
    const features = textToFeatures(text, tokens);
    const weights = modelParams.weights;
    const biases = modelParams.biases;
    
    // Calculate the logits for each class
    const logits = weights.map((weightsRow, i) => {
        const weightedSum = weightsRow.reduce((sum, weight, j) => sum + weight * features[j], biases[i]);
        return weightedSum;
    });
    
    // Apply softmax to logits to get probabilities
    const expLogits = logits.map(logit => Math.exp(logit));
    const sumExpLogits = expLogits.reduce((sum, expLogit) => sum + expLogit, 0);
    const probabilities = expLogits.map(expLogit => expLogit / sumExpLogits);
    
    // Find the class with the highest probability
    const predictedClass = probabilities.indexOf(Math.max(...probabilities));
    
    return predictedClass;
}

// Example usage
const inputText = "function example() { const x = 10; let y = 20; if (x > y) { console.log('x is greater'); } }";
const predictedClass = classifyText(inputText, tokens, modelParams);
console.log("Predicted class:", predictedClass);
