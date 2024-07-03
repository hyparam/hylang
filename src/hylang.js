// Tokenization and model parameters (weights and biases)
const tokens = ["function", "var", "const", "let", "if", "else", "for", "while"] // TODO
const languages = ["Python", "JavaScript", "Java"] // TODO
const weights = [
        [0.2, 0.5, -0.3, 0.8],
        [-0.6, 0.1, 0.3, 0.5],
        [0.7, -0.2, 0.1, -0.4]
    ]
const biases = [0.1, -0.1, 0.05]

/**
 * Convert input text to features
 * @param {string} text
 * @returns {number[]} 
 */
function textToFeatures(text) {
    const features = new Array(tokens.length).fill(0)
    tokens.forEach((token, index) => {
        const regex = new RegExp(`\\b${token}\\b`, 'g')
        features[index] = (text.match(regex) || []).length
    })
    return features
}

/**
 * Classify the input text based on features
 * @param {string} text - input text
 * @returns {string} predicted programming language
 */
export function detectLanguage(text) {
    const features = textToFeatures(text)
    
    // Calculate the logits for each class
    const logits = weights.map((weightsRow, i) => {
        const weightedSum = weightsRow.reduce((sum, weight, j) => sum + weight * features[j], biases[i])
        return weightedSum
    })
    
    // Apply softmax to logits to get probabilities
    const expLogits = logits.map(logit => Math.exp(logit))
    const sumExpLogits = expLogits.reduce((sum, expLogit) => sum + expLogit, 0)
    const probabilities = expLogits.map(expLogit => expLogit / sumExpLogits)
    
    // Find the class with the highest probability
    const predictedClass = probabilities.indexOf(Math.max(...probabilities))
    return languages[predictedClass]
}

// Example usage
const inputText = "function example() { const x = 10 let y = 20 if (x > y) { console.log('x is greater') } }"
const predictedClass = detectLanguage(inputText)
console.log("Predicted class:", predictedClass)
