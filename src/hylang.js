import { biases, languages, tokens, weights } from './params.json'

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
