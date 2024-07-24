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
 * Calculate logits for language detection
 * @param {string} text - input text
 * @returns {number[]} logits for each language
 */
export function detectLanguageLogits(text) {
  const features = textToFeatures(text)

  return weights.map((weightsRow, i) => {
    return weightsRow.reduce((sum, weight, j) => sum + weight * features[j], biases[i])
  })
}

/**
 * Apply softmax to an array of numbers
 * @param {number[]} arr - input array
 * @returns {number[]} softmax probabilities
 */
function softmax(arr) {
  const expArr = arr.map(Math.exp)
  const sumExpArr = expArr.reduce((sum, exp) => sum + exp, 0)
  return expArr.map(exp => exp / sumExpArr)
}

/**
 * Classify the input text based on features
 * @param {string} text - input text
 * @returns {string} predicted programming language
 */
export function detectLanguage(text) {
  const logits = detectLanguageLogits(text)
  const probabilities = softmax(logits)
  const predictedClass = probabilities.indexOf(Math.max(...probabilities))
  return languages[predictedClass]
}
