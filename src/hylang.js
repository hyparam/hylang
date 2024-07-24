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
 * @returns {Record<string, number>} logits for each language
 */
export function detectLanguageLogits(text) {
  const features = textToFeatures(text)

  /** @type {Record<string, number>} */
  const logits = {}
  weights.forEach((weightsRow, i) => {
    const weightedSum = weightsRow.reduce((sum, weight, j) => sum + weight * features[j], biases[i])
    logits[languages[i]] = weightedSum
  })

  return logits
}

/**
 * Apply softmax to an object of logits
 * @param {Record<string, number>} logits - input object of logits
 * @returns {Object} softmax probabilities
 */
function softmax(logits) {
  /** @type {Record<string, number>} */
  const probabilities = {}
  let sum = 0

  // Calculate exp of logits and sum
  for (const lang in logits) {
    probabilities[lang] = Math.exp(logits[lang])
    sum += probabilities[lang]
  }
  // Normalize to get probabilities
  for (const lang in probabilities) {
    probabilities[lang] /= sum
  }
  return probabilities
}

/**
 * Classify the input text based on features
 * @param {string} text - input text
 * @returns {string} predicted programming language
 */
export function detectLanguage(text) {
  const logits = detectLanguageLogits(text)
  const probabilities = softmax(logits)
  const predictedLanguage = Object.entries(probabilities).reduce(
    (max, [lang, prob]) => prob > max[1] ? [lang, prob] : max,
    ['', -Infinity],
  )[0]
  return predictedLanguage
}
