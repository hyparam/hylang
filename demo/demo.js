import { detectLanguage } from '../src/hylang.js'

const codeInput = /** @type {HTMLInputElement} */ (document.getElementById('codeInput'))
const result = document.getElementById('result')
if (!codeInput || !result) throw new Error('Invalid HTML elements')

document.addEventListener('DOMContentLoaded', () => {
  codeInput.addEventListener('input', () => {
      const code = codeInput.value.trim()
      if (code) {
          try {
              const language = detectLanguage(code)
              result.textContent = language
          } catch (error) {
              result.textContent = 'Unknown'
              console.error('Error detecting language:', error)
          }
      } else {
          result.textContent = 'None'
      }
  })
})
