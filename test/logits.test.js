import { describe, expect, it } from 'vitest'
import { detectLanguageLogits } from '../src/hylang.js'
import { languages } from '../src/params.json'

describe('detectLanguageLogits', () => {
  it('should return an array of logits for JavaScript', () => {
    const jsCode = `
      function greet(name) {
        console.log('Hello, ' + name + '!')
      }

      const numbers = [1, 2, 3, 4, 5]
      let sum = 0

      for (let i = 0; i < numbers.length; i++) {
        sum += numbers[i]
      }

      console.log('Sum:', sum)
    `

    const logits = detectLanguageLogits(jsCode)
    expect(Object.keys(logits)).toEqual(languages)
    expect(Object.values(logits).every(logit => typeof logit === 'number')).toBe(true)
    expect(logits['javascript']).toBeGreaterThan(logits['python'])
    // every logit should be less than or equal to javascript
    expect(Object.values(logits).every(logit => logit <= logits['javascript'])).toBe(true)
  })

  it('should return different logits for different inputs', () => {
    const jsCode = `
      function test() {
        return 'Hello, World!';
      }
    `

    const pythonCode = `
      def test():
          return 'Hello, World!'
    `

    const jsLogits = detectLanguageLogits(jsCode)
    const pythonLogits = detectLanguageLogits(pythonCode)
    expect(jsLogits).not.toEqual(pythonLogits)
  })
})
