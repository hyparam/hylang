import { describe, expect, it } from 'vitest'
import { detectLanguageLogits } from '../src/hylang.js'

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
    expect(logits).toBeInstanceOf(Array)
    expect(logits.length).toBeGreaterThan(0)
    expect(logits.every(logit => typeof logit === 'number')).toBe(true)
    expect(Math.max(...logits)).toBeGreaterThan(0)
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
