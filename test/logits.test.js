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

    const expected = {
      assembly: -0.7302241288125515,
      c: 0.14168843626976013,
      csharp: -1.5614262968301773,
      cpp: 0.38608236610889435,
      css: -1.1905248016119003,
      cuda: -1.317980371415615,
      go: -0.9581845551729202,
      html: -0.4717388153076172,
      java: -2.817721428349614,
      javascript: 6.35457331314683,
      json: 1.8853686526417732,
      kotlin: -1.16586297377944,
      lua: 0.4855741783976555,
      markdown: 2.3705248199403286,
      matlab: -2.1717951372265816,
      php: 0.36925166845321655,
      protobuf: -1.4156804345548153,
      python: -0.6433932483196259,
      r: -1.2239609584212303,
      ruby: -1.1991716846823692,
      rust: 1.3296690955758095,
      scala: -1.675346203148365,
      shell: -0.04831627756357193,
      sql: -0.5250878911465406,
      tex: 0.2085392934968695,
      typescript: 3.2758028879761696,
      yaml: 2.312015675008297,
    }

    const logits = detectLanguageLogits(jsCode)
    expect(logits).toEqual(expected)
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
