import { describe, expect, it } from 'vitest'
import { biases, languages, tokens, weights } from '../src/params.json'

describe('params.json', () => {
  it('weights rows matchs languages', () => {
    expect(weights.length).toBe(languages.length)
  })
  it('weights columns matchs tokens', () => {
    expect(weights[0].length).toBe(tokens.length)
  })
  it('biases matchs languages', () => {
    expect(biases.length).toBe(languages.length)
  })
})
