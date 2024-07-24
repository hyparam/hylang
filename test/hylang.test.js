import { describe, expect, it } from 'vitest'
import { detectLanguage } from '../src/hylang.js'

describe('detectLanguage', () => {
  it('should detect JavaScript', () => {
    const jsCode = `
      function greet(name) {
        console.log('Hello, ' + name + '!')
      }

      const numbers = [1, 2, 3, 4, 5]
      let sum = 0

      for (let i = 0 i < numbers.length i++) {
        sum += numbers[i]
      }

      console.log('Sum:', sum)
    `

    expect(detectLanguage(jsCode)).toBe('javascript')
  })

  it('should detect Python', () => {
    const pythonCode = `
      def greet(name):
          print(f"Hello, {name}!")
      
      numbers = [1, 2, 3, 4, 5]
      sum = 0

      for num in numbers:
          sum += num

      print("Sum:", sum)
    `

    expect(detectLanguage(pythonCode)).toBe('python')
  })

  it('should detect Java', () => {
    const javaCode = `
      public class HelloWorld {
          public static void main(String[] args) {
              System.out.println("Hello, World!")
              
              int[] numbers = {1, 2, 3, 4, 5}
              int sum = 0
              
              for (int i = 0 i < numbers.length i++) {
                  sum += numbers[i]
              }
              
              System.out.println("Sum: " + sum)
          }
      }
    `

    expect(detectLanguage(javaCode)).toBe('java')
  })

  it('should handle empty input', () => {
    expect(detectLanguage('')).toBe('javascript')
  })

  it('should handle input with no specific language features', () => {
    const genericCode = `
      x = 5
      y = 10
      z = x + y
    `

    expect(detectLanguage(genericCode)).toBe('javascript')
  })
})
