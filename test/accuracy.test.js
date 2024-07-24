import { createReadStream, promises as fs } from 'fs'
import { parquetRead } from 'hyparquet'
import { describe, expect, it } from 'vitest'
import { detectLanguage } from '../src/hylang.js'

const evalFile = 'language_eval.parquet'

async function evaluateModel() {
  const stat = await fs.stat(evalFile)
  const file = {
    byteLength: stat.size,
    /**
     * @param {number} start - start position
     * @param {number} end - end position
     * @returns {Promise<ArrayBuffer>}
     */
    async slice(start, end) {
      const readStream = createReadStream(evalFile, { start, end })
      return await readStreamToArrayBuffer(readStream)
    },
  }
  /**
   * @type {any[][]}
   */
  const evalData = await parquetReadPromise({ file })

  const predictions = evalData.map(row => detectLanguage(row[1]))
  const trueLabels = evalData.map(row => row[0])

  const correct = predictions.filter((pred, i) => pred?.toLowerCase() === trueLabels[i]).length
  return correct / predictions.length
}

describe('Language Classification Model', () => {
  it('should achieve accuracy above the specified threshold', async () => {
    const accuracyThreshold = 0.7
    const accuracy = await evaluateModel()

    console.log(`Model Accuracy: ${(accuracy * 100).toFixed(2)}%`)

    expect(accuracy).toBeGreaterThan(accuracyThreshold)
  }, 300000) // 5-minute timeout
})

/**
 * @param {import('hyparquet').ParquetReadOptions} options
 */
function parquetReadPromise(options) {
  return new Promise((resolve, reject) => {
    parquetRead({
      ...options,
      onComplete: (data) => {
        resolve(data)
      },
    }).catch(reject)
  })
}

/**
 * Convert a web ReadableStream to ArrayBuffer.
 *
 * @typedef {import('stream').Readable} ReadStream
 * @param {ReadStream} input
 * @returns {Promise<ArrayBuffer>}
 */
function readStreamToArrayBuffer(input) {
  return new Promise((resolve, reject) => {
    /** @type {Buffer[]} */
    const chunks = []
    input.on('data', chunk => chunks.push(chunk))
    input.on('end', () => {
      const buffer = Buffer.concat(chunks)
      resolve(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength))
    })
    input.on('error', reject)
  })
}
