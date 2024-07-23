# HyLang

[![workflow status](https://github.com/hyparam/hylang/actions/workflows/ci.yml/badge.svg)](https://github.com/hyparam/hylang/actions)
[![mit license](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![dependencies](https://img.shields.io/badge/Dependencies-0-blueviolet)](https://www.npmjs.com/package/hylang?activeTab=dependencies)
![coverage](https://img.shields.io/badge/Coverage-100-darkred)

A stupidly small and fast programming language detection model.

## Usage

```js
import { detectLanguage } from 'hylang'

const input = `
  function square(x) {
    return x * x
  }
`
console.log(`Predicted language: ${detectLanguage(input)}`)
```
