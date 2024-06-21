# HyLang

A stupidly small and fast programming language detection model.

## create_lite_dataset.py

Generates a sampled version of starcoderdata with 0.01% of the rows from each language.

## tfidf.py

Generates `output/top_tokens.parquet` with the top 1000 tfidf words from all documents.
Uses a custom tokenizer to get code-like tokens including punctuation `{`, `}`, `;`, etc.

97m58s

## featurize.py

Generates `output/featurized_data.parquet` which maps each document to `programming_language` and 1000 binary features for presence of each of the top 1000 tokens.

## train.py

Trains a single layer linear torch model.

## eval.py

return 'python' => 8.16%
return 'markdown' => 13.32%
one layer net => 91.45%
