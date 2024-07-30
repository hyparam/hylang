# HyLang Model Training

## create_dataset.py

Creates `output/train.parquet` training dataset and `output/eval.parquet` evaluation dataset by sampling from [starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata).

| File          | Sample Fraction | Rows    | File Size   |
|---------------|-----------------|---------|-------------|
| train.parquet | 0.001           | 157,837 | 237,419,792 |
| eval.parquet  | 0.0001          | 16,207  | 24,969,691  |

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
