# HyLang Model Training

## Environment setup

Setup your python environment, and download starcoderdata from hugging face (311 GB).

```bash
python -m venv .env-train
source .env-train/bin/activate
pip install -r requirements.txt
git lfs install
git clone git@hf.co:datasets/bigcode/starcoderdata
```

## create_dataset.py

Creates `output/data/train.parquet` training dataset and `output/data/eval.parquet` evaluation dataset by sampling from [starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata).

| File          | Sample Fraction | Rows    | Bytes       |
|---------------|-----------------|---------|-------------|
| train.parquet | 0.002           | 315,542 | 477,595,439 |
| eval.parquet  | 0.0001          | 16,207  | 24,969,691  |

146m58s

## tfidf.py

Generates `output/model-large/tokens.jsonl` with the top 1000 tfidf words from all documents.
Uses a custom tokenizer to get code-like tokens including punctuation `{`, `}`, `;`, etc.

4m00s

## featurize.py

Generates `output/model-large/featurized_data.parquet` which maps each document to `programming_language` and 1000 binary features for presence of each of the top 1000 tokens.

2m55s

## train.py

Trains a single layer linear torch model on the featurized dataset for 40 epochs.

Loss: 0.3577
4m30s

## reduce.py

Reduce the number of features from 1000 to 100 tokens.

2s

## eval.py

return 'javascript' => 12.43%
1000-tokens-one-layer-net => 92.83%
1000-tokens-one-layer-net-reduced-100 => 72.70%

0m34s
