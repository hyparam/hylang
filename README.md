# HyLang

A stupidly small and fast programming language detection model.

## tfidf.py

Generates `output/tfidf_scores_sorted_1000.parquet` with the top 1000 tfidf words from all documents.

## featurize.py

Generates `output/featurized_data.parquet` which maps each document to `programming_language` and 1000 binary features for presence of each of the top 1000 tokens.

## train.py

Trains a single layer linear torch model.

## eval.py

return 'python' => 6.23%
return 'markdown' => 10.18%
