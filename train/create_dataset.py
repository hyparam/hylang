import os
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

data_directory = 'starcoderdata/'

# Whitelist of languages to include
selected_languages = [
    'assembly', 'c', 'cpp', 'c-sharp', 'css', 'cuda', 'go', 'html', 'java',
    'javascript', 'json', 'kotlin', 'lua', 'markdown', 'php', 'python',
    'r', 'ruby', 'rust', 'scala', 'shell', 'sql', 'tex', 'typescript', 'yaml'
]

# Rename languages to remove special characters
language_map = {
    'c-sharp': 'csharp'
}

def sample_and_combine_data(language, input_files, sample_fraction):
    renamed_language = language_map.get(language, language)
    sampled_data = []

    for filename in tqdm(input_files, desc=f"Sampling {renamed_language}", leave=False):
        filepath = os.path.join(data_directory, language, filename)
        parquet_file = pq.ParquetFile(filepath)

        # Read and sample data
        for batch in parquet_file.iter_batches(batch_size=10000):
            df = batch.to_pandas()
            sample_size = max(1, int(len(df) * sample_fraction))
            df_sampled = df.sample(n=sample_size, random_state=1)
            df_sampled['language'] = renamed_language
            sampled_data.append(df_sampled[['language', 'content']])  # Keep only language and content columns

    return pd.concat(sampled_data, ignore_index=True) if sampled_data else pd.DataFrame(columns=['language', 'content'])

def create_dataset(data_directory, output_file, sample_fraction):
    print(f"Creating dataset {output_file} with sample fraction {sample_fraction}")

    # Iterate through selected language directories
    language_dirs = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d)) and d in selected_languages]
    all_data = []
    for language in tqdm(language_dirs, desc="Languages"):
        language_dir = os.path.join(data_directory, language)

        # Collect all Parquet files for the current language
        files = [f for f in os.listdir(language_dir) if f.endswith('.parquet')]

        # Sample and combine the Parquet data
        language_data = sample_and_combine_data(language, files, sample_fraction)
        all_data.append(language_data)

    # Combine into one DataFrame, shuffle and save to Parquet
    combined_df = pd.concat(all_data, ignore_index=True)
    shuffled_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    shuffled_df.to_parquet(output_file, index=False)

    print(f"Saved {len(combined_df)} rows to {output_file}")

# Create the training set
create_dataset(data_directory, 'output/data/train.parquet', 0.002)
# Create the evaluation set
create_dataset(data_directory, 'output/data/eval.parquet', 0.0001)
