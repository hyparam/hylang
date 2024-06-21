import os
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# Paths
data_directory = 'starcoderdata/'
output_directory = 'starcoderdata-lite/'

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Whitelist of languages to include
selected_languages = [
    'assembly', 'c', 'cpp', 'c-sharp', 'css', 'cuda', 'go', 'html', 'java', 'javascript',
    'json', 'kotlin', 'lua', 'markdown', 'matlab', 'php', 'protocol-buffer', 'python',
    'r', 'ruby', 'rust', 'scala', 'shell', 'sql', 'tex', 'typescript', 'yaml'
]

def sample_and_save_parquet(language, input_files, output_file, sample_fraction=0.0001):
    sampled_data = []

    for filename in tqdm(input_files, desc=f"Sampling {language}", leave=False):
        filepath = os.path.join(data_directory, language, filename)
        parquet_file = pq.ParquetFile(filepath)

        # Read and sample data
        for batch in parquet_file.iter_batches(batch_size=1000):
            df = batch.to_pandas()
            sample_size = max(1, int(len(df) * sample_fraction))
            df_sampled = df.sample(n=sample_size, random_state=1)
            sampled_data.append(df_sampled)

    # Combine all sampled data into a single DataFrame
    if sampled_data:
        sampled_df = pd.concat(sampled_data, ignore_index=True)
        sampled_df.to_parquet(output_file, index=False)
        print(f"Saved {len(sampled_df)} rows to {output_file}")
    else:
        print(f"No data sampled for {language}")

# Get all language directories and filter by whitelist
language_dirs = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d)) and d in selected_languages]

# Process each language directory
for language in tqdm(language_dirs, desc="Languages"):
    language_dir = os.path.join(data_directory, language)
    output_language_dir = os.path.join(output_directory, language)
    output_file = os.path.join(output_language_dir, f"{language}.parquet")

    # Create nested directory for the language if it doesn't exist
    os.makedirs(output_language_dir, exist_ok=True)

    # Collect all Parquet files for the current language
    files = [f for f in os.listdir(language_dir) if f.endswith('.parquet')]

    # Sample and save the Parquet data
    sample_and_save_parquet(language, files, output_file)
