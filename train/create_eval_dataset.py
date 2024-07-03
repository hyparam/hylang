import os
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# Paths
data_directory = 'starcoderdata/'
output_file = 'language_eval.parquet'

# Whitelist of languages to include
selected_languages = [
    'assembly', 'c', 'cpp', 'c-sharp', 'css', 'cuda', 'go', 'html', 'java', 'javascript',
    'json', 'kotlin', 'lua', 'markdown', 'matlab', 'php', 'protocol-buffer', 'python',
    'r', 'ruby', 'rust', 'scala', 'shell', 'sql', 'tex', 'typescript', 'yaml'
]

def sample_and_combine_data(language, input_files, sample_fraction=0.0001):
    sampled_data = []

    for filename in tqdm(input_files, desc=f"Sampling {language}", leave=False):
        filepath = os.path.join(data_directory, language, filename)
        parquet_file = pq.ParquetFile(filepath)

        # Read and sample data
        for batch in parquet_file.iter_batches(batch_size=1000):
            df = batch.to_pandas()
            sample_size = max(1, int(len(df) * sample_fraction))
            df_sampled = df.sample(n=sample_size, random_state=1)
            df_sampled['language'] = language  # Add language column
            sampled_data.append(df_sampled[['language', 'content']])  # Keep only language and content columns

    return pd.concat(sampled_data, ignore_index=True) if sampled_data else pd.DataFrame(columns=['language', 'content'])

# Get all language directories and filter by whitelist
language_dirs = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d)) and d in selected_languages]

# Process each language directory and combine data
all_data = []

for language in tqdm(language_dirs, desc="Languages"):
    language_dir = os.path.join(data_directory, language)

    # Collect all Parquet files for the current language
    files = [f for f in os.listdir(language_dir) if f.endswith('.parquet')]

    # Sample and combine the Parquet data
    language_data = sample_and_combine_data(language, files)
    all_data.append(language_data)

# Combine all language data into a single DataFrame
combined_df = pd.concat(all_data, ignore_index=True)

# Shuffle the combined DataFrame
shuffled_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the shuffled data to a Parquet file
shuffled_df.to_parquet(output_file, index=False)

print(f"Saved {len(combined_df)} rows to {output_file}")