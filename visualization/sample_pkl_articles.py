import os
import pickle
import random
from tqdm import tqdm

def sample_fixed_per_file(directory, output_file, num_per_file=10):
    # List all pkl files in the directory
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pkl')]
    
    sampled_articles = []
    articles = []

    for file_path in tqdm(all_files, desc="Processing files"):
        with open(file_path, 'rb') as f:

            while True:
                try:
                    
                    article = pickle.load(f)
                    articles.append(article)
                
                except EOFError:
                    break
            try:

                if len(articles) == 0:
                    print(f"Warning: File {file_path} did not contain a valid list. Skipping.")
                    continue
                
                # Randomly sample `num_per_file` articles, or all if fewer are available
                sampled = random.sample(articles, min(num_per_file, len(articles)))
                sampled_articles.extend(sampled)
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
                continue
    
    random.shuffle(sampled_articles)

    # Save the sampled articles to a new pkl file
    with open(output_file, 'wb') as f:
        pickle.dump(sampled_articles, f)
    
    print(f"Sampled {len(sampled_articles)} articles ({num_per_file} per file) and saved to {output_file}")

# Usage
sample_fixed_per_file('output', 'SFR_sampled_articles.pkl', num_per_file=10)
