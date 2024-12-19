import os
import torch
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import numpy as np
import random

def mean_pooling(embeddings, mask):
    """
    Mean pooling over token embeddings with attention mask.
    """
    input_mask_expanded = mask.unsqueeze(-1).expand(embeddings.size()).float()
    sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    return sum_embeddings / sum_mask

# Function to batch data
def batchify_data(data, batch_size):
    """
    Splits data into batches of specified size.
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

# Configuration
cache_dir = '/scratch/project_2000539/maryam/embed/.cache'
batch_size = 64
max_length = 128

# Initialize tokenizer and model
model_name = "dunzhang/stella_en_1.5B_v5"#"Salesforce/SFR-Embedding-Mistral"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.float16).cuda()

# Set environment variable to suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define paths
news_path = os.path.join(os.getenv("LOCAL_SCRATCH", "/tmp"), "news")
news_files = [os.path.join(news_path, f) for f in os.listdir(news_path) if os.path.isfile(os.path.join(news_path, f))]

total_files = len(news_files)

# Process files
for file_path in tqdm(news_files, desc="Processing documents", total=total_files):

    with open(file_path, "r") as f:
        news_txt = f.read()

    # Split document into sections
    news_doc_list = [doc.strip() for doc in news_txt.split("\n") if doc.strip()] #.split("\n@@")

    """if len(news_doc_list) < 10:
        random_docs = news_doc_list  # All items are selected
    else:
        random_docs = random.sample(news_doc_list, 10)
"""
    all_embeddings = []

    for doc in news_doc_list:

        all_doc_embeddings = []
        # Process documents in batches
        for doc_batch in batchify_data(doc, batch_size):
            # Tokenize batch
            inputs = tokenizer(doc_batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

            # Move to GPU
            inputs = {key: value.cuda() for key, value in inputs.items()}

            # Generate embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                batch_embeddings = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)

            # Perform mean pooling
            attention_mask = inputs['attention_mask']

            # Only keep non-padded tokens
            for i in range(batch_embeddings.size(0)):  # Iterate over the batch
                non_padded_indices = attention_mask[i].nonzero().squeeze(-1)  # Indices of non-padded tokens
                non_padded_embeddings = batch_embeddings[i, non_padded_indices, :]  # Select only non-padded embeddings
                all_doc_embeddings.append(non_padded_embeddings.cpu())

        # Concatenate all the non-padded tokens for the document
        all_doc_embeddings = torch.cat(all_doc_embeddings, dim=0)  # (total_seq_length, hidden_size)

        # Perform mean pooling over the entire document (averaging over all tokens)
        doc_attention_mask = torch.ones(all_doc_embeddings.shape[0], dtype=torch.int)  # Full attention for all tokens
        doc_embeddings = mean_pooling(all_doc_embeddings.unsqueeze(0), doc_attention_mask.unsqueeze(0))  # (1, hidden_size)

        # Convert to numpy and store
        all_embeddings.extend(doc_embeddings.cpu().numpy())
        
    if len(all_embeddings) == 0:
        continue
    # Save all document embeddings to a pickle file
    output_path = file_path.replace(".txt", ".pkl").replace("news/", "")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(np.stack(all_embeddings), f)
"""
    with open(output_path.replace(".pkl", ".txt"), "w") as f:
        for text in random_docs:
            f.write(text + '\n')
            """