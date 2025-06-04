import os
import torch
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import numpy as np
import re
from datasets import load_from_disk


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
output_path_base = "/scratch/project_2000539/maryam/adaptive-mrag/FineWeb-embedd-modified/"
cache_dir = '/scratch/project_2000539/maryam/embed/.cache'
batch_size = 64
max_length = 128

# Initialize tokenizer and model
model_name = "Salesforce/SFR-Embedding-Mistral" #"dunzhang/stella_en_1.5B_v5"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.float16).cuda()
model.layers[-1].self_attn.o_proj = torch.nn.Identity()

# Set environment variable to suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():

    dataset = load_from_disk("/scratch/project_2000539/maryam/fineweb_dataset")
    number_of_doc = 0

    # Process files
    for file in tqdm(dataset, desc="Processing documents", total=len(dataset)):

        if number_of_doc > 4000:
            break

        text = file['text']
        number_of_doc += 1

        all_doc_embeddings = []
        # Process documents in batches
        for doc_batch in batchify_data(text, batch_size):
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
        doc_embeddings = doc_embeddings.cpu().numpy()
            
        if len(doc_embeddings) == 0:
            continue
        # Save all document embeddings to a pickle file
        id = ''.join(re.findall(r'[\d-]+', file['id']))
        output_path = output_path_base + id + ".pkl"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "wb") as f:
            pickle.dump(np.stack(doc_embeddings), f)


main()