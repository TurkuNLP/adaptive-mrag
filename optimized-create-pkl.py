import os
import torch
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import numpy as np

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
model_name = "Salesforce/SFR-Embedding-Mistral"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.float16).cuda()

# Set environment variable to suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define paths
news_path = os.path.join(os.getenv("LOCAL_SCRATCH", "/tmp"), "news")
news_files = [os.path.join(news_path, f) for f in os.listdir(news_path) if os.path.isfile(os.path.join(news_path, f))]

# Process files
for file_path in news_files:
    with open(file_path, "r") as f:
        news_txt = f.read()

    # Split document into sections
    news_doc_list = [doc.strip() for doc in news_txt.split("\n@@") if doc.strip()]

    all_embeddings = []

    # Process documents in batches
    for doc_batch in tqdm(batchify_data(news_doc_list, batch_size), desc=f"Processing {file_path}"):
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
        doc_embeddings = mean_pooling(batch_embeddings, attention_mask)

        # Convert to numpy and store
        all_embeddings.extend(doc_embeddings.cpu().numpy())

    # Save all document embeddings to a pickle file
    output_path = file_path.replace(".txt", ".pkl").replace("news/", "")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(np.stack(all_embeddings), f)