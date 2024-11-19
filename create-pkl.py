from transformers import AutoTokenizer, AutoModel
import torch
import pickle
import os
from tqdm import tqdm

def mean_pooling(embeddings, mask):
    token_embeddings = embeddings  # (batch_size, seq_length, hidden_size)
    input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    return sum_embeddings / sum_mask


# Function to batch data
def batchify_data(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

cache_dir = '/scratch/project_2000539/maryam/embed/.cache'

# Disable tokenizers parallelism to avoid the warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define a batch size and max length for each input batch
batch_size = 64
max_length = 128

# Load the tokenizer and model from Hugging Face
model_name = "Salesforce/SFR-Embedding-Mistral"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.float16)

news_path = "/scratch/project_2000539/maryam/rag_data/"
news_files = ["volkswagen_articles.txt"] #, "microsoft_articles.txt", "small_data.txt", "boeing_articles.txt"
news_txt = ""

for file in news_files:
    file_path = news_path + file
    f = open(file_path, "r")
    file_text = f.read()
    news_txt += file_text

news_doc_list = news_txt.split("\n@@")

total_docs = len(news_doc_list)

with open("/scratch/project_2000539/maryam/volkswagen_file_embeddings.pkl", "wb") as f:

    for doc in tqdm(news_doc_list, desc="Processing documents", total=total_docs):

        # List to collect embeddings
        all_doc_embeddings = []
        # Process the document in batches of sentences or chunks
        for batch in batchify_data(doc, batch_size):
            # Tokenize the batch
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

            # Move inputs and model to GPU (if available)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            model = model.to(device)

            # Get embeddings for the batch
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)
                batch_embeddings = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)
            
            # Get the attention mask (to ignore padding tokens)
            attention_mask = inputs['attention_mask']  # Shape: (batch_size, seq_length)

            # Only keep non-padded tokens
            for i in range(batch_embeddings.size(0)):  # Iterate over the batch
                non_padded_indices = attention_mask[i].nonzero().squeeze(-1)  # Indices of non-padded tokens
                non_padded_embeddings = batch_embeddings[i, non_padded_indices, :]  # Select only non-padded embeddings
                all_doc_embeddings.append(non_padded_embeddings.cpu())
            
        # Concatenate all the non-padded tokens for the document
        all_doc_embeddings = torch.cat(all_doc_embeddings, dim=0)  # (total_seq_length, hidden_size)

        # Perform mean pooling over the entire document (averaging over all tokens)
        doc_attention_mask = torch.ones(all_doc_embeddings.shape[0], dtype=torch.int)  # Full attention for all tokens
        doc_embedding = mean_pooling(all_doc_embeddings.unsqueeze(0), doc_attention_mask.unsqueeze(0))  # (1, hidden_size)

        # Convert the final document embedding to numpy
        doc_embedding = doc_embedding.cpu().numpy()
        
        # Save the document embedding to the pickle file
        pickle.dump(doc_embedding, f)