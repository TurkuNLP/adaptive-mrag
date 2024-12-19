import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import os
from transformers import AutoTokenizer, AutoModel
import torch
import pickle

def get_tops_indices(embeddings, specific_heads):

    embeddings_array = np.array(embeddings)
    n_samples = embeddings_array.shape[0]

    # Step 1: Split each embedding into 32 parts (heads)
    num_heads = 32
    head_size = embeddings_array.shape[1] // num_heads
    split_embeddings = embeddings_array.reshape(n_samples, num_heads, head_size)

    tops = []

    for head_idx in specific_heads:

        # Compute magnitudes for the current head's embeddings
        head_embeddings = split_embeddings[:, head_idx, :]
        magnitudes = np.linalg.norm(head_embeddings, axis=1)
        
        sorted_indices = np.argsort(magnitudes)  # Sort magnitudes in ascending order
        top_5_indices = sorted_indices[-10:][::-1]  # Last 5, reversed for descending order
        low_5_indices = sorted_indices[:5] 

        tops.append([top_5_indices, low_5_indices])

    return tops

# Function to get word embeddings
def get_word_embeddings(text, tokenizer, model):
    # Tokenize the input text, keeping offset_mapping for later use
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        return_offsets_mapping=True,
        max_length=128
    )

    # Store offset_mapping separately before modifying inputs
    offset_mapping = inputs['offset_mapping']  # Extract offset_mapping for later use
    inputs = {k: v.cuda() for k, v in inputs.items() if k != 'offset_mapping'}  # Move other inputs to GPU

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Last layer hidden states

    token_embeddings = hidden_states.squeeze(0)  # Shape: (seq_length, hidden_dim)
    offset_mapping = offset_mapping.squeeze(0).cpu().numpy()  # Convert offset_mapping for further use
    
    word_embeddings = {}
    current_word = ""
    word_tokens = []
    
    for i, (start, end) in enumerate(offset_mapping):
        if start == 0 and end == 0:  # Skip special tokens
            continue
        token_embedding = token_embeddings[i].cpu().numpy()
        word_piece = text[start:end]
        if not word_piece.startswith(" "):  # Continuation of a word
            current_word += word_piece
            word_tokens.append(token_embedding)
        else:  # New word starts
            if current_word:
                word_embeddings[current_word] = np.mean(word_tokens, axis=0)
            current_word = word_piece.strip()
            word_tokens = [token_embedding]
    
    if current_word:
        word_embeddings[current_word] = np.mean(word_tokens, axis=0)
    
    return word_embeddings

# Function to calculate word contributions (Dot Product + Cosine Similarity)
def calculate_word_contributions(word_embeddings, doc_embedding):
    contributions = {}
    doc_norm = np.linalg.norm(doc_embedding)  # Norm of document embedding
    
    for word, embedding in word_embeddings.items():
        word_norm = np.linalg.norm(embedding)
        # Dot Product
        dot_product = np.dot(embedding, doc_embedding)
        # Cosine Similarity
        cosine_similarity = dot_product / (word_norm * doc_norm) if word_norm > 0 and doc_norm > 0 else 0.0
        contributions[word] = {"dot_product": dot_product, "cosine_similarity": cosine_similarity}
    
    return contributions

def get_embed_txt(directory_path):
    all_embd = []
    all_txt = []

    for filename in os.listdir(directory_path):
        if os.path.splitext(filename)[1] != ".pkl":
            with open(os.path.join(directory_path, filename), "r") as f:
                text = f.read()
                articles = text.strip().split("\n") 
                all_txt.extend(articles)
            continue
        with open(os.path.join(directory_path, filename), "rb") as f:
            while True:
                try:
                    # Load one batch at a time
                    padded_embeddings = pickle.load(f)
                    
                    # Apply mean pooling to the batch
                    #pooled_embeddings = mean_pooling(padded_embeddings)
                    all_embd.append(padded_embeddings)
                
                except EOFError:
                    # End of file reached
                    break

    all_txt = np.array(all_txt)
    all_embd = np.concatenate(all_embd, axis=0)
    return all_embd, all_txt

def main():

    directory_path = "output_03"

    all_embd, all_txt = get_embed_txt(directory_path)

    tops = get_tops_indices(all_embd,[0,12,29])

    # Configuration
    cache_dir = '/scratch/project_2000539/maryam/embed/.cache'

    # Initialize tokenizer and model
    model_name = "Salesforce/SFR-Embedding-Mistral" #"dunzhang/stella_en_1.5B_v5"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.float16).cuda()

    for head_idx, (top_5, low_5) in enumerate(tops):

        print("Head: ", head_idx)
        for idx, (article, doc_embedding) in enumerate(zip(all_txt[top_5], all_embd[top_5])):
            print(f"\nProcessing Article {idx+1}:")
            # Step 1: Get word embeddings
            word_embs = get_word_embeddings(article, tokenizer, model)
            
            # Step 2: Calculate word contributions
            contributions = calculate_word_contributions(word_embs, doc_embedding)
            
            # Step 3: Rank words by dot product and cosine similarity
            sorted_by_dot = sorted(contributions.items(), key=lambda x: x[1]['dot_product'], reverse=True)
            sorted_by_cosine = sorted(contributions.items(), key=lambda x: x[1]['cosine_similarity'], reverse=True)
            
            # Step 4: Display top contributing words
            print(f"\nTop Words by Dot Product:")
            for word, scores in sorted_by_dot[:5]:  # Top 5 words
                print(f"Word: {word}, Dot Product: {scores['dot_product']:.4f}")
            
            print(f"\nTop Words by Cosine Similarity:")
            for word, scores in sorted_by_cosine[:5]:  # Top 5 words
                print(f"Word: {word}, Cosine Similarity: {scores['cosine_similarity']:.4f}")

main()