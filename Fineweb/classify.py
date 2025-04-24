import os
import json
import openai
import csv
from datasets import load_from_disk
from dotenv import load_dotenv

load_dotenv()

# OpenAI API key (set this up securely, e.g., using environment variables)
client = openai.Client(api_key=os.environ["OPENAI_API_KEY"])

# Load dataset
dataset = load_from_disk("/scratch/project_2000539/maryam/fineweb_dataset")

# Load category list
categories_file = "images-txts/FineWeb/categories-narrowed.txt"
with open(categories_file, "r", encoding="utf-8") as f:
    categories = [line.strip() for line in f.readlines()]

# Output storage (adjust as needed)
output_file = "classified_topics_narrowed.jsonl"

def classify_text(text):
    """Send text to OpenAI and get the best matching category."""
    prompt = f"""
    Given the following text:
    ""
    {text[:800]}  # Limit to avoid excessive token usage
    ""
    Choose the most suitable category from the following numbered list:  {', '.join(categories)}. 
    Respond with the full entry exactly as written, including its number and commas. 
    Do not create new categories. Do not change the format.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Adjust model if needed
            messages=[{"role": "system", "content": "You are an expert text classifier."},
                      {"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error processing text: {e}")
        return "Unknown"

# Process dataset and classify topics
classified_data = []
count = 0
for file_index in range(0,len(dataset)):
    text = dataset[file_index]['text']
    doc_id = dataset[file_index].get('id', file_index)  # Store ID if available
    topic = classify_text(text)
    classified_data.append({"file_index": file_index, "id": doc_id, "text": text[:200], "topic": topic})  # Store snippet
    count += 1
    if count > 3000:
        break
    # Save incrementally
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(classified_data[-1]) + "\n")

print(f"Classification complete. Results saved to {output_file}")
