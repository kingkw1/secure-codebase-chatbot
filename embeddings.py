import json
import numpy as np
import faiss
import torch
from transformers import RobertaTokenizer, RobertaModel
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import metadata_path, index_path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Initialize the CodeBERT tokenizer and model
model_name = "microsoft/codebert-base"  # CodeBERT base model for code embeddings
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)

# Load the metadata JSON
def load_metadata(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

# Extract textual data for embedding
def extract_text_data(metadata):
    text_data = [metadata["readme_summary"]]
    text_data += [
        func["comment"] for file in metadata["files"]
        for func in file["structure"] if "comment" in func
    ]
    return text_data

# Generate embeddings for text data using CodeBERT
def generate_embeddings(text_data):
    embeddings = []
    for text in text_data:
        # Tokenize the text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Get the model's embeddings (last hidden state)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # We use the mean of the last hidden state for the embeddings
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        embeddings.append(embedding)
    
    return np.array(embeddings)

# Create FAISS index and save it
def create_faiss_index(embeddings, output_path):
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance for nearest neighbor search
    index.add(embeddings.astype('float32'))  # Add embeddings to the index
    faiss.write_index(index, output_path)
    return index

# Load FAISS index
def load_faiss_index(file_path):
    return faiss.read_index(file_path)

# Perform search on the FAISS index
def search_index(query, model, index, text_data, top_k=5):
    # Generate embedding for the query
    query_embedding = generate_embeddings([query])[0]
    
    # Perform the search on the FAISS index
    distances, indices = index.search(query_embedding[np.newaxis, :], top_k)
    
    # Return the matched text and the distances
    return [(text_data[i], distances[0][j]) for j, i in enumerate(indices[0])]

# Main script
def main():
    # Step 1: Load metadata and extract text data
    metadata = load_metadata(metadata_path)
    text_data = extract_text_data(metadata)

    # Step 2: Generate embeddings and create FAISS index
    embeddings = generate_embeddings(text_data)
    index = create_faiss_index(embeddings, index_path)

    # Step 3: Query the index
    query = "Describe the function that calculates mean"  # Example query
    results = search_index(query, model, index, text_data)
    
    # Display results
    for text, distance in results:
        print(f"Matched text: {text}\nDistance: {distance}\n")

if __name__ == "__main__":
    main()
