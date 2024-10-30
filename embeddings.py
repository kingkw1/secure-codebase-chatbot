import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Define paths for metadata JSON and FAISS index files
metadata_file_path = r"C:\Users\kingk\OneDrive\Documents\Projects\repo_chatbot\sample_metadata\test_metadata.json"
faiss_index_path = r"C:\Users\kingk\OneDrive\Documents\Projects\repo_chatbot\sample_metadata\embedding_index.faiss"

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

# Generate embeddings for text data
def generate_embeddings(text_data, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text_data, convert_to_tensor=True)

# Create FAISS index and save it
def create_faiss_index(embeddings, output_path):
    embedding_array = np.vstack([embedding.numpy() for embedding in embeddings])
    index = faiss.IndexFlatL2(embedding_array.shape[1])
    index.add(embedding_array)
    faiss.write_index(index, output_path)
    return index

# Load FAISS index
def load_faiss_index(file_path):
    return faiss.read_index(file_path)

# Perform search on the FAISS index
def search_index(query, model, index, text_data, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True).numpy()
    distances, indices = index.search(query_embedding[np.newaxis, :], top_k)
    return [(text_data[i], distances[0][j]) for j, i in enumerate(indices[0])]

# Main script
def main():
    # Step 1: Load metadata and extract text data
    metadata = load_metadata(metadata_file_path)
    text_data = extract_text_data(metadata)

    # Step 2: Generate embeddings and create FAISS index
    embeddings = generate_embeddings(text_data)
    index = create_faiss_index(embeddings, faiss_index_path)

    # Step 3: Query the index
    query = "Describe the function that calculates mean"  # Example query
    model = SentenceTransformer("all-MiniLM-L6-v2")
    results = search_index(query, model, index, text_data)
    
    # Display results
    for text, distance in results:
        print(f"Matched text: {text}\nDistance: {distance}\n")

if __name__ == "__main__":
    main()
