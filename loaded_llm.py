import faiss
import numpy as np
import json
import subprocess

# Load metadata from JSON file
def load_metadata(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Load embeddings from FAISS index file
def load_embeddings(index_path):
    # Load the FAISS index
    index = faiss.read_index(index_path)
    return index

# Query CodeLlama model using Ollama
def query_codellama(prompt):
    result = subprocess.run(['ollama', 'run', 'codellama', prompt], capture_output=True, text=True, encoding='utf-8')
    return result.stdout.strip()

# Find the closest embeddings for the query
def find_closest_embeddings(query_embedding, index, k=5):
    D, I = index.search(query_embedding.reshape(1, -1), k)  # Reshape for FAISS input
    return I[0]  # Return the indices of the closest embeddings

# Main function to run the script
def main():
    # Load metadata and embeddings
    metadata = load_metadata(r'C:\Users\kingk\OneDrive\Documents\Projects\repo_chatbot\sample_metadata\test_metadata.json')
    index = load_embeddings(r'C:\Users\kingk\OneDrive\Documents\Projects\repo_chatbot\sample_metadata\embedding_index.faiss')

    # Example input for querying
    query_input = "Explain the purpose of the mean function."
    
    # (Here you would generate the query embedding based on the input)
    query_embedding = np.random.random(index.d).astype('float32')  # Replace with actual embedding generation logic

    # Find the closest embeddings
    closest_indices = find_closest_embeddings(query_embedding, index)

    # Ensure we don't go out of bounds when accessing metadata
    valid_indices = [idx for idx in closest_indices if 0 <= idx < len(metadata['files'])]

    if not valid_indices:
        print("No valid indices found. Please check your metadata and embedding files.")
        return

    for idx in valid_indices:
        matched_file = metadata['files'][idx]
        for func in matched_file['structure']:
            # Change the prompt to make it more relevant
            prompt = f"What is the purpose of the function {func['name']}?"
            print(f"Prompting CodeLlama with: {prompt}")  # Debugging output
            response = query_codellama(prompt)
            print(f"Response for {func['name']}: {response}")

if __name__ == "__main__":
    main()
