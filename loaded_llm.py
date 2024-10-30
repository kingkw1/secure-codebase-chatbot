import faiss
import numpy as np
import json
import subprocess
from flask import Flask, request, jsonify

metadata_path = r'C:\Users\kingk\OneDrive\Documents\Projects\repo_chatbot\sample_metadata\test_metadata.json'
index_path = r'C:\Users\kingk\OneDrive\Documents\Projects\repo_chatbot\sample_metadata\embedding_index.faiss'


# Initialize Flask application
app = Flask(__name__)

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

@app.route('/query', methods=['POST'])
def handle_query():
    user_query = request.json.get('query', '')
    
    # Load metadata and embeddings inside the request to avoid global state issues
    metadata = load_metadata('test_metadata.json')
    index = load_embeddings('embedding_index.faiss')

    # Here you would typically generate your query embedding
    query_embedding = np.random.random(index.d).astype('float32')  # Placeholder for actual embedding logic

    # Find the closest embeddings
    closest_indices = find_closest_embeddings(query_embedding, index)
    valid_indices = [idx for idx in closest_indices if 0 <= idx < len(metadata['files'])]

    responses = []
    for idx in valid_indices:
        matched_file = metadata['files'][idx]
        for func in matched_file['structure']:
            prompt = f"What is the purpose of the function {func['name']}?"
            response = query_codellama(prompt)
            responses.append({"function": func['name'], "response": response})

    return jsonify(responses)

# Main function to run the script
def main():
    # Load metadata and embeddings
    metadata = load_metadata(metadata_path)
    index = load_embeddings(index_path)

    # Example input for querying
    query_input = "Explain the purpose of the mean function."
    
    # (Here you would generate the query embedding based on the input)
    query_embedding = np.random.random(index.d).astype('float32')  # Replace with actual embedding generation logic

    # query_embedding = query_codellama(query_input)  # Use your LLM to get the embedding
    # query_embedding = np.array(query_embedding).astype('float32')  # Ensure it's the correct type

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
    # app.run(host='0.0.0.0', port=5000)  # Change to a different port, e.g., 5001
