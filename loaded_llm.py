import faiss
import numpy as np
import json
import subprocess
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

# Setup logging
logging.basicConfig(level=logging.INFO)

metadata_path = r'C:\Users\kingk\OneDrive\Documents\Projects\repo_chatbot\sample_metadata\test_metadata.json'
index_path = r'C:\Users\kingk\OneDrive\Documents\Projects\repo_chatbot\sample_metadata\embedding_index.faiss'

# Initialize Flask application
app = Flask(__name__)
CORS(app)

# Load metadata from JSON file
def load_metadata(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Load embeddings from FAISS index file
def load_embeddings(index_path):
    return faiss.read_index(index_path)

# Query CodeLlama model using Ollama
# TODO: Update this function to use the query_ollama function from common.py
# def query_codellama(prompt):
#     try:
#         process = subprocess.Popen(
#             ['ollama', 'run', 'codellama', prompt],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             creationflags=subprocess.CREATE_NO_WINDOW  # For Windows
#         )
#         stdout, stderr = process.communicate()  # Get output and error
#         if stderr:
#             print("Error:", stderr.decode('utf-8'))
#         return stdout.decode('utf-8').strip()
#     except Exception as e:
#         return f"Error: {e}"

def query_codellama(prompt):
    return f"Mock response for: {prompt}"

# Find the closest embeddings for the query
def find_closest_embeddings(query_embedding, index, k=5):
    D, I = index.search(query_embedding.reshape(1, -1), k)
    return I[0]

@app.route('/query', methods=['POST'])
def handle_query():
    user_query = request.json.get('query', '')
    logging.info(f"Received query: {user_query}")
    
    # Load metadata and embeddings (for development; cache for production)
    metadata = load_metadata(metadata_path)
    index = load_embeddings(index_path)

    # Generate query embedding (update with meaningful embedding logic)
    query_embedding = np.random.random(index.d).astype('float32')  # Placeholder logic

    closest_indices = find_closest_embeddings(query_embedding, index)
    valid_indices = [idx for idx in closest_indices if 0 <= idx < len(metadata['files'])]

    if not valid_indices:
        logging.warning("No valid indices found.")
        return jsonify({"error": "No matching functions found."})

    responses = []
    for idx in valid_indices:
        matched_file = metadata['files'][idx]
        for func in matched_file['structure']:
            prompt = f"What is the purpose of the function {func['name']}?"
            response = query_codellama(prompt)
            responses.append({"function": func['name'], "response": response})
            logging.info(f"Prompted CodeLlama with: {prompt}, received: {response}")

    return jsonify(responses)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)  # Change port if needed for Open-WebUI integration
