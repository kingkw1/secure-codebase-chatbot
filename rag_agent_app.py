import faiss
import numpy as np
import json
import subprocess
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import sys
import os
from transformers import AutoTokenizer, AutoModel
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import metadata_path, index_path

llm_model_name = 'llama3.2'
embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask application
app = Flask(__name__)
CORS(app)

# Initialize embedding models
tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
llm_model = AutoModel.from_pretrained(embedding_model_name)

# Load metadata from JSON file
def load_metadata(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Load embeddings from FAISS index file
def load_embeddings(index_path):
    return faiss.read_index(index_path)

def generate_embedding(query):
    inputs = tokenizer(query, return_tensors="pt")
    outputs = llm_model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.detach().numpy()

def strip_ansi_codes(text):
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

# Query ollama  llm_model using Ollama
def query_ollama(prompt):
    try:
        process = subprocess.Popen(
            ['ollama', 'run', llm_model_name, prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW  # For Windows
        )
        
        stdout, stderr = process.communicate()  # Get output and error

        # Decode and clean up stdout
        response = stdout.decode('utf-8').strip()
        
        # Remove any terminal escape sequences from the response
        response = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', response)
        
        # Remove any ANSI color codes from the response
        response = strip_ansi_codes(response)

        # Remove response error messages related to console mode if present
        response = re.sub(
            r'failed to get console mode for stdout: The handle is invalid\.\n'
            r'failed to get console mode for stderr: The handle is invalid\.\n', 
            '', 
            response
        )

        # Remove stderr messages related to console mode if present
        stderr_output = stderr.decode('utf-8').strip()
        stderr_output = strip_ansi_codes(stderr_output)
        stderr_output = stderr_output.replace('⠙ ', '')  # Remove spinner or progress characters

        if stderr_output:
            logging.error(f"STDERR: {stderr_output}")

        # Concatenate cleaned response and filtered stderr output if needed
        final_response = response

        return final_response

    except Exception as e:
        return f"Error: {e}"

# Find the closest embeddings for the query
# def find_closest_embeddings(query_embedding, index, k=5):
#     D, I = index.search(query_embedding.reshape(1, -1), k)
#     return I[0]

def find_closest_embeddings(query_embedding, index, k=5, distance_threshold=5):
    # TODO: use plot embeddings functions to generate the distance threshold
    D, I = index.search(query_embedding.reshape(1, -1), k)
    if distance_threshold:
        # Select only indices in I where the corresponding distances in D are below the threshold
        I_filtered = I[0][D[0] < distance_threshold]
        if len(I_filtered) > 0:
            return I_filtered  # Return only the indices within the threshold
        else:
            return I[0]  # If none meet the threshold, return the original top-k indices
    return I[0]

@app.route('/query', methods=['POST'])
def handle_query():
    try:
        user_query = request.json.get('query', '')
        logging.info(f"Received query: {user_query}")
        
        # Load metadata and embeddings (for development; cache for production)
        metadata = load_metadata(metadata_path)
        index = load_embeddings(index_path)

        # Generate query embedding (update with meaningful embedding logic)
        # query_embedding = np.random.random(index.d).astype('float32')  # Placeholder logic
        query_embedding = generate_embedding(user_query).astype('float32')

        closest_indices = find_closest_embeddings(query_embedding, index)
        valid_indices = [idx for idx in closest_indices if 0 <= idx < len(metadata['files'])]

        if not valid_indices:
            logging.warning("No valid indices found.")
            return jsonify({"error": "No matching functions found."})

        # Extracting keywords from the user's query for more flexible function matching
        query_keywords = re.findall(r'\w+', user_query.lower())

        responses = []
        for idx in valid_indices:
            matched_file = metadata['files'][idx]

            # Add debug log to see all functions in the matched file
            logging.debug(f"Checking functions in file: {matched_file['file_path']}")

            # Match functions by checking if any keyword appears in the function name
            relevant_funcs = [
                func for func in matched_file['structure'] 
                if any(keyword in func.get('name', '').lower() for keyword in query_keywords)
            ]

            # Log matched functions for further troubleshooting
            if relevant_funcs:
                logging.info(f"Relevant functions found: {[func.get('name', 'Unnamed function') for func in relevant_funcs]}")
            else:
                logging.info("No relevant functions found in this file.")

            # Skip if no relevant functions were found
            if not relevant_funcs:
                continue

            for func in relevant_funcs:
                # Prepare a detailed prompt specifically about the matched function
                func_name = func.get('name', 'Unnamed function')
                func_comment = func.get('comment', '')  # If details like comments or docstrings are available
                func_code = func.get('code', '')  # If code snippets are available
                prompt = (
                    f"Explain the purpose of the '{func_name}' function within this codebase. "
                    f"Here is the function's definition: {func_code}. "
                    f"Here is the function's comments: {func_comment}. "
                    f"Query: {user_query}"
                )

                response = query_ollama(prompt)
                responses.append({"function": func_name, "response": response})
                logging.info(f"Prompted {llm_model_name} with: {prompt}, received: {response}")

        return jsonify(responses)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)  # Change port if needed for Open-WebUI integration
