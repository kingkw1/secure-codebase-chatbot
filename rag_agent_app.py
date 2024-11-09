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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import metadata_path, index_path

model = 'llama3.2'

# Setup logging
logging.basicConfig(level=logging.INFO)

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

def strip_ansi_codes(text):
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

# Query ollama  model using Ollama
def query_ollama(prompt):
    try:
        process = subprocess.Popen(
            ['ollama', 'run', model, prompt],
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
def find_closest_embeddings(query_embedding, index, k=5):
    D, I = index.search(query_embedding.reshape(1, -1), k)
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
        query_embedding = np.random.random(index.d).astype('float32')  # Placeholder logic

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
                prompt = f"{user_query} The function is {func.get('name', 'Unnamed function')}."
                response = query_ollama(prompt)
                responses.append({"function": func.get('name', 'Unnamed function'), "response": response})
                logging.info(f"Prompted {model} with: {prompt}, received: {response}")

        return jsonify(responses)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)  # Change port if needed for Open-WebUI integration
