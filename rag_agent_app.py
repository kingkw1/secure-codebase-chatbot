import faiss
import json
import subprocess
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import sys
import os
from torch.nn.functional import softmax
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import metadata_path, index_path, query_ollama
from models import embedding_model, embedding_tokenizer, cross_encoder_model, cross_encoder_tokenizer, agent_model_name

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask application
app = Flask(__name__)
CORS(app)

def load_metadata(file_path):
    """
    Load metadata from a JSON file.
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def load_embeddings(index_path):
    """
    Load the FAISS index from the specified file path.
    """
    assert os.path.exists(index_path), f"Index file not found: {index_path}"
    return faiss.read_index(index_path)


def generate_embedding(query):
    inputs = embedding_tokenizer(query, return_tensors="pt")
    outputs = embedding_model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.detach().numpy()


def strip_ansi_codes(text):
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)


def custom_score(query, matched_file, distance):
    """
    Define a hierarchical context-aware scoring function
    """
    query_keywords = set(re.findall(r'\w+', query.lower()))
    func_matches = [
        func for func in matched_file['structure'] 
        if any(keyword in func.get('name', '').lower() for keyword in query_keywords)
    ]
    keyword_score = len(func_matches)

    # Define additional contextual scoring
    file_path_context_score = sum(
        [1 for keyword in query_keywords if keyword in matched_file['file_path'].lower()]
    )
    type_context_score = 1 if matched_file.get('type', '').lower() in query_keywords else 0

    # Combine context scores with distance for a final custom score
    custom_score = -distance + (keyword_score * 0.5) + (file_path_context_score * 0.3) + (type_context_score * 0.2)
    return custom_score, func_matches


def multi_stage_retrieval_with_cross_encoder_reranking(query_embedding, index, metadata, query, first_stage_k=10, second_stage_k=5, distance_threshold=None):
    """
    Perform multi-stage retrieval with cross-encoder reranking
    """
    # Stage 1: Retrieve initial results using FAISS
    try:
        D, I = index.search(query_embedding.reshape(1, -1), first_stage_k)
        logging.info(f"FAISS distances (D): {D}")
        logging.info(f"FAISS indices (I): {I}")
    except Exception as e:
        logging.error(f"Error during FAISS search: {e}")
        return []

    candidates = []

    # Stage 2: Rerank top-k results using a cross-encoder model
    for dist, idx in zip(D[0], I[0]):
        if not (0 <= idx < len(metadata['files'])):
            logging.warning(f"Index {idx} out of bounds for metadata.")
            continue
        if distance_threshold is not None and dist >= distance_threshold:
            logging.info(f"Skipping result at index {idx} due to distance threshold: {dist} >= {distance_threshold}")
            continue

        matched_file = metadata['files'][idx]
        logging.info(f"Processing metadata entry: {matched_file['file_path']} with distance {dist}")

        score, matched_funcs = custom_score(query, matched_file, dist)
        logging.info(f"Custom score for {matched_file['file_path']}: {score}, Matched functions: {[func['name'] for func in matched_funcs]}")

        # Log function-level details for debugging
        for func in matched_funcs:
            func_name = func.get('name', 'Unnamed function')
            func_code = func.get('code', '')
            func_comment = func.get('comment', '')
            logging.info(f"Matched function: {func_name}, Code: {func_code[:30]}..., Comment: {func_comment[:30]}...")  # Log partial code/comment for brevity

        if matched_funcs:
            for func in matched_funcs:
                func_text = func.get('code', '') + " " + func.get('comment', '')
                # Calculate cross-encoder relevance score
                cross_encoder_score_val = cross_encoder_score(query, func_text)
                
                # Combine FAISS distance score and cross-encoder score
                final_score = score + cross_encoder_score_val  # Adjust weighting as needed
                candidates.append((final_score, idx, func, matched_file))

    # Sort by final combined score
    reranked_results = sorted(candidates, key=lambda x: x[0], reverse=True)[:second_stage_k]
    return reranked_results


def find_closest_embeddings(query_embedding, index, k=5, distance_threshold=5):
    """
    Find the closest embeddings to the query embedding using the FAISS index.
    """
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


def cross_encoder_score(query, candidate_text):
    """
    Compute the relevance score between a query and a candidate text using a cross-encoder model.
    """
    # Tokenize the input pair
    inputs = cross_encoder_tokenizer(query, candidate_text, return_tensors="pt", truncation=True, padding=True)
    # Get the model's output
    outputs = cross_encoder_model(**inputs)
    scores = softmax(outputs.logits, dim=-1)  # Apply softmax to get probabilities
    
    # Check if the output has two logits (binary classification)
    if scores.shape[1] == 2:
        # Return the relevance score (second index for relevance, first index for irrelevance)
        return scores[0][1].item()  # Assuming the second index is the relevance score
    else:
        # If only one score, return it (in case of single-label classification)
        return scores[0][0].item()


@app.route('/query', methods=['POST'])
def handle_query():
    try:
        user_query = request.json.get('query', '')
        logging.info(f"Received query: {user_query}")

        metadata = load_metadata(metadata_path)
        index = load_embeddings(index_path)
        query_embedding = generate_embedding(user_query).astype('float32')

        # Use multi-stage retrieval with cross-encoder reranking
        scored_results = multi_stage_retrieval_with_cross_encoder_reranking(query_embedding, index, metadata, user_query)

        if not scored_results:
            logging.warning("No valid indices found.")
            return jsonify({"error": "No matching functions found."})

        responses = []
        for final_score, idx, func, matched_file in scored_results:
            func_name = func.get('name', 'Unnamed function')
            func_comment = func.get('comment', '')
            func_code = func.get('code', '')
            prompt = (
                f"Explain the purpose of the '{func_name}' function within this codebase. "
                f"Here is the function's definition: {func_code}. "
                f"Here is the function's comments: {func_comment}. "
                f"Query: {user_query}"
            )

            response = query_ollama(prompt, model_name=agent_model_name)
            responses.append({
                "function": func_name,
                "file": matched_file['file_path'],
                "score": final_score,
                "response": response
            })
            logging.info(f"Prompted {agent_model_name} with: {prompt}, received: {response}")

        return jsonify(responses)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)  # Change port if needed for Open-WebUI integration
