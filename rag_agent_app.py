import faiss
import json
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


test_function_penalty = 100

def load_metadata(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_embeddings(index_path):
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
    query_keywords = set(re.findall(r'\w+', query.lower()))
    exact_match_funcs = [
        func for func in matched_file['structure']
        if func.get('name', '').lower() == query.strip().lower()
    ]
    partial_match_funcs = [
        func for func in matched_file['structure']
        if any(keyword in func.get('name', '').lower() for keyword in query_keywords)
    ]

    func_matches = exact_match_funcs or partial_match_funcs
    keyword_score = len(exact_match_funcs) * 2.0 + len(partial_match_funcs) * 0.5

    file_path_context_score = sum(
        [1 for keyword in query_keywords if keyword in matched_file['file_path'].lower()]
    )
    type_context_score = 1 if matched_file.get('type', '').lower() in query_keywords else 0

    if 'test' not in query.lower():
        for func in func_matches:
            if 'test' in func.get('name', '').lower():
                logging.info(f"Penalizing test function: {func['name']} as query does not mention 'test'")
                keyword_score -= test_function_penalty

    custom_score = -distance + (keyword_score * 0.5) + (file_path_context_score * 0.3) + (type_context_score * 0.2)
    return custom_score, func_matches

def multi_stage_retrieval_with_cross_encoder_reranking(query_embedding, index, metadata, query, first_stage_k=10, second_stage_k=5, distance_threshold=None):
    try:
        # Step 1: Initial retrieval using FAISS
        D, I = index.search(query_embedding.reshape(1, -1), first_stage_k)
        logging.info(f"FAISS distances (D): {D}")
        logging.info(f"FAISS indices (I): {I}")
    except Exception as e:
        logging.error(f"Error during FAISS search: {e}")
        return []

    candidates = []
    test_function_penalty = 5  # Adjust penalty for test functions

    for dist, idx in zip(D[0], I[0]):
        # Skip invalid indices
        if not (0 <= idx < len(metadata['files'])):
            logging.warning(f"Index {idx} out of bounds for metadata.")
            continue
        
        # Skip entries exceeding the distance threshold
        if distance_threshold is not None and dist >= distance_threshold:
            logging.info(f"Skipping result at index {idx} due to distance threshold: {dist} >= {distance_threshold}")
            continue

        matched_file = metadata['files'][idx]
        logging.info(f"Processing metadata entry: {matched_file['file_path']} with distance {dist}")

        # Step 2: Compute custom score and retrieve matched functions
        score, matched_funcs = custom_score(query, matched_file, dist)
        logging.info(f"Custom score for {matched_file['file_path']}: {score}, Matched functions: {[func['name'] for func in matched_funcs]}")

        # Step 3: Apply query context filtering
        context_filtered_funcs = []
        for func in matched_funcs:
            # Penalize test functions if query does not mention 'test'
            if 'test' not in query.lower() and 'test' in func.get('name', '').lower():
                logging.info(f"Penalizing test function: {func['name']} as query does not mention 'test'")
                score -= test_function_penalty
                continue
            
            # Add functions that match the query context
            if func['name'].lower() in query.lower() or any(keyword in func.get('code', '').lower() for keyword in query.split()):
                context_filtered_funcs.append(func)

        # Skip files without relevant functions
        if not context_filtered_funcs:
            continue

        # Step 4: Calculate cross-encoder scores for each function
        for func in context_filtered_funcs:
            func_name = func.get('name', 'Unnamed function')
            func_code = func.get('code', '')
            func_comment = func.get('comment', '')
            logging.info(f"Matched function: {func_name}, Code: {func_code[:30]}..., Comment: {func_comment[:30]}...")

            func_text = func.get('code', '') + " " + func.get('comment', '')
            cross_encoder_score_val = cross_encoder_score(query, func_text)
            
            # Combine custom score and cross-encoder score
            final_score = score + cross_encoder_score_val
            candidates.append((final_score, idx, func, matched_file))

    # Step 5: Rerank results based on final score
    reranked_results = sorted(candidates, key=lambda x: x[0], reverse=True)[:second_stage_k]
    
    # Log reranked results
    for result in reranked_results:
        logging.info(f"Reranked Result - Function: {result[2]['name']}, Score: {result[0]}, File: {result[3]['file_path']}")
    
    return reranked_results

def find_closest_embeddings(query_embedding, index, k=5, distance_threshold=5):
    D, I = index.search(query_embedding.reshape(1, -1), k)
    if distance_threshold:
        I_filtered = I[0][D[0] < distance_threshold]
        if len(I_filtered) > 0:
            return I_filtered
        else:
            return I[0]
    return I[0]

def cross_encoder_score(query, candidate_text):
    inputs = cross_encoder_tokenizer(query, candidate_text, return_tensors="pt", truncation=True, padding=True)
    outputs = cross_encoder_model(**inputs)
    scores = softmax(outputs.logits, dim=-1)
    
    if scores.shape[1] == 2:
        return scores[0][1].item()
    else:
        return scores[0][0].item()

@app.route('/query', methods=['POST'])
def handle_query():
    try:
        user_query = request.json.get('query', '')
        logging.info(f"Received query: {user_query}")

        metadata = load_metadata(metadata_path)
        index = load_embeddings(index_path)
        query_embedding = generate_embedding(user_query).astype('float32')

        scored_results = multi_stage_retrieval_with_cross_encoder_reranking(query_embedding, index, metadata, user_query)

        if not scored_results:
            # Fallback for unrelated or unmatched queries
            logging.warning("No valid indices found.")
            fallback_response = [{
                "response": f"The query '{user_query}' does not appear to be related to the codebase. Please refine your query.",
                "function": None,
                "file": None,
                "score": None
            }]
            return jsonify(fallback_response), 200

        responses = []
        for final_score, idx, func, matched_file in scored_results:
            func_name = func.get('name', 'Unnamed function')
            func_comment = func.get('comment', '')
            func_code = func.get('code', '')
            prompt = (
                f"Explain the purpose of the '{func_name}' function within this codebase. "
                f"Function code: {func_code}. "
                f"Comments: {func_comment}. "
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
    app.run(host='0.0.0.0', port=5001)
