import faiss
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import sys
import os
import requests
from torch.nn.functional import softmax
from collections import deque
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from transformers import GPT2Tokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import query_ollama, get_meta_paths
from models import embedding_model, embedding_tokenizer, cross_encoder_model, cross_encoder_tokenizer, agent_model_name


# Input variables  ------------------
test_function_penalty = 100
MAX_HISTORY_LENGTH = 5
ENABLE_CHAT_HISTORY = False
ENABLE_AZURE_SEMANTIC_SEARCH = False  # Toggle for Azure Semantic Search
# -----------------------------------

# Initialize Flask application
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO) # Setup logging

# Initialize chat history
chat_histories = {}

# Get paths for metadata and index files
metadata_path, index_path = get_meta_paths()

# Azure Search configuration
AZURE_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX_NAME")
AZURE_SEARCH_API_KEY = os.environ.get("AZURE_SEARCH_ADMIN_KEY")
AZURE_SEARCH_API_VERSION = "2023-07-01-Preview"

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Tokenizer to count tokens (adjust based on your LLM)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

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

# Helper function to check if the query is likely unrelated to the codebase or context
def is_unrelated_query(query, metadata):
    # Check for the presence of codebase-related keywords in the query
    codebase_keywords = [
        func.get("name", "").lower() for file in metadata["files"] for func in file.get("structure", [])
    ]
    query_keywords = set(word.lower() for word in re.findall(r"\w+", query))

    # If there are no relevant codebase terms in the query, it's likely unrelated to the codebase
    return not any(keyword in query_keywords for keyword in codebase_keywords)


def azure_semantic_search(query):
    url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX}/docs/search?api-version={AZURE_SEARCH_API_VERSION}"
    headers = {"Content-Type": "application/json", "api-key": AZURE_SEARCH_API_KEY}
    body = {"search": query, "queryType": "semantic", "top": 5}

    try:
        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Azure Search API error: {e.response.text if e.response else e}")
        return None
    
        
def azure_search_query(query, top_k=10):
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_SEARCH_API_KEY
    }

    payload = {
        "search": query,
        "top": top_k,
        "queryType": "semantic",
        "semanticConfiguration": "default",
        "captions": "extractive",
    }

    url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX}/docs/search?api-version={AZURE_SEARCH_API_VERSION}"

    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code != 200:
        logging.error(f"Azure Search API error: {response.status_code} {response.text}")
        return []
    
    return response.json().get("value", [])

# Helper function to count tokens in the history
def count_tokens_in_history(history):
    history_text = "\n".join([entry['content'] for entry in history])
    return len(tokenizer.encode(history_text))

# Helper function to truncate history based on token length
def truncate_history(user_id, max_tokens=3000):
    history = chat_histories.get(user_id, [])
    total_tokens = sum([len(entry['content'].split()) for entry in history])

    # Truncate history if total tokens exceed max_tokens
    while total_tokens > max_tokens:
        removed_entry = chat_histories[user_id].popleft()
        total_tokens -= len(removed_entry['content'].split())
        logging.info(f"Truncating history: Removed entry '{removed_entry['content']}' to stay within token limit.")
    return history


@app.route('/query', methods=['POST'])
def handle_query():
    try:
        user_query = request.json.get('query', '').strip()
        user_id = request.json.get('user_id', 'default_user')
        clear_history = request.json.get('clear_history', False)

        logging.info(f"Received query from {user_id}: {user_query} (clear_history={clear_history})")

        if clear_history:
            logging.info(f"Clearing chat history for user: {user_id}")
            chat_histories[user_id] = deque(maxlen=MAX_HISTORY_LENGTH)
            return jsonify({"response": "Chat history cleared."})

        if user_id not in chat_histories:
            chat_histories[user_id] = deque(maxlen=MAX_HISTORY_LENGTH)

        chat_histories[user_id].append({'role': 'user', 'content': user_query})

        # Truncate history if it exceeds the token limit
        chat_histories[user_id] = truncate_history(user_id, max_tokens=3000)

        # Integrate ENABLE_CHAT_HISTORY toggle when building context
        if ENABLE_CHAT_HISTORY:
            history_context = "\n".join([f"{entry['role']}: {entry['content']}" for entry in list(chat_histories[user_id])[-3:]])
        else:
            history_context = ""
            
        if ENABLE_AZURE_SEMANTIC_SEARCH:
            # Step 1: Azure Semantic Search
            search_results = azure_search_query(user_query, top_k=10)
            logging.info(f"Azure Search returned {len(search_results)} documents.")
        else:
            # Step 1: Local keyword + embedding search
            metadata = load_metadata(metadata_path)
            index = load_embeddings(index_path)
            query_embedding = generate_embedding(user_query).astype('float32')
            
            # Validate metadata now as a dict containing "files"
            if not metadata or not isinstance(metadata, dict) or "files" not in metadata:
                logging.error("Metadata is empty or does not contain 'files'. Aborting search.")
                return jsonify({"response": "Error: No valid metadata found."})
            
            # Check if the FAISS index is empty using ntotal instead of len(index)
            if not index or not hasattr(index, "ntotal") or index.ntotal == 0:
                logging.error("Index is empty. Aborting search.")
                return jsonify({"response": "Error: No valid embeddings index found."})

            scored_results = multi_stage_retrieval_with_cross_encoder_reranking(
                query_embedding, index, metadata, user_query
            )
            logging.info(f"Local multi-stage retrieval returned {len(scored_results)} documents.")

            # Check if scored_results is valid
            if not scored_results:
                logging.warning("No results found in scored_results.")
            
            # Adapt scored_results to a common format like search_results
            search_results = []
            for final_score, idx, func, matched_file in scored_results:
                # Validate that the index is within bounds
                if idx >= len(metadata):
                    logging.error(f"Index {idx} is out of bounds for metadata of length {len(metadata)}.")
                    continue  # Skip this result

                # Ensure matched_file is valid
                if not isinstance(matched_file, dict) or 'file_path' not in matched_file:
                    logging.error(f"Invalid matched_file structure: {matched_file}")
                    continue  # Skip this result

                combined_doc = {
                    "name": func.get('name', ''),
                    "comment": func.get('comment', ''),
                    "code": func.get('code', ''),
                    "file_path": matched_file.get('file_path', ''),
                }
                search_results.append(combined_doc)

            logging.info(f"Processed {len(search_results)} valid search results.")

        # If no relevant documents, fallback to general LLM
        if not search_results:
            logging.info("No search results found, falling back to general LLM response.")
            logging.info(f"Calling query_ollama fallback with model: {agent_model_name} and context: {history_context}")
            
            # Ensure history context is not empty or invalid before sending it as a prompt
            if not history_context:
                logging.error("History context is empty, using default fallback prompt.")
                history_context = "No relevant context available."
            
            general_response = query_ollama(history_context, model_name=agent_model_name)
            if general_response is None or not general_response.strip():
                general_response = "LLM endpoint returned no response. Please check configuration."
                logging.error("query_ollama returned None or empty response in fallback call")

            cleaned_general_response = strip_ansi_codes(general_response)
            chat_histories[user_id].append({'role': 'assistant', 'content': cleaned_general_response})
            return jsonify({"response": cleaned_general_response})

        # Step 2: Rerank (always done with cross-encoder, but heavier reranking when Azure is disabled)
        reranked = []
        rerank_top_n = 5 if ENABLE_AZURE_SEMANTIC_SEARCH else 10  # Larger rerank set when local
        for doc in search_results:
            doc_text = doc.get("code", "") + " " + doc.get("comment", "")
            score = cross_encoder_score(user_query, doc_text)
            reranked.append((score, doc))

        reranked = sorted(reranked, key=lambda x: x[0], reverse=True)[:rerank_top_n]

        # Step 3: Query Ollama/LLM with top reranked items
        final_responses = []
        for score, doc in reranked:
            # Prepare context (combine history context with relevant code and comment)
            prompt = f"{history_context}\nUser Query:\n{user_query}\nCode:\n{doc.get('code', '')}\nComment:\n{doc.get('comment', '')}"
            logging.info(f"Calling query_ollama with model: {agent_model_name} and context: {prompt}")
            
            # Here, make sure to pass user_query as 'prompt' and context for the LLM query
            response = query_ollama(prompt, model_name=agent_model_name)  # Corrected argument order
            if response is None or not response.strip():
                response = "LLM endpoint returned no response. Please check configuration."
                logging.error(f"query_ollama returned None or empty for context: {prompt}")
            cleaned_response = strip_ansi_codes(response)
            final_responses.append({
                "file_path": doc.get("file_path"),
                "function": doc.get("name"),
                "response": cleaned_response,
                "score": score
            })

        # Append the last top response to chat history
        if final_responses:
            chat_histories[user_id].append({'role': 'assistant', 'content': final_responses[0]['response']})

        return jsonify({"response": final_responses})

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
