import sys
import os
import numpy as np

# Add the parent directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_agent_app import query_ollama, load_embeddings, load_metadata, find_closest_embeddings, metadata_path, index_path


# Main function to run the script
def test_embeddings():
    # Load metadata and embeddings
    metadata = load_metadata(metadata_path)
    index = load_embeddings(index_path)

    # Example input for querying
    query_input = "Explain the purpose of the mean function."
    
    # Here you would generate the query embedding based on the input
    # Replace the random embedding with the actual embedding generation logic
    query_embedding = np.random.random(index.d).astype('float32')  # Example placeholder

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
            # Use the query_input for context in your prompt
            prompt = f"{query_input} What is the purpose of the function {func['name']}?"
            print(f"Prompting CodeLlama with: {prompt}")  # Debugging output
            response = query_ollama(prompt)
            print(f"Response for {func['name']}: {response}")
        matched_file = metadata['files'][idx]
        for func in matched_file['structure']:
            # Change the prompt to make it more relevant
            prompt = f"What is the purpose of the function {func['name']}?"
            print(f"Prompting CodeLlama with: {prompt}")  # Debugging output
            response = query_ollama(prompt)
            print(f"Response for {func['name']}: {response}")

if __name__ == "__main__":
    test_embeddings()