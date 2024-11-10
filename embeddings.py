import json
import numpy as np
import faiss
import torch
from transformers import RobertaTokenizer, RobertaModel
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyzer import generate_code_structure, generate_comment_with_model, generate_readme_summary, identify_dependencies, parse_code_structure
from common import metadata_path, index_path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Initialize the CodeBERT tokenizer and model
model_name = "microsoft/codebert-base"  # CodeBERT base model for code embeddings
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)


def load_metadata(file_path):
    """
    Load metadata from a JSON file.
    """
    with open(file_path, "r") as file:
        return json.load(file)


def extract_text_data(metadata):
    """
    Extracts the text data from the metadata for generating embeddings.
    """
    text_data = [metadata["readme_summary"]]
    text_data += [
        func["comment"] for file in metadata["files"]
        for func in file["structure"] if "comment" in func
    ]
    return text_data


def generate_embeddings(text_data):
    """
    Generate embeddings for the given text data using the CodeBERT model.
    """
    embeddings = []
    for text in text_data:
        # Tokenize the text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Get the model's embeddings (last hidden state)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # We use the mean of the last hidden state for the embeddings
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        embeddings.append(embedding)
    
    return np.array(embeddings)


def create_faiss_index(embeddings):
    """
    Create a FAISS index for the given embeddings.
    """
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance for nearest neighbor search
    index.add(embeddings.astype('float32'))  # Add embeddings to the index
    return index


def save_faiss_index(index, output_path):
    """
    Save the FAISS index to the specified file path.
    """
    faiss.write_index(index, output_path)


def load_faiss_index(file_path):
    """
    Load the FAISS index from the specified file path.
    """
    return faiss.read_index(file_path)


def search_index(query, model, index, text_data, top_k=5):
    """
    Perform a search on the FAISS index using the query and retrieve the top-k results.
    """
    # Generate embedding for the query
    query_embedding = generate_embeddings([query])[0]
    
    # Perform the search on the FAISS index
    distances, indices = index.search(query_embedding[np.newaxis, :], top_k)
    
    # Return the matched text and the distances
    return [(text_data[i], distances[0][j]) for j, i in enumerate(indices[0])]


def add_comment_to_code(file_path, code_structure):
    """
    Add comments to the code structure in the specified file.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    for code_block in code_structure:
        comment = generate_comment_with_model(code_block["code"])
        # Insert the comment in the code structure right before the start line
        lines.insert(code_block["start_line"] - 1, f"# {comment}\n")

    # Write the updated lines back to the file or save as a new file
    with open("updated_" + file_path, "w") as file:
        file.writelines(lines)


def extract_repository_metadata(directory='.'):
    """
    Extract metadata for a code repository located in the specified directory.
    """
    repository_metadata = {
        "directory": directory,
        "files": [],
        "dependencies": identify_dependencies(directory)
    }

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                file_structure = parse_code_structure(file_path)

                # Add comments generated for each function/class in the file
                for item in file_structure:
                    item["comment"] = generate_comment_with_model(item["code"])

                repository_metadata["files"].append({
                    "file_path": os.path.relpath(file_path, directory),
                    "structure": file_structure
                })

    # Generate README summary and add it to metadata
    code_structure_summary = generate_code_structure(directory)
    repository_metadata["readme_summary"] = generate_readme_summary(code_structure_summary)

    return repository_metadata


def save_metadata(metadata, output_path="metadata.json"):
    """
    Save the repository metadata to a JSON file.
    """
    with open(output_path, "w") as file:
        json.dump(metadata, file, indent=2)


def main():
    """
    Main function to extract repository metadata, generate embeddings, and create the FAISS index.
    """

    # Step 1: Extract repository metadata
    print("Extracting repository metadata...")
    metadata = extract_repository_metadata()
    save_metadata(metadata)
    print("Metadata saved to metadata.json\n")

    # Step 2: Load metadata and extract text data
    text_data = extract_text_data(metadata)

    # Step 3: Generate embeddings 
    print("Generating embeddings for text data...")
    embeddings = generate_embeddings(text_data)
    print("Embeddings generated successfully!\n")

    # Step 4: Create and save the FAISS index
    print("Creating the FAISS index...")
    index = create_faiss_index(embeddings, index_path)
    save_faiss_index(index, index_path)
    print("Index created successfully!\n")


if __name__ == "__main__":
    main()
