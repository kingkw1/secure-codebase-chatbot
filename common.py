import subprocess
import requests
import json
import re
import logging
import os
from dotenv import load_dotenv

DEFAULT_CODEBASE_DIRECTORY = os.path.join(os.path.dirname(__file__), 'sample_repo')  # Specify the directory containing the codebase
DEFAULT_METADATA_FILE = ".metadata.json"
DEFAULT_INDEX_FILE = ".index.faiss"


def get_codebase_directory():
    load_dotenv()
    codebase_directory = os.getenv('CODEBASE_DIRECTORY', DEFAULT_CODEBASE_DIRECTORY)
    
    if not os.path.isabs(codebase_directory):
        codebase_directory = os.path.join(os.path.dirname(__file__), codebase_directory)
    
    if not os.path.exists(codebase_directory):
        logging.error(f"Specified codebase directory does not exist: {codebase_directory}. Using default.")
        return DEFAULT_CODEBASE_DIRECTORY
    
    return codebase_directory

CODEBASE_DIRECTORY = get_codebase_directory()

def find_file(directory, filename):
    for root, dirs, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"{filename} not found in {directory}")


def find_metadata_file(directory=CODEBASE_DIRECTORY, filename=DEFAULT_METADATA_FILE):
    return find_file(directory, filename)


def find_index_file(directory=CODEBASE_DIRECTORY, filename=DEFAULT_INDEX_FILE):
    return find_file(directory, filename)


def get_meta_paths(directory=CODEBASE_DIRECTORY, metadata_filename=DEFAULT_METADATA_FILE, index_filename=DEFAULT_INDEX_FILE, generate_new=False):
    try: 
        metadata_file = find_metadata_file(directory, metadata_filename)
        index_file = find_index_file(directory, index_filename)
    except FileNotFoundError as e:
        if generate_new:
            metadata_file = os.path.join(directory, metadata_filename)
            index_file = os.path.join(directory, index_filename)
        else:
            logging.error(f"Error finding metadata or index file: {e}")
            raise Exception("Metadata or index file not found. Please ensure they exist in the specified directory.")
    
    return metadata_file, index_file


def strip_ansi_codes(text):
    return re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)


def correct_spacing_issues(text):
    # # Substitute multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    # Replace sequences of single spaces within words
    # Find instances where there's a single letter followed by a space and another letter
    # text = re.sub(r'\b(\w)\s(\w)\b', r'\1\2', text)  # Joins single characters separated by spaces

    # # Replace additional cases with common letter pairings or separations
    # text = re.sub(r'(\w)\s+([a-zA-Z])', r'\1\2', text)

    # return re.sub(r'\s+', ' ', complete_response)
    return text


def query_ollama(prompt, model_name="codellama", api_mode=True, max_tokens=100, temperature=0.2):
    """
    Query the OLLAMA API or subprocess to generate a response for the given prompt using the specified model.
    """
    if api_mode:
        try:
            url = "http://localhost:11434/api/generate"
            # Log the call details
            logging.info(f"Calling Ollama API at {url} with model {model_name} and prompt: {prompt[:50]}...")
            response = requests.post(
                url,
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            )
            if response.status_code == 200:
                response_lines = response.content.decode('utf-8').split('\n')
                responses = [json.loads(line).get("response", "").strip() for line in response_lines if line.strip()]
                complete_response = ' '.join(responses).strip()
                cleaned_response = correct_spacing_issues(complete_response)
                return cleaned_response
            else:
                logging.error(f"Ollama API Error: {response.status_code} - {response.text}")
                return "LLM API returned an error. Please check the configuration."  # Fallback message
        except requests.RequestException as e:
            logging.error(f"Ollama Request Exception: {e}")
            return "Network error occurred when contacting the LLM API."  # Fallback message
        except json.JSONDecodeError as e:
            logging.error(f"Ollama JSON Decode Error: {e}")
            return "Error decoding the response from the LLM API."  # Fallback message
    else:
        # Use the subprocess method
        try:
            process = subprocess.Popen(
                ['ollama', 'run', model_name, prompt],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW  # For Windows
            )

            stdout, stderr = process.communicate()
            response = stdout.decode('utf-8').strip()
            response = strip_ansi_codes(response)
            stderr_output = strip_ansi_codes(stderr.decode('utf-8').strip())

            if stderr_output:
                logging.error(f"STDERR: {stderr_output}")

            # Normalize spaces in response text
            return re.sub(r'\s+', ' ', response) if response else "Error: No response from subprocess."  # Fallback message

        except Exception as e:
            logging.error(f"Subprocess Exception: {e}")
            return "Error occurred while querying the local LLM subprocess."  # Fallback message
