import json
import requests
import re
import os

metadata_path = os.path.join(os.path.dirname(__file__), 'sample_metadata', 'test_metadata.json')
index_path = os.path.join(os.path.dirname(__file__), 'sample_metadata', 'embedding_index.faiss')

import subprocess
import requests
import json
import re
import logging

def strip_ansi_codes(text):
    return re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)

def query_ollama(prompt, model_name="codellama", api_mode=True, max_tokens=100, temperature=0.2):
    """
    Query the OLLAMA API or subprocess to generate a response for the given prompt using the specified model.
    """
    if api_mode:
        # Use the HTTP API method
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            )

            if response.status_code == 200:
                # Split response by lines and parse JSON
                response_lines = response.content.decode('utf-8').split('\n')
                responses = [json.loads(line).get("response", "") for line in response_lines if line.strip()]
                complete_response = ' '.join(responses).strip()
                return re.sub(r'\s+', ' ', complete_response)
            else:
                logging.error(f"API Error: {response.status_code} - {response.reason}")
                return None
        except requests.RequestException as e:
            logging.error(f"Request Exception: {e}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"JSON Decode Error: {e}")
            return None

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

            # Log stderr output if present
            if stderr_output:
                logging.error(f"STDERR: {stderr_output}")

            return response

        except Exception as e:
            logging.error(f"Subprocess Exception: {e}")
            return None
