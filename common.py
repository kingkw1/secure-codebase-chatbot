import json
import requests
import re

def query_ollama(prompt, model_name="codellama"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "max_tokens": 100,
            "temperature": 0.2
        }
    )

    if response.status_code == 200:
        try:
            # Split the response content by newline characters
            response_lines = response.content.decode('utf-8').split('\n')
            responses = []

            for line in response_lines:
                if line.strip():  # Skip empty lines
                    json_obj = json.loads(line)
                    responses.append(json_obj.get("response", ""))

            # Join the responses to form the complete response
            complete_response = ' '.join(responses)
            
            # Clean up the response text
            clean_response = re.sub(r'\s+', ' ', complete_response).strip()
            
            return clean_response
        except json.JSONDecodeError as e:
            print("JSON Decode Error:", e)
            return None
    else:
        print("Error:", response.status_code, response.reason)
        return None