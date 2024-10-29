import ast
import requests
import json 

def parse_code_structure(file_path):
    with open(file_path, "r") as file:
        file_content = file.read()

    # Parse the file content into an AST
    tree = ast.parse(file_content)
    
    # Store code structure details
    code_structure = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            code_structure.append({
                "type": "function",
                "name": node.name,
                "start_line": node.lineno,
                "end_line": node.body[-1].lineno,
                "code": ast.get_source_segment(file_content, node)
            })
        elif isinstance(node, ast.ClassDef):
            code_structure.append({
                "type": "class",
                "name": node.name,
                "start_line": node.lineno,
                "end_line": node.body[-1].lineno,
                "code": ast.get_source_segment(file_content, node)
            })

    return code_structure


def generate_comment_with_model(code_block, model_name="codellama"):
    # Define the prompt to instruct the model on generating a comment
    prompt = f"Please provide a concise, descriptive comment for the following code:\n\n{code_block}\n\n# Comment:"

    # Specify the endpoint, assuming it is accessible at localhost
    response = requests.post(
        "http://localhost:11434/api/generate",  # Use your endpoint
        json={
            "model": model_name,             # Use the provided model name
            "prompt": prompt,                # Prompt text
            "max_tokens": 100,               # Token limit for a concise comment
            "temperature": 0.2               # Low temperature for more factual responses
        }
    )

    # Print the response content for debugging
    print("Response Content:", response.content)

    if response.status_code == 200:
        try:
            # Split the response content by newline characters
            response_lines = response.content.decode('utf-8').split('\n')
            comments = []

            for line in response_lines:
                if line.strip():  # Skip empty lines
                    json_obj = json.loads(line)
                    comments.append(json_obj.get("response", ""))

            # Join the comments to form the complete comment
            return ' '.join(comments)
        except json.JSONDecodeError as e:
            print("JSON Decode Error:", e)
            return None
    else:
        print("Error:", response.status_code, response.reason)
        return None


def add_comment_to_code(file_path, code_structure):
    with open(file_path, "r") as file:
        lines = file.readlines()

    for code_block in code_structure:
        comment = generate_comment_with_model(code_block["code"])
        # Insert the comment in the code structure right before the start line
        lines.insert(code_block["start_line"] - 1, f"# {comment}\n")

    # Write the updated lines back to the file or save as a new file
    with open("updated_" + file_path, "w") as file:
        file.writelines(lines)
