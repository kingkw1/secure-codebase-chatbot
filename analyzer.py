import ast
import json 
import os
import re
from collections import Counter

# Add the parent directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import query_ollama

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

    return query_ollama(prompt, model_name)


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


def extract_repository_metadata(directory='.'):
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
    with open(output_path, "w") as file:
        json.dump(metadata, file, indent=2)

        
def identify_dependencies(directory='.'):
    # Regular expression to find import statements
    import_pattern = re.compile(r'^(?:import|from)\s+([a-zA-Z0-9_]+)')
    dependencies = Counter()

    # Walk through all .py files in the specified directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                # Read each file to find imports
                with open(file_path, 'r') as f:
                    for line in f:
                        match = import_pattern.match(line)
                        if match:
                            dependencies[match.group(1)] += 1

    # Filter out common built-in modules
    built_in_modules = set([
        'os', 'sys', 're', 'time', 'math', 'json', 'random', 'datetime',
        'collections', 'itertools', 'subprocess', 'shutil', 'pathlib', 'logging'
    ])
    
    required_dependencies = [dep for dep in dependencies if dep not in built_in_modules]
    
    print("Dependencies found (excluding built-in modules):")
    for dep in required_dependencies:
        print(dep)
        
    return required_dependencies


def generate_code_structure(directory='.'):
    """ Generates a detailed outline of the code structure by listing key files, functions, and classes. """
    structure = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                relative_path = os.path.relpath(os.path.join(root, file), directory)
                structure.append(f"- {relative_path}:")
                
                # Get detailed structure of each file (functions and classes)
                file_structure = parse_code_structure(os.path.join(root, file))
                
                # Add functions and classes to the structure
                for item in file_structure:
                    if item['type'] == 'function':
                        structure.append(f"    - Function `{item['name']}` (lines {item['start_line']}-{item['end_line']})")
                    elif item['type'] == 'class':
                        structure.append(f"    - Class `{item['name']}` (lines {item['start_line']}-{item['end_line']})")

    return "\n".join(structure)


def generate_readme_summary(code_structure):
    # Create the prompt
    prompt = f"""
    Create a README for a code repository with the following structure:
    
    - **Main Files and Functions**:
    {code_structure}
    
    Please generate a README with sections for Project Title, Purpose, Usage, and Features.
    """

    # Send the prompt to your LLM model
    readme_text = query_ollama(prompt)
    
    if readme_text:
        # Step 1: Remove excessive whitespace and spaces around punctuation
        readme_text = re.sub(r'\s+', ' ', readme_text).strip()
        readme_text = re.sub(r'\s([.,;:])', r'\1', readme_text)

        # Step 2: Fix broken words (e.g., "data _ processor" to "data_processor")
        readme_text = re.sub(r'\b(\w+)\s+([._]\s*\w+)', r'\1\2', readme_text)

        # Step 3: Add consistent line breaks for Markdown headers
        sections = ["Project Title", "Purpose", "Usage", "Features"]
        for section in sections:
            readme_text = re.sub(f"## {section}", f"\n## {section}\n", readme_text)

        # Step 4: Ensure line breaks for Markdown list items
        readme_text = re.sub(r'(\* )', r'\n\1', readme_text)
        readme_text = re.sub(r'(\*\*)', r'\n\1', readme_text)  # Handles extra list markers

        # Step 5: Ensure proper formatting for Markdown code blocks
        readme_text = re.sub(r'(```)', r'\n\1', readme_text)
        
        # Step 6: Final cleanup for extra spaces or line breaks
        readme_text = re.sub(r'\n\s+', '\n', readme_text)
        readme_text = re.sub(r'\n{2,}', '\n\n', readme_text)  # Ensure no double newlines

        return readme_text
    else:
        return "No response from LLM"


def create_readme_for_directory(directory='.'): 
    # Generate code structure summary 
    code_structure = generate_code_structure(directory)
    
    # Generate README content using the summary function
    readme_content = generate_readme_summary(code_structure)

    return readme_content


def save_readme(readme_text, output_path="README.md"):
    with open(output_path, "w") as file:
        file.write(readme_text)