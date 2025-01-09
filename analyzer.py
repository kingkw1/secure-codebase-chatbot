import ast
import os
import re
from collections import Counter
import sys
import javalang

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import query_ollama
from models import code_parser_model_name


def clean_text(response_text):
    # Step 1: Remove excessive whitespace
    response_text = re.sub(r'\s+', ' ', response_text).strip()

    # Step 2: Remove spaces around punctuation
    response_text = re.sub(r'\s([.,;:])', r'\1', response_text)
    response_text = re.sub(r'([.,;:])\s', r'\1 ', response_text)

    # Step 3: Fix broken words (e.g., "sub t ract" to "subtract")
    # This regex looks for sequences of single characters separated by spaces and joins them
    response_text = re.sub(r'\b(\w(?:\s\w)+)\b', lambda m: m.group(0).replace(' ', ''), response_text)

    # Step 4: Fix broken words within single quotations
    response_text = re.sub(r'`([^`]+)`', lambda m: '`' + m.group(1).replace(' ', '') + '`', response_text)

    # Step 5: Fix broken words with single characters followed by a space and another character
    response_text = re.sub(r'\b(\w)\s+(\w)\b', r'\1\2', response_text)

    # Step 6: Fix broken words with multiple characters followed by a space and another character
    response_text = re.sub(r'(\w)([A-Z])', r'\1 \2', response_text)

    # Step 7: Fix broken words with multiple characters followed by a space and another character
    response_text = re.sub(r'(\w)([A-Z])', r'\1 \2', response_text)

    return response_text


def parse_python_code_structure(file_path):
    """
    Parse the code structure of a Python file and extract functions and classes.
    """
    if not file_path.endswith('.py'):
        raise ValueError("The file is not a Python file")
        
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


def parse_java_code_structure(file_path):
    """
    Parse the code structure of a Java file and extract methods and classes.
    """
    if not file_path.endswith('.java'):
        raise ValueError("The file is not a Java file")
        
    with open(file_path, "r") as file:
        file_content = file.read()

    # Parse the file content using javalang
    try:
        tree = javalang.parse.parse(file_content)
    except javalang.parser.JavaSyntaxError as e:
        raise ValueError(f"Failed to parse Java file {file_path}: {e}")
    
    code_structure = []

    # Extract classes and methods
    for path, node in tree.filter(javalang.tree.ClassDeclaration):
        class_start = node.position.line
        class_end = max(member.position.line for member in node.body if member.position) if node.body else class_start

        code_structure.append({
            "type": "class",
            "name": node.name,
            "start_line": class_start,
            "end_line": class_end,
            "code": "\n".join(file_content.splitlines()[class_start - 1:class_end])
        })

    for path, node in tree.filter(javalang.tree.MethodDeclaration):
        method_start = node.position.line
        method_end = max(statement.position.line for statement in node.body if statement.position) if node.body else method_start
        annotations = [ann.name for ann in node.annotations] if node.annotations else []

        code_structure.append({
            "type": "method",
            "name": node.name,
            "start_line": method_start,
            "end_line": method_end,
            "parameters": [param.name for param in node.parameters],
            "annotations": annotations,
            "code": "\n".join(file_content.splitlines()[method_start - 1:method_end])
        })

    return code_structure


def parse_code_structure(file_path):
    """
    Parse the code structure of a file and extract functions and classes.
    """
    if file_path.endswith('.py'):
        return parse_python_code_structure(file_path)
    elif file_path.endswith('.java'):
        return parse_java_code_structure(file_path)


def generate_code_structure(directory='.'):
    """ Generates a detailed outline of the code structure by listing key files, functions, and classes. """
    structure = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                relative_path = os.path.relpath(os.path.join(root, file), directory)
                structure.append(f"- {relative_path}:")
                
                # Get detailed structure of each file (functions and classes)
                file_structure = parse_python_code_structure(os.path.join(root, file))
                
                # Add functions and classes to the structure
                for item in file_structure:
                    if item['type'] == 'function':
                        structure.append(f"    - Function `{item['name']}` (lines {item['start_line']}-{item['end_line']})")
                    elif item['type'] == 'class':
                        structure.append(f"    - Class `{item['name']}` (lines {item['start_line']}-{item['end_line']})")

    return "\n".join(structure)


def generate_comment_with_model(code_block):
    """
    Generate a comment for the given code block using a language model.
    """
    # Define the prompt to instruct the model on generating a comment
    prompt = f"Please provide a concise, descriptive comment for the following code:\n\n{code_block}\n\n# Comment:"

    response = query_ollama(prompt, code_parser_model_name)

    cleaned_response = clean_text(response)
    return cleaned_response


def generate_readme_summary(code_structure):
    """
    Generate a README summary based on the code structure of a repository.
    """
    # Create the prompt
    prompt = f"""
    Create a README for a code repository with the following structure:
    
    - **Main Files and Functions**:
    {code_structure}
    
    Please generate a README with sections for Project Title, Purpose, Usage, and Features.
    """

    # Send the prompt to your LLM model
    readme_text = query_ollama(prompt, code_parser_model_name)
    
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
    """
    Generate a README file for a code repository based on its structure.
    """
    # Generate code structure summary 
    code_structure = generate_code_structure(directory)
    
    # Generate README content using the summary function
    readme_content = generate_readme_summary(code_structure)

    return readme_content


def save_readme(readme_text, output_path="README.md"):
    """
    Save the generated README content to a file.
    """
    with open(output_path, "w") as file:
        file.write(readme_text)


def identify_dependencies(directory='.'):
    """
    Identify the dependencies of a Python code repository by analyzing import statements.
    """
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
            
    return required_dependencies
