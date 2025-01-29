import ast
import os
import re
from collections import Counter
import sys
import javalang

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import query_ollama
from models import code_parser_model_name

# Registry for parsers
PARSERS = {}

def register_parser(file_extension, parser_function):
    """
    Register a parser for a specific file extension.
    """
    PARSERS[file_extension] = parser_function

def clean_text(response_text):
    """
    Clean text by removing excessive whitespace, fixing broken words,
    and ensuring proper punctuation and formatting.
    """
    # Step 1: Remove excessive whitespace
    response_text = re.sub(r'\s+', ' ', response_text).strip()

    # Step 2: Remove spaces around punctuation
    response_text = re.sub(r'\s([.,;:])', r'\1', response_text)
    response_text = re.sub(r'([.,;:])\s', r'\1 ', response_text)

    # Step 3: Fix broken words (e.g., "sub t ract" to "subtract")
    response_text = re.sub(r'\b(\w(?:\s\w)+)\b', lambda m: m.group(0).replace(' ', ''), response_text)

    # Step 4: Fix broken words within single quotations
    response_text = re.sub(r'`([^`]+)`', lambda m: '`' + m.group(1).replace(' ', '') + '`', response_text)

    # Step 5: Fix broken words with single characters followed by a space and another character
    response_text = re.sub(r'\b(\w)\s+(\w)\b', r'\1\2', response_text)

    # Step 6: Fix camel case breaking into separate words
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
    for _, node in tree.filter(javalang.tree.ClassDeclaration):
        class_start = node.position.line
        class_end = max(member.position.line for member in node.body if member.position) if node.body else class_start

        code_structure.append({
            "type": "class",
            "name": node.name,
            "start_line": class_start,
            "end_line": class_end,
            "code": "\n".join(file_content.splitlines()[class_start - 1:class_end])
        })

    for _, node in tree.filter(javalang.tree.MethodDeclaration):
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
    Parse the code structure of a file and extract functions, methods, and classes.
    This function dynamically selects the appropriate parser based on the file extension.
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    parser = PARSERS.get(file_extension)
    if not parser:
        raise ValueError(f"No parser registered for file extension: {file_extension}")
    return parser(file_path)

# Register parsers
register_parser('.py', parse_python_code_structure)
register_parser('.java', parse_java_code_structure)

def generate_code_structure(directory='.'):
    """
    Generates a detailed outline of the code structure by listing key files, functions, and classes.
    """
    structure = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_extension = os.path.splitext(file)[1].lower()
            if file_extension in PARSERS:
                relative_path = os.path.relpath(os.path.join(root, file), directory)
                structure.append(f"- {relative_path}:")
                try:
                    file_structure = parse_code_structure(os.path.join(root, file))
                    for item in file_structure:
                        structure.append(
                            f"    - {item['type'].capitalize()} `{item['name']}` (lines {item['start_line']}-{item['end_line']})"
                        )
                except Exception as e:
                    structure.append(f"    - Error parsing file: {e}")
    return "\n".join(structure)

def generate_comment_with_model(code_block):
    """
    Generate a concise, descriptive comment for the given code block using a language model.
    """
    prompt = f"Please provide a concise, descriptive comment for the following code:\n\n{code_block}\n\n# Comment:"
    response = query_ollama(prompt, code_parser_model_name)
    return clean_text(response)

def generate_readme_summary(code_structure):
    """
    Generate a README summary based on the code structure of a repository.
    """
    prompt = f"""
    Create a README for a code repository with the following structure:
    - **Main Files and Functions**:
    {code_structure}
    Please generate a README with sections for Project Title, Purpose, Usage, and Features.
    """
    response = query_ollama(prompt, code_parser_model_name)
    return clean_text(response) if response else "No response from LLM"

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
    import_pattern = re.compile(r'^(?:import|from)\s+([a-zA-Z0-9_]+)')
    dependencies = Counter()
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                with open(os.path.join(root, file), 'r') as f:
                    for line in f:
                        match = import_pattern.match(line)
                        if match:
                            dependencies[match.group(1)] += 1
    built_in_modules = {'os', 'sys', 're', 'time', 'math', 'json', 'random', 'datetime', 'collections'}
    return [dep for dep in dependencies if dep not in built_in_modules]
