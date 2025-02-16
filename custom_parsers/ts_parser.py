import re

class Node:
    def __init__(self, type, name, start_line, end_line, code):
        self.type = type
        self.name = name
        self.start_line = start_line
        self.end_line = end_line
        self.code = code

def parse_ts_code_structure(file_path):
    """
    Custom parser for the code structure of a TypeScript file to extract functions and classes.
    """
    if not file_path.endswith('.ts'):
        raise ValueError("The file is not a TypeScript file")
        
    with open(file_path, "r") as file:
        file_content = file.read()

    lines = file_content.splitlines()
    code_structure = []

    class_pattern = re.compile(r'^\s*class\s+(\w+)')
    function_pattern = re.compile(r'^\s*function\s+(\w+)')
    method_pattern = re.compile(r'^\s*(\w+)\s*\(.*\)\s*{')

    current_class = None
    current_function = None

    for i, line in enumerate(lines):
        class_match = class_pattern.match(line)
        function_match = function_pattern.match(line)
        method_match = method_pattern.match(line)

        if class_match:
            if current_class:
                current_class.end_line = i
                code_structure.append(current_class)
            current_class = Node('ClassDeclaration', class_match.group(1), i + 1, len(lines), line)
        elif function_match:
            if current_function:
                current_function.end_line = i
                code_structure.append(current_function)
            current_function = Node('FunctionDeclaration', function_match.group(1), i + 1, len(lines), line)
        elif method_match and current_class:
            method_node = Node('MethodDeclaration', method_match.group(1), i + 1, len(lines), line)
            code_structure.append(method_node)

    if current_class:
        current_class.end_line = len(lines)
        code_structure.append(current_class)
    if current_function:
        current_function.end_line = len(lines)
        code_structure.append(current_function)

    for node in code_structure:
        node.code = "\n".join(lines[node.start_line - 1:node.end_line])

    return [node.__dict__ for node in code_structure]
