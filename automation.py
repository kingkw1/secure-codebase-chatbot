import os
import re
from collections import Counter
from analyzer import generate_readme_summary  # Import your function here

# Mock query_llm function for testing purposes
def query_llm(prompt, dependencies):
    # Simulate a response from an LLM for testing purposes
    return f"""
        # Project Title

        ## Purpose
        This project serves as a test to generate a README summary based on code structure and dependencies.

        ## Installation
        Run the following command to install the dependencies:
        ```bash
        pip install {" ".join(dependencies)}

        ## Usage
        Instructions on how to use the code go here.

        ## Features
            - Example feature 1
            - Example feature 2

        ## Dependencies
        {dependencies} 
"""

# Step 1: Identify Dependencies

def identify_dependencies(directory='.'): 
    """ Identifies external dependencies by scanning all .py files in the specified directory. Returns a list of dependency names (excluding common built-in libraries). """ 
    import_pattern = re.compile(r'^(?|from)\s+([a-zA-Z0-9_]+)') 
    dependencies = Counter()

    # Walk through all .py files in the directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                # Extract import statements
                with open(file_path, 'r') as f:
                    for line in f:
                        match = import_pattern.match(line)
                        if match:
                            dependencies[match.group(1)] += 1

    # Exclude common built-in modules
    built_in_modules = set([
        'os', 'sys', 're', 'time', 'math', 'json', 'random', 'datetime',
        'collections', 'itertools', 'subprocess', 'shutil', 'pathlib', 'logging'
    ])

    return [dep for dep in dependencies if dep not in built_in_modules]

# Step 2: Generate Code Structure Summary
def generate_code_structure(directory='.'): 
    """ Generates a simple outline of the code structure by listing key files and functions. """ 
    structure = [] 
    for root, _, files in os.walk(directory): 
        for file in files: 
            if file.endswith('.py'): 
                # Add main files with indentation based on directory level 
                relative_path = os.path.relpath(os.path.join(root, file), directory) 
                structure.append(f"- {relative_path}: Main module or functionality")
        
    return "\n".join(structure)

# Step 3: Generate README Summary and Save to File
def create_readme(directory='.'): # Identify dependencies and generate code structure summary 
    dependencies = identify_dependencies(directory) 
    code_structure = generate_code_structure(directory)

    # Generate README content using the summary function
    readme_content = generate_readme_summary(code_structure, dependencies)

    # Save to a README file
    # with open(os.path.join(directory, "README_suggestion.md"), "w") as f:
    #     f.write(readme_content)

    # print("README.md generated successfully!")
    print(readme_content)

# Run the complete workflow
if __name__ == "__main__":
    create_readme()
    