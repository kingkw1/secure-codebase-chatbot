import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automation import generate_readme_summary, generate_code_structure


def test_summary_simple():
    base_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    directory = os.path.join(base_directory, 'sample_repo')

    print("Testing README summary generation for directory:", directory)

    # Generate README content using the summary function
    code_structure = generate_code_structure(directory)

    # Generate README content using the summary function
    readme_content = generate_readme_summary(code_structure)

    print("Generated README Content:")
    print(readme_content)


if __name__ == "__main__":
    test_summary_simple()