import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyzer import generate_readme_summary
from common import CODEBASE_DIRECTORY


def test_summary_simple():
    directory = CODEBASE_DIRECTORY

    print("Testing README summary generation for directory:", directory)

    # Generate README content using the summary function
    readme_content = generate_readme_summary(directory)

    print("Generated README Content:")
    print(readme_content)


if __name__ == "__main__":
    test_summary_simple()