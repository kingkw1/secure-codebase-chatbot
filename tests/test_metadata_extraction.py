import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embeddings import extract_repository_metadata
from common import DEFAULT_CODEBASE_DIRECTORY

def test_metadata_extraction():
    directory = DEFAULT_CODEBASE_DIRECTORY

    print("Testing metadata extraction for directory:", directory)

    # Extract repository metadata
    metadata = extract_repository_metadata(directory)

    print("Extracted Metadata:")
    print(metadata)
    

if __name__ == "__main__":
    test_metadata_extraction()
