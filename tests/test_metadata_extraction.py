import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzer import extract_repository_metadata

def test_metadata_extraction():
    base_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    directory = os.path.join(base_directory, 'sample_repo')

    print("Testing metadata extraction for directory:", directory)

    # Extract repository metadata
    metadata = extract_repository_metadata(directory)

    print("Extracted Metadata:")
    print(metadata)
    

if __name__ == "__main__":
    test_metadata_extraction()
