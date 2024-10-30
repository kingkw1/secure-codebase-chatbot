import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzer import generate_comment_with_model

def test_comment():
    try:
        code_example = """
        def calculate_area(radius):
            return 3.14 * radius ** 2
        """

        # Call the function to generate a comment for the example code
        comment = generate_comment_with_model(code_example)
        print(f"Generated Comment: {comment}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_comment()