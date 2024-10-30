import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzer import generate_readme_summary

def test_generate_readme_summary():
    test_cases = [
        {
            "description": "Valid input",
            "input": "This is a sample README file content.",
            "dependencies": ["OpenCV", "PIL"],
            "expected": "Summary: This is a sample README file content. with dependencies: [dependency1, dependency2]"
        },
        {
            "description": "Empty input",
            "input": "",
            "dependencies": ["dependency1", "dependency2"],
            "expected": "Summary: No content provided. with dependencies: [dependency1, dependency2]"
        },
        {
            "description": "Special characters",
            "input": "README with special characters! @#&*()",
            "dependencies": ["dependency1", "dependency2"],
            "expected": "Summary: README with special characters! @#&*() with dependencies: [dependency1, dependency2]"
        },
        {
            "description": "Long input",
            "input": "This is a very long README file content that exceeds normal length.",
            "dependencies": ["dependency1", "dependency2"],
            "expected": "Summary: This is a very long README file content that exceeds normal length. with dependencies: [dependency1, dependency2]"
        }
    ]

    for case in test_cases:
        result = generate_readme_summary(case["input"], case["dependencies"])
        print(f"Test: {case['description']}")
        print(f"Input: {case['input']}")
        print(f"Dependencies: {case['dependencies']}")
        print(f"Expected: {case['expected']}")
        print(f"Result: {result}")
        print("-" * 40)

if __name__ == '__main__':
    test_generate_readme_summary()