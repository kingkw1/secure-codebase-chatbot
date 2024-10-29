from analyzer import generate_comment_with_model


code_example = """
def calculate_area(radius):
    return 3.14 * radius ** 2
"""

# Call the function to generate a comment for the example code
comment = generate_comment_with_model(code_example)
print(f"Generated Comment: {comment}")