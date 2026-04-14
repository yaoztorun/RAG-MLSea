
import re

def mock_parse(text_response):
    # This is the exact regex from our code
    match = re.search(r"(?:^|\s|:)([01])(?:\s|$|\.)", text_response)
    score = int(match.group(1)) if match else 0
    return score

test_cases = [
    ("1. The answer is correct.", 1),
    ("0: The answer is wrong.", 0),
    ("Result: 1", 1),
    ("The year is 2019 and the score is 1.", 1),
    ("The year is 2019 but it's 0 because of names.", 0),
    ("I think it is correct but I won't give a number.", 0),
]

print("--- JUDGE PARSER TEST ---")
for text, expected in test_cases:
    actual = mock_parse(text)
    status = "PASS" if actual == expected else "FAIL"
    print(f"[{status}] Input: '{text}' -> Result: {actual}")

