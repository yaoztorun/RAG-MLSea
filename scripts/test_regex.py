import re

def test_parsing_logic(text_response):
    # This is the exact regex we use in llama_generation.py
    match = re.search(r"(?:^|\s|:)([01])(?:\s|$|\.)", text_response)
    score = int(match.group(1)) if match else 0
    return score

# Test cases that often appear in LLM outputs
test_cases = [
    ("1. The answer is correct.", 1),
    ("0. The dates do not match.", 0),
    ("Result: 1", 1),
    ("The score is 0 because of missing authors.", 0),
    ("1", 1),
    ("Factual Correctness: 1. Reason: Matches.", 1),
    ("2019 is the year, so 0.", 0), # Tests that it ignores '2019'
]

print("--- REGEX PARSING VERIFICATION ---")
for text, expected in test_cases:
    actual = test_parsing_logic(text)
    status = "PASS" if actual == expected else "FAIL"
    print(f"[{status}] Input: '{text}' -> Parsed: {actual} (Expected: {expected})")
