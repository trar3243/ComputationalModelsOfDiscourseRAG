import json
import sys 
sys.stdout.reconfigure(encoding='utf-8')
def compare_jsonl_answers(answer_key_path, test_path):
    # Dictionary to store the answer key data mapped by 'id'
    key_answers = {}
    
    # Read the answer key file
    with open(answer_key_path, 'r', encoding='utf-8') as f1:
        for line in f1:
            if line.strip():
                try:
                    data = json.loads(line)
                    if 'id' in data and 'answer' in data:
                        key_answers[data['id']] = data['answer']
                except json.JSONDecodeError:
                    print("Error decoding JSON in answer key.")

    # Variables for scoring
    total_compared = 0
    total_matches = 0

    # Read the test file and compare
    with open(test_path, 'r', encoding='utf-8') as f2:
        for line in f2:
            if line.strip():
                try:
                    test_data = json.loads(line)
                    doc_id = test_data.get('id')
                    test_answer = test_data.get('answer', '')

                    if doc_id in key_answers:
                        key_answer = key_answers[doc_id]
                        total_compared += 1
                        
                        # TYPE-SAFE COMPARISON
                        # If both are strings, strip whitespace and compare
                        if isinstance(key_answer, str) and isinstance(test_answer, str):
                            is_match = key_answer.strip() == test_answer.strip()
                        # If they are dicts, lists, or a mix of types, compare directly
                        else:
                            is_match = key_answer == test_answer
                        
                        if is_match:
                            total_matches += 1
                        
                        # Format output for readability in case they are dictionaries
                        key_display = json.dumps(key_answer, indent=2) if isinstance(key_answer, (dict, list)) else key_answer
                        test_display = json.dumps(test_answer, indent=2) if isinstance(test_answer, (dict, list)) else test_answer

                        # Output the comparison
                        print(f"{'='*60}")
                        print(f"ID: {doc_id}")
                        print(f"DIRECT MATCH: {is_match}")
                        print(f"{'-'*60}")
                        print("ANSWER KEY OUTPUT:")
                        print(key_display)
                        print(f"\n{'-'*60}")
                        print("TEST OUTPUT:")
                        print(test_display)
                        print(f"{'='*60}\n")
                        
                    else:
                        print(f"Warning: ID {doc_id} found in test file but not in answer key.\n")
                        
                except json.JSONDecodeError:
                    print("Error decoding JSON in test file.")

    # Output Final Score
    print("\n" + "#"*60)
    print("FINAL SCORE SUMMARY")
    print("#"*60)
    print(f"Total Items Compared: {total_compared}")
    print(f"Total Direct Matches: {total_matches}")
    
    if total_compared > 0:
        accuracy = (total_matches / total_compared) * 100
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("Accuracy: N/A (0 items compared)")
    print("#"*60 + "\n")

# Set your file paths here
answer_key = "loong.jsonl"
test = "method=most_recent_target_size_tokens=258_search_window=254_full_sent=False_weighted=False_evaluated_loong.jsonl"
test = "evaluated_loong.jsonl"
# Run the comparison
if __name__ == "__main__":
    compare_jsonl_answers(answer_key, test)
