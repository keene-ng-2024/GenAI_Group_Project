import json

def inspect_review_object(filepath):
    """Extracts and prints the precise schema of a single review object."""
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                
                if 'review#1' in record:
                    review_data = record['review#1']
                    print("Data type of review#1:", type(review_data))
                    
                    if isinstance(review_data, dict):
                        print("\nKeys inside review#1:")
                        print(list(review_data.keys()))
                        print("\nFull content of review#1:")
                        print(json.dumps(review_data, indent=2))
                    else:
                        print("\nContent of review#1 (Not a dictionary):")
                        print(review_data)
                else:
                    print("The first record does not contain 'review#1'.")
                
                # Stop after the first valid record
                break

if __name__ == "__main__":
    inspect_review_object('ReviewCritique.jsonl')