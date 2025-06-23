from datasets import load_dataset
import json
import os

def save_qa_pairs(output_file="data/qa_pairs.json"):
    os.makedirs("data", exist_ok=True)
    dataset = load_dataset("b-mc2/cli-commands-explained", split="train")
    qa_pairs = []
    for row in dataset:
        if row['code'] and row['description']:
            qa_pairs.append({
                "question": row['description'].strip(),
                "answer": f"{row['code'].strip()}"
            })
        if len(qa_pairs) >= 200:  # Grab 200 examples
            break
    with open(output_file, "w") as f:
        json.dump(qa_pairs, f, indent=2)

if __name__ == "__main__":
    save_qa_pairs()
