import json
import os
from sklearn.model_selection import train_test_split

def format_for_unsloth(input_file, output_train, output_val):
    """Format data for Unsloth fine-tuning."""
    data = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)

            # Get instruction, input, and output from instruction-based format
            instruction = item["instruction"]
            input_text = item["input"]
            output = item["output"]

            # Format for Unsloth
            formatted_text = f"<|im_start|>user\n{instruction}\n\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"

            data.append({"text": formatted_text})

    # Split into train and validation
    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

    os.makedirs(os.path.dirname(output_train), exist_ok=True)

    # Save train data
    with open(output_train, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Save validation data
    with open(output_val, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Created {len(train_data)} training examples and {len(val_data)} validation examples")

if __name__ == "__main__":
    input_file = "data/curated_final_dataset.jsonl"  # Updated to use the new instruction-based format
    output_train = "data/train.jsonl"
    output_val = "data/val.jsonl"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        exit(1)

    format_for_unsloth(input_file, output_train, output_val)
    print("Data preparation for Unsloth complete!")