import torch
import json
import numpy as np
from datasets import load_dataset
from peft import PeftModel
from unsloth import FastLanguageModel
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load the models
def load_models():
    print("Loading models...")
    # Load base model
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/SmolLM2-360M-Instruct-bnb-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
    )

    # Load fine-tuned model
    ft_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/SmolLM2-360M-Instruct-bnb-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
    )

    # Load the fine-tuned weights
    ft_model = PeftModel.from_pretrained(
        ft_model,
        "/content/drive/MyDrive/Colab Notebooks/lora_model"
    )

    return base_model, ft_model, tokenizer

# Function to generate predictions
def generate_predictions(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.1,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    response = output_text.split("<|im_start|>assistant\n")[1].split("<|im_end|>")[0].strip()

    # Try to parse as JSON
    try: 
        parsed_json = json.loads(response)
        return response, parsed_json
    except:
        return response, None

# Function to extract field values
def extract_fields(parsed_json):
    if not parsed_json:
        return {
            'action': None,
            'date': None,
            'time': None,
            'attendees': None,
            'location': None,
            'duration': None,
            'recurrence': None,
            'notes': None
        }

    return {
        'action': parsed_json.get('action'),
        'date': parsed_json.get('date'),
        'time': parsed_json.get('time'),
        'attendees': parsed_json.get('attendees'),
        'location': parsed_json.get('location'),
        'duration': parsed_json.get('duration'),
        'recurrence': parsed_json.get('recurrence'),
        'notes': parsed_json.get('notes')
    }

# Function to evaluate field accuracy
def evaluate_field_accuracy(expected, predicted):
    if expected is None and predicted is None:
        return 1.0  # Both are null, correct
    if expected is None or predicted is None:
        return 0.0  # One is null, the other isn't

    # Handle lists (like attendees)
    if isinstance(expected, list) and isinstance(predicted, list):
        if len(expected) == 0 and len(predicted) == 0:
            return 1.0
        if len(expected) == 0 or len(predicted) == 0:
            return 0.0

        # Calculate overlap
        expected_set = set(expected)
        predicted_set = set(predicted)

        if len(expected_set) == 0:
            return 0.0

        # Jaccard similarity for lists
        intersection = len(expected_set.intersection(predicted_set))
        union = len(expected_set.union(predicted_set))
        return intersection / union

    # Simple string comparison for other fields
    return 1.0 if str(expected).lower() == str(predicted).lower() else 0.0

# Main evaluation function
def evaluate_models():
    # Load the models
    base_model, ft_model, tokenizer = load_models()

    # Load validation dataset
    val_dataset = load_dataset("json", data_files="/content/data/val.jsonl", split="train")

    print(f"Loaded {len(val_dataset)} validation samples")

    # Results storage
    results = {
        "base_model": {
            "field_accuracies": {
                "action": [], "date": [], "time": [], "attendees": [],
                "location": [], "duration": [], "recurrence": [], "notes": []
            },
            "json_parse_success": [],
            "raw_responses": []
        },
        "fine_tuned_model": {
            "field_accuracies": {
                "action": [], "date": [], "time": [], "attendees": [],
                "location": [], "duration": [], "recurrence": [], "notes": []
            },
            "json_parse_success": [],
            "raw_responses": []
        }
    }

    # Process each validation example
    for idx, example in enumerate(tqdm(val_dataset, desc="Evaluating")):
        # Extract instruction and input from the validation data
        text = example['text']

        # Extract the expected output JSON from the validation data
        try:
            # Extract expected output from the assistant part
            expected_output = text.split("<|im_start|>assistant\n")[1].split("<|im_end|>")[0].strip()
            expected_json = json.loads(expected_output)

            # Extract the input prompt from the user part
            input_prompt = text.split("<|im_start|>user\n")[1].split("<|im_end|>")[0].strip()

            # Create a new prompt without the assistant's response
            prompt = f"<|im_start|>user\n{input_prompt}<|im_end|>\n<|im_start|>assistant\n"

            # Get predictions from both models
            base_response, base_json = generate_predictions(base_model, tokenizer, prompt)
            ft_response, ft_json = generate_predictions(ft_model, tokenizer, prompt)

            # Store raw responses
            results["base_model"]["raw_responses"].append(base_response)
            results["fine_tuned_model"]["raw_responses"].append(ft_response)

            # Check JSON parsing success
            results["base_model"]["json_parse_success"].append(1 if base_json else 0)
            results["fine_tuned_model"]["json_parse_success"].append(1 if ft_json else 0)

            # Extract fields from expected and predicted JSONs
            expected_fields = extract_fields(expected_json)
            base_fields = extract_fields(base_json)
            ft_fields = extract_fields(ft_json)

            # Evaluate field accuracies
            for field in expected_fields:
                base_accuracy = evaluate_field_accuracy(expected_fields[field], base_fields[field])
                ft_accuracy = evaluate_field_accuracy(expected_fields[field], ft_fields[field])

                results["base_model"]["field_accuracies"][field].append(base_accuracy)
                results["fine_tuned_model"]["field_accuracies"][field].append(ft_accuracy)

        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            continue

        # Print progress every 10 examples
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(val_dataset)} examples")

    return results

# Function to calculate metrics
def calculate_metrics(results):
    metrics = {
        "base_model": {
            "json_parse_rate": np.mean(results["base_model"]["json_parse_success"]),
            "field_accuracies": {},
            "overall_accuracy": 0
        },
        "fine_tuned_model": {
            "json_parse_rate": np.mean(results["fine_tuned_model"]["json_parse_success"]),
            "field_accuracies": {},
            "overall_accuracy": 0
        }
    }

    # Calculate field accuracies
    all_base_accuracies = []
    all_ft_accuracies = []

    for field in results["base_model"]["field_accuracies"]:
        base_acc = np.mean(results["base_model"]["field_accuracies"][field])
        ft_acc = np.mean(results["fine_tuned_model"]["field_accuracies"][field])

        metrics["base_model"]["field_accuracies"][field] = base_acc
        metrics["fine_tuned_model"]["field_accuracies"][field] = ft_acc

        all_base_accuracies.extend(results["base_model"]["field_accuracies"][field])
        all_ft_accuracies.extend(results["fine_tuned_model"]["field_accuracies"][field])

    # Calculate overall accuracy
    metrics["base_model"]["overall_accuracy"] = np.mean(all_base_accuracies)
    metrics["fine_tuned_model"]["overall_accuracy"] = np.mean(all_ft_accuracies)

    return metrics

# Function to visualize results
def visualize_results(metrics):
    # Create a DataFrame for field accuracies
    field_names = list(metrics["base_model"]["field_accuracies"].keys())
    base_accs = [metrics["base_model"]["field_accuracies"][field] for field in field_names]
    ft_accs = [metrics["fine_tuned_model"]["field_accuracies"][field] for field in field_names]

    df = pd.DataFrame({
        'Field': field_names + ['Overall', 'JSON Parse Rate'],
        'Base Model': base_accs + [metrics["base_model"]["overall_accuracy"], metrics["base_model"]["json_parse_rate"]],
        'Fine-tuned Model': ft_accs + [metrics["fine_tuned_model"]["overall_accuracy"], metrics["fine_tuned_model"]["json_parse_rate"]]
    })

    # Create a summary table
    print("\n--- Performance Summary ---")
    print(df.set_index('Field'))

    # Calculate improvement percentages
    improvement = {
        field: (metrics["fine_tuned_model"]["field_accuracies"][field] - metrics["base_model"]["field_accuracies"][field]) /
               max(0.001, metrics["base_model"]["field_accuracies"][field]) * 100
        for field in field_names
    }
    improvement["Overall"] = (metrics["fine_tuned_model"]["overall_accuracy"] - metrics["base_model"]["overall_accuracy"]) / max(0.001, metrics["base_model"]["overall_accuracy"]) * 100
    improvement["JSON Parse Rate"] = (metrics["fine_tuned_model"]["json_parse_rate"] - metrics["base_model"]["json_parse_rate"]) / max(0.001, metrics["base_model"]["json_parse_rate"]) * 100

    print("\n--- Improvement Percentages ---")
    for field, imp in improvement.items():
        print(f"{field}: {imp:.2f}%")

    return df

# Main execution
if __name__ == "__main__":
    print("Starting model evaluation...")
    results = evaluate_models()
    metrics = calculate_metrics(results)
    df = visualize_results(metrics)