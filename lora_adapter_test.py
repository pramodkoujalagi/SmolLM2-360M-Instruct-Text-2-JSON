import torch
import json
from unsloth import FastLanguageModel
from peft import PeftModel

def load_lora_model():
    """Load the base model and apply the LoRA adapter."""
    print("Loading model...")

    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/SmolLM2-360M-Instruct-bnb-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
    )

    # Load the fine-tuned weights
    model = PeftModel.from_pretrained(
        model,
        "/content/lora_model"
    )

    return model, tokenizer

def generate_json_response(model, tokenizer, prompt):
    """Generate a JSON response from the model given a prompt."""
    # Format the prompt to explicitly request JSON output
    formatted_prompt = f"""<|im_start|>user
Extract the relevant event information from this text and organize it into a JSON structure with fields for action, date, time, attendees, location, duration, recurrence, and notes. If a field is not present, return null for that field.

Text: {prompt}
<|im_end|>
<|im_start|>assistant
"""

    # Tokenize the prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    # Generate a response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,      # Deterministic for JSON generation
            temperature=0.1,      # Low temperature for more precise outputs
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode the response
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract just the assistant's response
    try:
        response = output_text.split("<|im_start|>assistant\n")[1].split("<|im_end|>")[0].strip()
    except IndexError:
        response = output_text  # Fallback if the splitting fails

    return response

def format_json_output(response):
    """Attempt to extract and format JSON from the response."""
    try:
        # Try direct JSON parsing first
        parsed_json = json.loads(response)
        return json.dumps(parsed_json, indent=2)
    except json.JSONDecodeError:
        # If that fails, try to find JSON-like structure in the text
        try:
            # Look for patterns like {" or { "
            start_idx = response.find('{')
            if start_idx >= 0:
                end_idx = response.rfind('}') + 1
                if end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                    parsed_json = json.loads(json_str)
                    return json.dumps(parsed_json, indent=2)
        except:
            pass

        # Return the original if we can't parse it
        return response

def interactive_cli():
    """Run an interactive command line interface to test the model."""
    # Load the model
    model, tokenizer = load_lora_model()
    print("Model loaded successfully! You can now interact with it.")
    print("Type 'quit', 'exit', or 'bye' to end the session.")
    print("\nEnter meeting text to extract JSON meeting information.")

    while True:
        # Get user input
        user_input = input("\n\nEnter meeting text: ")

        # Check if the user wants to exit
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Exiting the interactive session.")
            break

        # Generate and display response
        print("\nThinking...")
        response = generate_json_response(model, tokenizer, user_input)

        print("\n--- Model Response ---")
        print(response)

        # Try to format as JSON if possible
        formatted_json = format_json_output(response)
        if formatted_json != response:
            print("\n--- Formatted JSON ---")
            print(formatted_json)

if __name__ == "__main__":
    interactive_cli()
