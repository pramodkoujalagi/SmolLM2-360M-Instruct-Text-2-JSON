from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from unsloth import UnslothTrainer, UnslothTrainingArguments, is_bfloat16_supported
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from transformers import TrainerCallback

# Model parameters
max_seq_length = 2048  # Context length
dtype = None  # Auto-detect proper dtype
load_in_4bit = True  # Use 4-bit quantization - QLoRA

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/SmolLM2-360M-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,  # Enabling 4-bit quantization - QLoRA memory efficiency
)

# Setup LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=64,  # LoRA rank - higher is better but uses more VRAM
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj", "lm_head", "embed_tokens"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Optimize memory usage
    random_state=42,
    use_rslora=True,  # Rank stabilized LoRA
)

# Create output directories
output_dir = "outputs"
logs_dir = os.path.join(output_dir, "logs")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# Load training dataset
train_dataset = load_dataset(
    "json",
    data_files="data/train.jsonl",
    split="train",
)

# Load validation dataset
val_dataset = load_dataset(
    "json",
    data_files="data/val.jsonl",
    split="train",
)

print(f"Dataset loaded: {train_dataset.column_names}")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Sample train: {train_dataset[0]}")
print(f"Sample val: {val_dataset[0]}")

# Custom callback to track and save training/validation metrics for plotting
class MetricsTracker(TrainerCallback):
    """Custom callback for tracking losses and saving learning curves."""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.train_losses = []
        self.eval_losses = []
        self.best_eval_loss = float('inf')
        self.no_improvement_count = 0
        self.best_model_state = None
        self.eval_steps = 30
        self.patience = 3
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logs are about to be saved."""
        if logs is not None:
            # Track training loss
            if 'loss' in logs:
                self.train_losses.append((state.global_step, logs['loss']))
                print(f"Step {state.global_step} - Train Loss: {logs['loss']:.4f}")

            # Track validation loss if just did evaluation
            if 'eval_loss' in logs:
                self.eval_losses.append((state.global_step, logs['eval_loss']))
                eval_loss = logs['eval_loss']
                print(f"Step {state.global_step} - Eval Loss: {eval_loss:.4f}")
                
                # Check for improvement
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self.no_improvement_count = 0
                    print(f"New best eval loss: {eval_loss:.4f}")
                else:
                    self.no_improvement_count += 1
                    print(f"No improvement for {self.no_improvement_count} evaluations")
                    
                    # Early stopping
                    if self.no_improvement_count >= self.patience:
                        print("\nEarly stopping triggered! No improvement for several evaluations.")
                        control.should_training_stop = True
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of a training step."""
        # Check if it's time to evaluate (every eval_steps)
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            # Request evaluation on next opportunity
            control.should_evaluate = True
        return control
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        if len(self.train_losses) > 0:
            self._plot_learning_curves()
    
    def _plot_learning_curves(self):
        """Plot and save training and validation learning curves."""
        plt.figure(figsize=(12, 6))
        
        # Plot training loss
        if self.train_losses:
            steps, losses = zip(*self.train_losses)
            plt.plot(steps, losses, label='Training Loss', color='blue')
        
        # Plot validation loss on the same chart
        if self.eval_losses:
            steps, losses = zip(*self.eval_losses)
            plt.plot(steps, losses, label='Validation Loss', color='red')
            
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add annotations about potential overfitting
        if self.eval_losses and self.train_losses:
            final_train_loss = self.train_losses[-1][1]
            final_eval_loss = self.eval_losses[-1][1]
            
            if final_eval_loss > final_train_loss * 1.1:  # More than 10% difference
                plt.annotate(
                    f'Potential overfitting\nTrain: {final_train_loss:.4f}, Val: {final_eval_loss:.4f}',
                    xy=(0.7, 0.9), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5)
                )
            else:
                plt.annotate(
                    f'Good generalization\nTrain: {final_train_loss:.4f}, Val: {final_eval_loss:.4f}',
                    xy=(0.7, 0.9), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.5)
                )
        
        # Save the plot
        plt.tight_layout()
        learning_curve_path = os.path.join(self.output_dir, "learning_curves.png")
        plt.savefig(learning_curve_path)
        print(f"Learning curves saved to {learning_curve_path}")
        plt.close()

# Create metrics tracker callback
metrics_tracker = MetricsTracker(output_dir=logs_dir)

# Setup trainer with validation and evaluation settings - using only supported parameters
training_args = UnslothTrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_ratio=0.1,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=10,  # More frequent logging to get better curves
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=42,
    output_dir=output_dir,
    report_to="none",  # Disable default reporting, using custom callback
    
    # Only include parameters that are definitely supported
    save_steps=30,
    save_total_limit=3,  # Keep only the 3 best checkpoints
)

# Setup trainer with validation dataset
trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # Add validation dataset
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=4,
    args=training_args,
    callbacks=[metrics_tracker],  # Use our metrics tracker
)

print("Starting training with validation tracking...")
start_time = time.time()
trainer_stats = trainer.train()
training_time = time.time() - start_time

# Calculate and log perplexity on validation set
with torch.no_grad():
    eval_loss = trainer.evaluate()["eval_loss"]
    eval_perplexity = torch.exp(torch.tensor(eval_loss)).item()

# Print final metrics
print(f"\n{'='*50}")
print("Training completed!")
print(f"Training time: {training_time:.2f} seconds")
print(f"Final training loss: {trainer_stats.training_loss:.4f}")
print(f"Final validation loss: {eval_loss:.4f}")
print(f"Validation perplexity: {eval_perplexity:.4f}")

# Analysis of potential overfitting
if eval_loss > trainer_stats.training_loss * 1.1:
    print("\nWarning: Potential overfitting detected!")
    print(f"Validation loss is {(eval_loss/trainer_stats.training_loss - 1)*100:.2f}% higher than training loss")
    print("Consider: reducing model complexity, increasing dropout, or using more training data")
else:
    print("\nGood generalization: Model does not show signs of overfitting")

# Save the fine-tuned model
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# Save training metrics to a JSON file for future reference
metrics = {
    "training_time_seconds": training_time,
    "epochs": training_args.num_train_epochs,
    "final_training_loss": float(trainer_stats.training_loss),
    "final_validation_loss": float(eval_loss),
    "validation_perplexity": float(eval_perplexity),
    "training_samples": len(train_dataset),
    "validation_samples": len(val_dataset),
}

metrics_path = os.path.join(output_dir, "training_metrics.json")
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"Training metrics saved to {metrics_path}")

# Optional: Save in GGUF format for easier deployment
model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")

print("Fine-tuning complete!")

# Test examples and inference code
test_examples = [
    "Plan an exhibition walkthrough on 15th, April 2028 at 3 PM with Harper, Grace, and Alex in the art gallery for 1 hour.",
    "Book holiday planning 2023-11-27 1:15pm with Sarah for 40 minutes"
]

instruction = "Extract the relevant event information from this text and organize it into a JSON structure with fields for action, date, time, attendees, location, duration, recurrence, and notes. If a field is not present, return null for that field."

for example in test_examples:
    # Format exactly like the validation examples
    prompt = f"<|im_start|>user\n{instruction}\n\n{example}<|im_end|>\n<|im_start|>assistant\n"

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

    print(f"\nExample: {example}")
    print(f"Response: {response}")
