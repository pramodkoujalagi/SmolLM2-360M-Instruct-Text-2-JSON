# Entity Extraction from Calendar Event Texts
Fine-tuning a Small Language Model to parse raw, unstructured text and extract relevant entities related to scheduling a calendar event.

<div align="center">
<img src="https://github.com/user-attachments/assets/daa843b7-c347-4c5a-815b-c9b837191666" alt="Pramod" width="650"/>
</div>

**SmolLM2-360M-Instruct-Text-2-JSON** - A fine tuned version of SmolLM2-360M-Instruct-bnb-4bit specialized for parsing unstructured calendar event requests into structured JSON data.

### 👉 [Deployed Live App: Checkout the demo here!](https://huggingface.co/spaces/pramodkoujalagi/SmolLM2-360M-Instruct-Text-2-JSON)



## 📚 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
  - [Data Analysis](#data-analysis)
  - [Data Standardization](#data-standardization)
  - [Data Augmentation](#data-augmentation)
  - [Instruction Format Transformation](#instruction-format-transformation)
- [Fine-tuning Methodology](#fine-tuning-methodology)
  - [Model Selection](#model-selection)
  - [QLoRA Configuration](#qlora-configuration)
  - [Training Process](#training-process)
  - [Training Metrics](#training-metrics)
- [Performance Evaluation](#performance-evaluation)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Benchmark Results](#benchmark-results)
- [Technical Implementation](#technical-implementation)
  - [Code Structure](#code-structure)
  - [Deployment](##deployment)
- [Future Enhancements](#future-enhancements)
- [Conclusion](#conclusion)

## Project Overview

The aim of this project is to fine-tune a Small Language Model (SmolLM2-360M-Instruct-bnb-4bit) to parse unstructured calendar event requests and extract structured information. The model identifies key scheduling entities such as action, date, time, attendees, location, duration, recurrence, and notes from natural language text.

**Example:**
```
Input: "Plan an exhibition walkthrough on 15th, April 2028 at 3 PM with Harper, Grace, and Alex in the art gallery for 1 hour, bring your bag."
```
```
Output: {
  "action": "Plan an exhibition walkthrough",
  "date": "15/04/2028",
  "time": "3:00 PM",
  "attendees": [
    "Harper",
    "Grace",
    "Alex"
  ],
  "location": "art gallery",
  "duration": "1 hour",
  "recurrence": null,
  "notes": Bring your bag
}
```

## Dataset

### Data Analysis

Initial analysis on the dataset ([`event_text_mapping.jsonl`](event_text_mapping.jsonl)):
- 792 total examples
- Field presence distribution:
  - action, date, time: 100% (792/792)
  - attendees: 75.3% (596/792)
  - location: 65.5% (519/792)
  - duration: 81.2% (643/792)
  - recurrence: 3.3% (26/792)
  - notes: 1.8% (14/792)
- Format variations:
  - Date formats: DD/MM/YYYY (88.5%), YYYY-MM-DD (11.5%)
  - Time formats: 12-hour (93.2%), 24-hour (6.8%)
  - Various duration formats

### Data Standardization

Data standardization ([`standardize_data.py`](standardize_data.py)) was implemented to ensure consistency across:
- Date formats (standardized to DD/MM/YYYY)
- Time formats (standardized to 12-hour format with AM/PM)
- Duration expressions
- Attendees lists
- 

### Data Augmentation

The dataset was augmented to create a more balanced distribution:
- Increased total examples to 1,149
- Improved representation of less frequent fields:
  - recurrence: increased to 20.1% (231/1,149)
  - notes: increased to 19.5% (224/1,149)
- Maintained high coverage of core fields (action, date, time)

### Instruction Format Transformation

For improved fine-tuning performance, the data was restructured into an instruction-based ([`instruction_format.py`](instruction_format.py)) format:

From
```json
{"event_text": "Late night study session at the café on 15th, Dec 2024 at 9:00 pm for 2 hours.", "output": {"action": "study session", "date": "15/12/2024", "time": "9:00 PM", "attendees": null, "location": "café", "duration": "2 hours", "recurrence": null, "notes": null}}
```

To
```json
{
  "instruction": "Extract the relevant event information from this text and organize it into a JSON structure with fields for action, date, time, attendees, location, duration, recurrence, and notes. If a field is not present, return null for that field.",
  "input": "Late night study session at the café on 15th, Dec 2024 at 9:00 pm for 2 hours.",
  "output": "{\"action\": \"study session\", \"date\": \"15/12/2024\", \"time\": \"9:00 PM\", \"attendees\": null, \"location\": \"café\", \"duration\": \"2 hours\", \"recurrence\": null, \"notes\": null}"
}
```

**This approach provided several critical advantages:**

**Clear task definition**: Explicit instructions helped the model understand exactly what was expected

**Format specification**: The instruction clearly defined the required output structure

**Null-handling guidance**: Explicit instructions on how to handle missing fields

**Improved generalization**: The instruction-based format better leveraged the base model's instruction-following capabilities

The processed data was then prepared for Unsloth fine-tuning using [`prepare_unsloth_data.py`](prepare_unsloth_data.py), which:
- Formats data in the Unsloth-compatible chat template
- Creates a train-validation split (90-10)
- Results in 1,034 training examples and 115 validation examples

## Fine-tuning Methodology

### Model Selection

- Base model: SmolLM2-360M-Instruct-bnb-4bit
- Context length: 2048 tokens
- Quantization: 4-bit quantization for memory-efficient training

### QLoRA Configuration

The fine-tuning approach implemented in [`finetune_lora.py`](finetune_lora.py) uses Quantized Low-Rank Adaptation (QLoRA), combining 4-bit quantization with LoRA for parameter-efficient fine-tuning:

- LoRA rank: 64 (higher for better performance)
- Target modules: All key model components including attention modules, projections, and embedding layers
- LoRA alpha: 32
- Rank-stabilized LoRA (rsLoRA): Enabled for better stability
- Gradient checkpointing: Enabled with Unsloth optimizations for memory efficiency

### Training Process

The model was fine-tuned using the Unsloth framework with the following configuration:
- Batch size: 8 (2 per device × 4 gradient accumulation steps)
- Learning rate: 2e-4 with cosine scheduler
- Epochs: 3
- Weight decay: 0.01
- Optimizer: AdamW (8-bit)

### Training Metrics

- **Training time**: 881.28 seconds
- **Final training loss**: 0.3102
- **Final validation loss**: 0.2417
- **Validation perplexity**: 1.2735 (excellent perplexity, close to 1.0)

The validation framework included:
- Regular evaluation every 30 steps
- Early stopping with patience of 3 evaluations
- Automated learning curve generation
- Overfitting detection and prevention

The low validation perplexity (1.2735) indicates strong model performance, with the validation loss being lower than the training loss suggesting good generalization without overfitting. The difference between training and validation loss demonstrates that the model learned the task effectively while maintaining generalization capabilities.

## Performance Evaluation

### Evaluation Metrics

The evaluation process ([`eval.py`](eval.py)) focused on:

1. **Per-field accuracy**: Measures correctness of each extracted entity
   - String-based matching for simple fields
   - Jaccard similarity for list-type fields (e.g., attendees)
   
2. **Overall accuracy**: Average accuracy across all fields

3. **JSON parse rate**: Percentage of responses that parse as valid JSON

### Benchmark Results

#### Performance Comparison

<div align="center">

| Field | Base Model | Fine-tuned Model | Improvement |
|-------|------------|------------------|-------------|
| action | 0.000000 | 0.947826 | +94.78% |
| date | 0.000000 | 0.991304 | +99.13% |
| time | 0.000000 | 0.991304 | +99.13% |
| attendees | 0.228070 | 0.988304 | 	+333.35% |
| location | 0.342105 | 0.964912 | +181.97% |
| duration | 0.105263 | 1.000000 | +850.00% |
| recurrence | 0.815789 | 0.982456 | +20.43% |
| notes | 0.842105 | 1.000000 | +18.75% |
| **Overall** | **0.290710** | **0.983242** | **+238.22%** |
| JSON Parse Rate | 0.860870 | 1.000000 | +16.17% |

</div>


#### Key Observations
- The fine-tuned model achieved near-perfect performance on most fields
- Most dramatic improvements were in action, date, time, and duration fields
- Perfect JSON parse rate shows the model learned to maintain structured output format
- Perfect handling of duration field demonstrates successful standardization of varied time expressions

## Technical Implementation

### Code Structure

The project's implementation is organized into several modular components:

- **Data Processing**
  - [`standardize_data.py`](standardize_data.py): Normalizes date, time, duration formats
  - [`instruction_format.py`](instruction_format.py): Converts to instruction-based format

- **Model Training**
  - [`prepare_unsloth_data.py`](prepare_unsloth_data.py): Prepares data for Unsloth
  - [`finetune_lora.py`](finetune_lora.py): Implements LoRA fine-tuning

- **Evaluation**
  - [`eval.py`](eval.py): Evaluates model performance

### Deployment
Created a [**Gradio-based demo app**](https://huggingface.co/spaces/pramodkoujalagi/SmolLM2-360M-Instruct-Text-2-JSON) to interact with the model in real time. I have deployed using the Hugging Face **Spaces** platform.

📦 **App Stack**:
- `Gradio` for frontend
- `transformers` for loading the model
- `Hugging Face Spaces` for hosting

## Future Enhancements

While the current implementation achieves impressive results, several strategies could potentially improve performance further:

### Advanced Data Techniques
- **Synthetic Data Generation**: Generate additional examples using templates or larger language models to cover edge cases
- **Adversarial Examples**: Create challenging inputs to strengthen model robustness
- **Cross-lingual Augmentation**: Expand to multiple languages for broader applicability

### Model Optimization
- **Advanced QLoRA Tuning**: Experiment with different LoRA ranks and target modules to find optimal configurations (our current implementation already uses 4-bit quantization with LoRA)
- **Controlled Hyperparameters**: Experiment with lower learning rates (e.g., 5e-5) and more epochs (5-10) to potentially improve convergence
- **Layer-specific Fine-tuning**: Experiment with freezing certain layers and only fine-tuning others

### Training Methodology
- **Few-shot Learning**: Structure the fine-tuning to leverage the model's existing capabilities through in-context examples
- **Curriculum Learning**: Train on progressively more difficult examples
- **Ensemble Approach**: Train multiple models on different data splits and combine their predictions

### Evaluation Refinement
- **Human Evaluation**: Supplement automatic metrics with human judgments
- **Edge Case Testing**: Create a specialized test set of particularly challenging cases
- **Robustness Analysis**: Test model performance with noisy or malformed inputs


## Conclusion

The project successfully transforms the base SmolLM2-360M-Instruct-bnb-4bit model into a specialized entity extraction tool for calendar event scheduling. Through careful data curation, format standardization, and targeted fine-tuning, we can see the significant improvements across evaluation metrics.

The instruction-based fine-tuning approach proved particularly effective, allowing the model to generate consistently structured outputs while handling diverse input phrasing. The resulting model demonstrates impressive capabilities in extracting structured information from unstructured text, with near-perfect accuracy across multiple entity types.
