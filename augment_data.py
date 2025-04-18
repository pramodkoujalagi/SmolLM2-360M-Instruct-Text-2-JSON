import json
import random
import copy
from datetime import datetime, timedelta
import re

def load_jsonl(file_path):
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def write_jsonl(data, file_path):
    """Write data to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Wrote {len(data)} items to {file_path}")

def parse_json_string(json_str):
    """Parse JSON string to object safely."""
    try:
        return json.loads(json_str)
    except:
        print(f"Error parsing JSON: {json_str}")
        return {}

def create_recurrence_examples(base_examples, num_to_create=50):
    """Create examples with various recurrence patterns."""
    new_examples = []
    recurrence_patterns = [
        "weekly", "monthly", "quarterly", "annually", "bi-weekly",
        "every Monday", "daily", "every two weeks", "every other Tuesday",
        "first Monday of the month", "last Friday of the month",
        "Monday through Friday", "weekends only"
    ]

    recurrence_phrases = [
        "Repeats {}", "Recurring {}", "{} repeating",
        "Happens {}", "Set up {} meeting", "Schedule {} occurrence",
        "{} event"
    ]

    for _ in range(num_to_create):
        # Pick a random base example
        base = random.choice(base_examples)
        new_example = copy.deepcopy(base)

        # Parse the output if it's a string
        if isinstance(new_example["output"], str):
            output_obj = parse_json_string(new_example["output"])
        else:
            output_obj = new_example["output"]

        # Add recurrence
        recurrence = random.choice(recurrence_patterns)
        output_obj["recurrence"] = recurrence

        # Modify the event text to include recurrence
        recurrence_phrase = random.choice(recurrence_phrases).format(recurrence)

        # Add recurrence to the input text
        input_text = new_example["input"]
        input_text = f"{input_text} {recurrence_phrase}"
        new_example["input"] = input_text

        # Update the output
        if isinstance(base["output"], str):
            new_example["output"] = json.dumps(output_obj)
        else:
            new_example["output"] = output_obj

        new_examples.append(new_example)

    return new_examples

def create_notes_examples(base_examples, num_to_create=50):
    """Create examples with notes field."""
    new_examples = []
    note_templates = [
        "Bring {}", "Remember to {}", "Don't forget {}",
        "Note: {}", "Important: {}", "{} required",
        "Will need {}", "Ask about {}"
    ]

    note_contents = [
        "laptop", "presentation materials", "project proposal",
        "quarterly report", "team metrics", "budget numbers",
        "client feedback", "research findings", "survey results",
        "agenda items", "meeting minutes", "action items from last time",
        "market analysis", "competitor updates", "financial forecast",
        "product roadmap", "customer feedback", "technical specifications"
    ]

    for _ in range(num_to_create):
        # Pick a random base example
        base = random.choice(base_examples)
        new_example = copy.deepcopy(base)

        # Parse the output if it's a string
        if isinstance(new_example["output"], str):
            output_obj = parse_json_string(new_example["output"])
        else:
            output_obj = new_example["output"]

        # Create and add a note
        note_template = random.choice(note_templates)
        note_content = random.choice(note_contents)
        note = note_template.format(note_content)
        output_obj["notes"] = note

        # Modify the event text to include the note
        input_text = new_example["input"]
        input_text = f"{input_text} {note}"
        new_example["input"] = input_text

        # Update the output
        if isinstance(base["output"], str):
            new_example["output"] = json.dumps(output_obj)
        else:
            new_example["output"] = output_obj

        new_examples.append(new_example)

    return new_examples

def create_24h_time_examples(base_examples, num_to_create=30):
    """Create examples with 24-hour time format."""
    new_examples = []

    for _ in range(num_to_create):
        # Pick a random base example
        base = random.choice(base_examples)
        new_example = copy.deepcopy(base)

        # Parse the output if it's a string
        if isinstance(new_example["output"], str):
            output_obj = parse_json_string(new_example["output"])
        else:
            output_obj = new_example["output"]

        # Get the time and convert to 24-hour format if it's in 12-hour format
        time_str = output_obj.get("time")
        if time_str and "AM" in time_str or "PM" in time_str:
            # Parse the time
            match = re.match(r"(\d{1,2}):(\d{2})\s(AM|PM)", time_str)
            if match:
                hour, minute, period = match.groups()
                hour = int(hour)
                if period == "PM" and hour < 12:
                    hour += 12
                elif period == "AM" and hour == 12:
                    hour = 0

                # Create 24h time format
                time_24h = f"{hour:02d}:{minute}"

                # Update input text - replace the original time with 24h format
                input_text = new_example["input"]

                old_time_patterns = [
                    fr"{hour}:{minute}\s*{period}",
                    fr"{hour}:{minute}\s*{period.lower()}",
                    fr"{hour}:{minute}"
                ]

                for pattern in old_time_patterns:
                    if re.search(pattern, input_text, re.IGNORECASE):
                        input_text = re.sub(pattern, time_24h, input_text, flags=re.IGNORECASE)
                        break

                new_example["input"] = input_text

        # Update the example
        if isinstance(base["output"], str):
            new_example["output"] = json.dumps(output_obj)
        else:
            new_example["output"] = output_obj

        new_examples.append(new_example)

    return new_examples

def create_edge_case_examples():
    """Create specific edge case examples."""
    instruction = "Extract the relevant event information from this text and organize it into a JSON structure with fields for action, date, time, attendees, location, duration, recurrence, and notes. If a field is not present, return null for that field."

    edge_cases = [
        # Multi-day event
        {
            "instruction": instruction,
            "input": "Annual conference from 15/06/2025 to 18/06/2025, all day at Convention Center with the entire team.",
            "output": json.dumps({
                "action": "Annual conference",
                "date": "15/06/2025 to 18/06/2025",
                "time": "All day",
                "attendees": ["entire team"],
                "location": "Convention Center",
                "duration": "4 days",
                "recurrence": "Annual",
                "notes": None
            })
        },
        # Event crossing midnight
        {
            "instruction": instruction,
            "input": "Late night coding session on 25/07/2025 starting at 10:00 PM until 2:00 AM the next day.",
            "output": json.dumps({
                "action": "Late night coding session",
                "date": "25/07/2025",
                "time": "10:00 PM to 2:00 AM",
                "attendees": None,
                "location": None,
                "duration": "4 hours",
                "recurrence": None,
                "notes": "Extends to next day"
            })
        },
        # Recurring event with end date
        {
            "instruction": instruction,
            "input": "Weekly team stand-up every Monday at 9:30 AM on Zoom, 30 minutes, starting Jan 1 until Mar 31.",
            "output": json.dumps({
                "action": "Weekly team stand-up",
                "date": "01/01/2025 to 31/03/2025",
                "time": "9:30 AM",
                "attendees": ["team"],
                "location": "Zoom",
                "duration": "30 minutes",
                "recurrence": "Weekly on Mondays",
                "notes": None
            })
        },
        # Hybrid meeting
        {
            "instruction": instruction,
            "input": "Quarterly review on 20/10/2025 at 1:30 PM, in Conference Room A and via Zoom, with executive team and department heads.",
            "output": json.dumps({
                "action": "Quarterly review",
                "date": "20/10/2025",
                "time": "1:30 PM",
                "attendees": ["executive team", "department heads"],
                "location": "Conference Room A and via Zoom",
                "duration": None,
                "recurrence": "Quarterly",
                "notes": "Hybrid meeting"
            })
        },
        # All-day event
        {
            "instruction": instruction,
            "input": "Company retreat on Saturday, August 5th, all day at Mountain Lodge.",
            "output": json.dumps({
                "action": "Company retreat",
                "date": "05/08/2025",
                "time": "All day",
                "attendees": None,
                "location": "Mountain Lodge",
                "duration": "Full day",
                "recurrence": None,
                "notes": None
            })
        },
        # Event with unusual time format
        {
            "instruction": instruction,
            "input": "Doctor appointment at quarter past 3 in the afternoon on 12th of May, Dr. Smith's office",
            "output": json.dumps({
                "action": "Doctor appointment",
                "date": "12/05/2025",
                "time": "3:15 PM",
                "attendees": ["Dr. Smith"],
                "location": "Dr. Smith's office",
                "duration": None,
                "recurrence": None,
                "notes": None
            })
        },
        # Event with relative date
        {
            "instruction": instruction,
            "input": "Team lunch next Friday at noon, Italiano Restaurant, 90 minutes",
            "output": json.dumps({
                "action": "Team lunch",
                "date": "Next Friday",
                "time": "12:00 PM",
                "attendees": ["team"],
                "location": "Italiano Restaurant",
                "duration": "90 minutes",
                "recurrence": None,
                "notes": None
            })
        }
    ]

    return edge_cases

def balance_field_combinations(base_examples, num_to_create=50):
    """Create examples with diverse field combinations."""
    new_examples = []

    # Define missing field combinations we want to target
    target_combinations = [
        ["action", "date", "time", "attendees", "location", "duration", "recurrence", "notes"],
        ["action", "date", "time", "attendees", "location", "recurrence", "notes"],
        ["action", "date", "time", "location", "recurrence", "notes"],
        ["action", "date", "time", "attendees", "duration", "recurrence", "notes"]
    ]

    for combo in target_combinations:
        for _ in range(num_to_create // len(target_combinations)):
            # Pick a random base example
            base = random.choice(base_examples)
            new_example = copy.deepcopy(base)

            # Parse the output if it's a string
            if isinstance(new_example["output"], str):
                output_obj = parse_json_string(new_example["output"])
            else:
                output_obj = new_example["output"]

            # Add missing fields based on the target combination
            modified_input = new_example["input"]

            # Add recurrence if needed
            if "recurrence" in combo and output_obj.get("recurrence") is None:
                recurrence = random.choice(["weekly", "monthly", "quarterly"])
                output_obj["recurrence"] = recurrence
                modified_input += f" Repeats {recurrence}."

            # Add notes if needed
            if "notes" in combo and output_obj.get("notes") is None:
                note = "Bring materials"
                output_obj["notes"] = note
                modified_input += f" Note: {note}."

            # Add attendees if needed
            if "attendees" in combo and (output_obj.get("attendees") is None or len(output_obj.get("attendees", [])) == 0):
                attendees = ["John", "Sarah"]
                output_obj["attendees"] = attendees
                modified_input += f" With {', '.join(attendees)}."

            # Add location if needed
            if "location" in combo and output_obj.get("location") is None:
                location = "Conference Room"
                output_obj["location"] = location
                modified_input += f" At {location}."

            # Add duration if needed
            if "duration" in combo and output_obj.get("duration") is None:
                duration = "1 hour"
                output_obj["duration"] = duration
                modified_input += f" For {duration}."

            # Update the example
            new_example["input"] = modified_input

            if isinstance(base["output"], str):
                new_example["output"] = json.dumps(output_obj)
            else:
                new_example["output"] = output_obj

            new_examples.append(new_example)

    return new_examples

def main():
    # Load standardized data (already in instruction format)
    data = load_jsonl("data/standardized_data.jsonl")
    print(f"Loaded {len(data)} examples from standardized data")

    # Create examples with recurrence
    recurrence_examples = create_recurrence_examples(data, num_to_create=100)
    print(f"Created {len(recurrence_examples)} examples with recurrence")

    # Create examples with notes
    notes_examples = create_notes_examples(data, num_to_create=100)
    print(f"Created {len(notes_examples)} examples with notes")

    # Create examples with 24h time format
    time_24h_examples = create_24h_time_examples(data, num_to_create=50)
    print(f"Created {len(time_24h_examples)} examples with 24-hour time format")

    # Create edge case examples
    edge_cases = create_edge_case_examples()
    print(f"Created {len(edge_cases)} edge case examples")

    # Create examples with balanced field combinations
    balanced_examples = balance_field_combinations(data, num_to_create=100)
    print(f"Created {len(balanced_examples)} examples with balanced field combinations")

    # Combine all examples
    augmented_data = data + recurrence_examples + notes_examples + time_24h_examples + edge_cases + balanced_examples
    print(f"Total augmented dataset size: {len(augmented_data)}")

    # Write augmented data to file
    write_jsonl(augmented_data, "data/curated_final_dataset.jsonl")

if __name__ == "__main__":
    main()