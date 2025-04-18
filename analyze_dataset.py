import json
import pandas as pd
from collections import Counter
import datetime
import re

# Function to load JSONL file
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Load the dataset
print("Loading dataset...")
data = load_jsonl('event_text_mapping.jsonl')
print(f"Loaded {len(data)} examples")

# Convert to pandas DataFrame for easier analysis
df = pd.DataFrame(data)

# Basic statistics
print("\n--- Basic Statistics ---")
print(f"Number of examples: {len(df)}")
print(f"Average event_text length: {df['event_text'].str.len().mean():.2f} characters")
print(f"Min event_text length: {df['event_text'].str.len().min()} characters")
print(f"Max event_text length: {df['event_text'].str.len().max()} characters")

# Analyze output fields
print("\n--- Output Fields Analysis ---")
field_presence = {}
field_values = {}

for field in ['action', 'date', 'time', 'attendees', 'location', 'duration', 'recurrence', 'notes']:
    # Count non-null values
    non_null_count = sum(1 for item in data if item['output'][field] is not None)
    field_presence[field] = non_null_count
    
    # Sample values (for non-list fields)
    if field != 'attendees':
        values = [item['output'][field] for item in data if item['output'][field] is not None]
        field_values[field] = Counter(values).most_common(10)
    else:
        # For attendees, count number of attendees per event
        attendee_counts = []
        for item in data:
            if item['output']['attendees'] is not None:
                attendee_counts.append(len(item['output']['attendees']))
        field_values[field] = Counter(attendee_counts).most_common(5)

# Print field presence
print("\nField presence (non-null counts):")
for field, count in field_presence.items():
    print(f"  {field}: {count}/{len(data)} ({count/len(data)*100:.1f}%)")

# Print common values for each field
print("\nCommon values for each field:")
for field, values in field_values.items():
    if field == 'attendees':
        print(f"  {field} (number of attendees per event): {values}")
    else:
        print(f"  {field} sample values: {values[:3] if values else 'None'}")

# Date format analysis
print("\n--- Date Format Analysis ---")
date_patterns = Counter()
for item in data:
    date = item['output']['date']
    if date:
        # Detect format using regex patterns
        if re.match(r'\d{2}/\d{2}/\d{4}', date):
            date_patterns['DD/MM/YYYY'] += 1
        elif re.match(r'\d{4}-\d{2}-\d{2}', date):
            date_patterns['YYYY-MM-DD'] += 1
        elif re.match(r'\d{1,2}/\d{1,2}/\d{4}', date):
            date_patterns['D/M/YYYY'] += 1
        else:
            date_patterns['other'] += 1

print("Date formats distribution:")
for pattern, count in date_patterns.most_common():
    print(f"  {pattern}: {count} ({count/sum(date_patterns.values())*100:.1f}%)")

# Time format analysis
print("\n--- Time Format Analysis ---")
time_patterns = Counter()
for item in data:
    time = item['output']['time']
    if time:
        # Detect format using regex patterns
        if re.match(r'\d{1,2}:\d{2} [AP]M', time):
            time_patterns['H:MM AM/PM'] += 1
        elif re.match(r'\d{1,2}:\d{2}', time):
            time_patterns['H:MM (24h)'] += 1
        else:
            time_patterns['other'] += 1

print("Time formats distribution:")
for pattern, count in time_patterns.most_common():
    print(f"  {pattern}: {count} ({count/sum(time_patterns.values())*100:.1f}%)")

# Duration format analysis
print("\n--- Duration Format Analysis ---")
duration_patterns = Counter()
for item in data:
    duration = item['output']['duration']
    if duration:
        # Detect format using regex patterns
        if re.match(r'\d+ hour', duration):
            duration_patterns['X hour(s)'] += 1
        elif re.match(r'\d+ min', duration):
            duration_patterns['X min(s)'] += 1
        elif re.match(r'\d+\s*hr', duration):
            duration_patterns['X hr(s)'] += 1
        else:
            duration_patterns['other'] += 1

print("Duration formats distribution:")
for pattern, count in duration_patterns.most_common():
    print(f"  {pattern}: {count} ({count/sum(duration_patterns.values())*100:.1f}%)")

# Analyze combinations of fields
print("\n--- Field Combinations Analysis ---")
field_combinations = Counter()
for item in data:
    present_fields = tuple(sorted([field for field in item['output'] 
                                   if item['output'][field] is not None]))
    field_combinations[present_fields] += 1

print("Top field combinations:")
for combo, count in field_combinations.most_common(5):
    print(f"  {combo}: {count} ({count/len(data)*100:.1f}%)")

# Print some examples with different patterns
print("\n--- Example Patterns ---")
for pattern_idx, (combo, _) in enumerate(field_combinations.most_common(3)):
    print(f"\nPattern {pattern_idx+1} ({combo}):")
    # Find an example with this combination
    for item in data:
        present_fields = tuple(sorted([field for field in item['output'] 
                                      if item['output'][field] is not None]))
        if present_fields == combo:
            print(f"  Input: {item['event_text']}")
            print(f"  Output: {json.dumps(item['output'], indent=2)}")
            break 