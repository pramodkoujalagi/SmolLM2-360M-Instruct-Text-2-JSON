import json
import re
import os
from datetime import datetime
from dateutil import parser
import traceback

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

def standardize_time(time_str):
    """Standardize time format to 'HH:MM AM/PM'."""
    if time_str is None:
        return None

    try:
        # Handle special cases
        if isinstance(time_str, str):
            time_str = time_str.strip().upper()
            if time_str in ["NOON", "12 NOON", "12PM", "12 PM"]:
                return "12:00 PM"
            if time_str in ["MIDNIGHT", "12 MIDNIGHT", "12AM", "12 AM"]:
                return "12:00 AM"

        # Try to parse the time string
        try:
            time_obj = parser.parse(time_str).time()
            # Format as HH:MM AM/PM
            return time_obj.strftime("%I:%M %p").lstrip("0").replace(" 0", " ")
        except:
            # Manual parsing for more complex cases
            # 24-hour format (14:30)
            time_24h = re.match(r'^(\d{1,2}):(\d{2})$', str(time_str))
            # 12-hour with am/pm (2:30pm)
            time_12h_ampm = re.match(r'^(\d{1,2}):(\d{2})\s*(am|pm|AM|PM)$', str(time_str))
            # Hour only with am/pm (2pm)
            hour_ampm = re.match(r'^(\d{1,2})\s*(am|pm|AM|PM)$', str(time_str))

            if time_24h:
                hour, minute = int(time_24h.group(1)), int(time_24h.group(2))
                period = "AM" if hour < 12 else "PM"
                hour = hour % 12
                if hour == 0:
                    hour = 12
                return f"{hour}:{minute:02d} {period}"

            elif time_12h_ampm:
                hour, minute, period = int(time_12h_ampm.group(1)), int(time_12h_ampm.group(2)), time_12h_ampm.group(3).upper()
                if hour == 0:
                    hour = 12
                return f"{hour}:{minute:02d} {period}"

            elif hour_ampm:
                hour, period = int(hour_ampm.group(1)), hour_ampm.group(2).upper()
                if hour == 0:
                    hour = 12
                return f"{hour}:00 {period}"

            # If all parsing methods fail, return the original
            return time_str
    except Exception as e:
        print(f"Error standardizing time '{time_str}': {e}")
        return time_str

def standardize_date(date_str):
    """Standardize date format to 'DD/MM/YYYY'."""
    if date_str is None:
        return None

    try:
        # Try to parse with dateutil parser first (handles most cases)
        try:
            date_obj = parser.parse(date_str).date()
            return date_obj.strftime("%d/%m/%Y")
        except:
            # Fall back to manual parsing for special cases
            # Common date formats to try
            formats = [
                "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y",
                "%d %B %Y", "%B %d, %Y", "%d %b %Y", "%b %d, %Y"
            ]

            # Try to parse with different formats
            for fmt in formats:
                try:
                    date_obj = datetime.strptime(date_str, fmt)
                    # Return in the standardized format DD/MM/YYYY
                    return date_obj.strftime("%d/%m/%Y")
                except ValueError:
                    continue

            # Handle special case for "next Monday", "this Friday", etc.
            day_mapping = {
                "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
                "friday": 4, "saturday": 5, "sunday": 6
            }

            # Check for patterns like "next Monday"
            if isinstance(date_str, str):
                for day, day_num in day_mapping.items():
                    if day in date_str.lower():
                        # Return a placeholder that will be consistent
                        return f"Next {day.capitalize()}"

            # If parsing fails, return the original
            return date_str

    except Exception as e:
        print(f"Error standardizing date '{date_str}': {e}")
        return date_str

def standardize_duration(duration_str):
    """Standardize duration format."""
    if duration_str is None:
        return None

    try:
        if not isinstance(duration_str, str):
            return duration_str

        duration_str = duration_str.strip().lower()

        # Match patterns like "30 minutes", "1 hour", "1.5 hours", "1 hour and 30 minutes"
        # Convert to a standard format

        # Check for combined hours and minutes
        if "hour" in duration_str and ("minute" in duration_str or "min" in duration_str):
            hour_match = re.search(r'(\d+\.?\d*)\s*hour', duration_str)
            minute_match = re.search(r'(\d+)\s*(minute|min)', duration_str)

            hours = float(hour_match.group(1)) if hour_match else 0
            minutes = int(minute_match.group(1)) if minute_match else 0

            if hours == 1:
                hour_text = "1 hour"
            else:
                hour_text = f"{hours:.1f}".rstrip('0').rstrip('.') + " hours"

            if minutes == 1:
                minute_text = "1 minute"
            else:
                minute_text = f"{minutes} minutes"

            if hours > 0 and minutes > 0:
                return f"{hour_text} and {minute_text}"
            elif hours > 0:
                return hour_text
            else:
                return minute_text

        # Hours only
        elif "hour" in duration_str:
            hour_match = re.search(r'(\d+\.?\d*)\s*hour', duration_str)
            if hour_match:
                hours = float(hour_match.group(1))
                if hours == 1:
                    return "1 hour"
                else:
                    return f"{hours:.1f}".rstrip('0').rstrip('.') + " hours"

        # Minutes only
        elif "minute" in duration_str or "min" in duration_str:
            minute_match = re.search(r'(\d+)\s*(minute|min)', duration_str)
            if minute_match:
                minutes = int(minute_match.group(1))
                if minutes == 1:
                    return "1 minute"
                else:
                    return f"{minutes} minutes"

        # Try to parse duration expressions like "30m", "1h30m"
        hour_min_match = re.match(r'(\d+)h(\d+)m', duration_str)
        if hour_min_match:
            hours = int(hour_min_match.group(1))
            minutes = int(hour_min_match.group(2))

            hour_text = "1 hour" if hours == 1 else f"{hours} hours"
            minute_text = "1 minute" if minutes == 1 else f"{minutes} minutes"

            if hours > 0 and minutes > 0:
                return f"{hour_text} and {minute_text}"
            elif hours > 0:
                return hour_text
            else:
                return minute_text

        # Just hours (e.g., "2h")
        hour_match = re.match(r'(\d+)h$', duration_str)
        if hour_match:
            hours = int(hour_match.group(1))
            return "1 hour" if hours == 1 else f"{hours} hours"

        # Just minutes (e.g., "30m")
        min_match = re.match(r'(\d+)m$', duration_str)
        if min_match:
            minutes = int(min_match.group(1))
            return "1 minute" if minutes == 1 else f"{minutes} minutes"

        # Return as is if no pattern matched
        return duration_str
    except Exception as e:
        print(f"Error standardizing duration '{duration_str}': {e}")
        return duration_str

def standardize_attendees(attendees):
    """Standardize attendees list."""
    if attendees is None:
        return None

    try:
        # If it's a string, try to convert to list
        if isinstance(attendees, str):
            # Split by common separators and clean
            attendee_list = re.split(r'[,;&|]+', attendees)
            attendee_list = [name.strip() for name in attendee_list if name.strip()]
            return attendee_list

        # If it's already a list, clean each entry
        elif isinstance(attendees, list):
            return [name.strip() for name in attendees if name and name.strip()]

        return attendees
    except Exception as e:
        print(f"Error standardizing attendees '{attendees}': {e}")
        return attendees

def standardize_data(data):
    """Convert data from Format 1 to instruction-based Format 2."""
    standardized_data = []

    # Standard instruction for all examples
    instruction = "Extract the relevant event information from this text and organize it into a JSON structure with fields for action, date, time, attendees, location, duration, recurrence, and notes. If a field is not present, return null for that field."

    for item in data:
        event_text = item['event_text']
        output_obj = item.get('output', {})

        # Standardize the output fields
        standardized_output = {
            'action': output_obj.get('action'),
            'date': standardize_date(output_obj.get('date')),
            'time': standardize_time(output_obj.get('time')),
            'attendees': standardize_attendees(output_obj.get('attendees')),
            'location': output_obj.get('location'),
            'duration': standardize_duration(output_obj.get('duration')),
            'recurrence': output_obj.get('recurrence'),
            'notes': output_obj.get('notes')
        }

        # Convert the output object to a JSON string as required by Format 2
        output_json_str = json.dumps(standardized_output, ensure_ascii=False)

        # Create the new instruction-based format item
        standardized_item = {
            'instruction': instruction,
            'input': event_text,
            'output': output_json_str
        }

        standardized_data.append(standardized_item)

    return standardized_data

def main():
    try:
        # Create necessary directories
        os.makedirs('data', exist_ok=True)

        # Load original dataset
        original_data = load_jsonl('event_text_mapping.jsonl')
        print(f"Loaded {len(original_data)} items from event_text_mapping.jsonl")

        # Standardize the data
        standardized_data = standardize_data(original_data)
        print(f"Standardized {len(standardized_data)} items")

        # Save standardized dataset
        write_jsonl(standardized_data, 'data/standardized_data.jsonl')

        print("Data standardization complete!")
    except Exception as e:
        print(f"Error in main function: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()