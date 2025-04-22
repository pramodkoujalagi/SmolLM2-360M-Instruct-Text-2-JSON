#!/usr/bin/env python3
"""
Curate fine-tuning dataset for event extraction.
"""

import json, random, re, argparse, pathlib, os, sys
import pandas as pd
from dateutil import parser as du
from datetime import timedelta, datetime
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq

load_dotenv()
client = None
USE_OPENAI = False
USE_POST_PROCESS = False

# --- Standardization Helpers (inlined) ---
def standardize_time(time_str):
    """Standardize time format to 'HH:MM AM/PM'."""
    if time_str is None:
        return None
    try:
        if isinstance(time_str, str):
            t = time_str.strip().upper()
            if t in ["NOON", "12 NOON", "12PM", "12 PM"]:
                return "12:00 PM"
            if t in ["MIDNIGHT", "12 MIDNIGHT", "12AM", "12 AM"]:
                return "12:00 AM"
        try:
            time_obj = du.parse(time_str).time()
            return time_obj.strftime("%I:%M %p").lstrip("0").replace(" 0", " ")
        except:
            m24 = re.match(r'^(\d{1,2}):(\d{2})$', str(time_str))
            m12 = re.match(r'^(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)$', str(time_str))
            mhr = re.match(r'^(\d{1,2})\s*(AM|PM|am|pm)$', str(time_str))
            if m24:
                h,mi = int(m24.group(1)), int(m24.group(2))
                p = "AM" if h < 12 else "PM"
                h = h%12 or 12
                return f"{h}:{mi:02d} {p}"
            if m12:
                h,mi,p = int(m12.group(1)), int(m12.group(2)), m12.group(3).upper()
                h = h or 12
                return f"{h}:{mi:02d} {p}"
            if mhr:
                h,p = int(mhr.group(1)), mhr.group(2).upper()
                h = h or 12
                return f"{h}:00 {p}"
        return time_str
    except:
        return time_str

def standardize_date(date_str):
    """Standardize date format to 'DD/MM/YYYY'."""
    if date_str is None:
        return None
    try:
        try:
            d = du.parse(date_str, dayfirst=False).date()
            return d.strftime("%d/%m/%Y")
        except:
            fmts = ["%d/%m/%Y","%m/%d/%Y","%Y-%m-%d","%d-%m-%Y","%m-%d-%Y","%d %B %Y","%B %d, %Y","%d %b %Y","%b %d, %Y"]
            for fm in fmts:
                try:
                    d = datetime.strptime(date_str, fm)
                    return d.strftime("%d/%m/%Y")
                except:
                    continue
        return date_str
    except:
        return date_str

def standardize_duration(duration_str):
    """Standardize duration format."""
    if not duration_str:
        return None
    try:
        m = re.match(r"(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|m|min|mins|minute|minutes)", duration_str, re.I)
        if not m:
            return duration_str
        val,unit = m.groups(); unit=unit.lower()
        if unit.startswith('h'):
            hrs=float(val)
            if hrs.is_integer():
                h=int(hrs)
                return f"{h} hour{'s' if h!=1 else ''}"
            mins = int(round((hrs-int(hrs))*60))
            parts=[]
            if int(hrs): parts.append(f"{int(hrs)} hour{'s' if int(hrs)!=1 else ''}")
            if mins: parts.append(f"{mins} minute{'s' if mins!=1 else ''}")
            return ' and '.join(parts)
        else:
            mins=int(float(val))
            return f"{mins} minute{'s' if mins!=1 else ''}"
    except:
        return duration_str

def standardize_attendees(attendees):
    """Standardize attendees list."""
    if attendees is None:
        return None
    if isinstance(attendees,str):
        lst=re.split(r'[\,;&\|]+',attendees)
        return [n.strip() for n in lst if n.strip()]
    if isinstance(attendees,list):
        return [n.strip() for n in attendees if n and n.strip()]
    return attendees


def load_lookup(path):
    return [l.strip() for l in open(path, encoding='utf-8') if l.strip()]


def normalise_duration(s):
    if not s:
        return None
    m = re.match(r"(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|m|min|mins|minute|minutes)", s, re.I)
    if not m:
        return s
    val, unit = m.groups()
    val = str(int(float(val)))
    unit = 'hour' if unit.lower().startswith('h') else 'minutes'
    if val != '1' and unit == 'hour':
        unit += 's'
    return f"{val} {unit}"


def augment_row(row, names, locs, recurs):
    row_dict = row.to_dict()
    out = row_dict.get('output', {})
    text = row_dict.get('event_text', '')
    new_out = dict(out)
    # Recurrence injection
    if new_out.get('recurrence') is None and random.random() < 0.15:
        rec_val = random.choice(recurs)
        new_out['recurrence'] = rec_val
        text = f"{text} every {rec_val}"
    # Attendee name swap
    attendees = new_out.get('attendees')
    if attendees and random.random() < 0.1:
        new_names = random.sample(names, len(attendees))
        for old, new in zip(attendees, new_names):
            text = text.replace(old, new)
        new_out['attendees'] = new_names
    # Location swap
    location = new_out.get('location')
    if location and random.random() < 0.1:
        new_loc = random.choice([l for l in locs if l.lower() != location.lower()])
        text = text.replace(location, new_loc)
        new_out['location'] = new_loc
    return {'event_text': text, 'output': new_out}


def render_chat(row, skeleton_json):
    schema = json.dumps(skeleton_json, indent=2)
    prompt = "<|system|>\n\n"
    prompt += "You are an API that extracts structured information from text.\n\n"
    prompt += "Return ONLY valid JSON conforming exactly to the keys given below.\n\n"
    prompt += "Missing items must be null.\n\n"
    prompt += "Schema:\n\n"
    prompt += schema + "\n\n"
    prompt += "\n\n"
    prompt += f"TEXT: {row['event_text']}\n\n"
    prompt += "<|assistant|>\n"
    return prompt


# --- Paraphrase helper ---
def paraphrase_text(text):
    # toggle between Groq (default) and OpenAI
    if not USE_OPENAI:
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        groq_client = Groq(api_key=groq_key)
        messages = [
            {"role": "system", "content": "You are a helpful assistant that paraphrases event descriptions."},
            {"role": "user", "content": f"Paraphrase the following event description in a natural but equivalent style: \"{text}\""}
        ]
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=200
        )
        return resp.choices[0].message.content.strip()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    # initialize OpenAI client
    client = OpenAI(api_key=api_key)
    messages = [
        {"role": "system", "content": "You are a helpful assistant that paraphrases event descriptions."},
        {"role": "user", "content": f"Paraphrase the following event description in a natural but equivalent style: \"{text}\""}
    ]
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            temperature=0.7,
            max_tokens=200
        )
    except Exception:
        print("[Warning] gpt-4.1 failed, falling back to gpt-3.5-turbo...", file=sys.stderr)
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=200
        )
    return resp.choices[0].message.content.strip()


# Synthetic augmentation: generate new variants via LLM
def synthetic_augment_text(text, num_variants=1):
    """
    Use LLM to create new event descriptions by modifying names, locations, times, etc.
    Return a list of generated strings.
    """
    prompt = (
        f"Generate {num_variants} new and diverse event descriptions based on this one, "
        "changing names, locations, times, durations, recurrence, and notes. "
        "Return each description on its own line without numbering:\n" + text
    )
    raw = call_model_api(prompt)
    return [l.strip() for l in raw.splitlines() if l.strip()]


def call_model_api(prompt):
    # initialize Groq or OpenAI client for inference
    if USE_OPENAI:
        key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
    else:
        key = os.getenv("GROQ_API_KEY")
        client = Groq(api_key=key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
    return resp.choices[0].message.content


def sanitize_json_str(s):
    """Extract JSON object from a raw LLM string by finding first '{' and last '}'"""
    start = s.find('{')
    end = s.rfind('}')
    if start != -1 and end != -1 and end > start:
        return s[start:end+1]
    return s


def post_process_model_output(raw_json_str):
    # sanitize raw output (remove code fences or extra text)
    cleaned_str = sanitize_json_str(raw_json_str)
    raw = json.loads(cleaned_str)
    processed = {
        "action": raw.get("action", "").strip().title(),
        "date": standardize_date(raw.get("date", "")),
        "time": standardize_time(raw.get("time", "")),
        "attendees": standardize_attendees(raw.get("attendees") or []),
        "location": raw.get("location").title() if raw.get("location") else None,
        "duration": standardize_duration(raw.get("duration")) if raw.get("duration") else None,
        "recurrence": raw.get("recurrence"),
        "notes": raw.get("notes")
    }
    return processed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='event_text_mapping.jsonl')
    parser.add_argument('--paraphrase-demo', action='store_true', help='Test synthetic augmentation on sample examples and show structured output (default: 5 samples)')
    parser.add_argument('--use-openai', action='store_true', help='Use OpenAI for paraphrasing instead of Groq')
    parser.add_argument('--dry-run', action='store_true', help='Dry-run inference on sample prompts')
    parser.add_argument('--dry-run-n', type=int, default=5, help='Number of samples for dry-run')
    parser.add_argument('--post-process', action='store_true', help='Post-process model outputs after inference')
    args = parser.parse_args()
    global USE_OPENAI
    global USE_POST_PROCESS
    USE_OPENAI = args.use_openai
    USE_POST_PROCESS = args.post_process

    # Dry-run inference before processing
    if args.dry_run:
        data_dir = pathlib.Path(__file__).parent.parent / 'data'
        train_file = data_dir / 'curated_train.jsonl'
        if not train_file.is_file():
            print(f"Train file not found: {train_file}", file=sys.stderr)
            sys.exit(1)
        lines = open(train_file, 'r', encoding='utf-8').readlines()
        sample = random.sample(lines, min(args.dry_run_n, len(lines)))
        for line in sample:
            rec = json.loads(line)
            prompt_txt = rec['prompt']
            print("=== Prompt ===")
            print(prompt_txt)
            print("=== Inference ===")
            raw_out = call_model_api(prompt_txt)
            if USE_POST_PROCESS:
                try:
                    clean = post_process_model_output(raw_out)
                    print(json.dumps(clean), "\n")
                except json.JSONDecodeError:
                    print("Warning: failed to parse model output as JSON. Raw output:\n", raw_out, "\n")
            else:
                print(raw_out, "\n")
        sys.exit(0)

    # Load JSONL from file path
    src_path = pathlib.Path(args.src)
    if not src_path.is_file():
        parser.error(f"Source file not found: {args.src}")
    with open(src_path, 'r', encoding='utf-8') as f:
        df = pd.read_json(f, lines=True)

    if args.paraphrase_demo:
        skeleton = {
            "action": "",
            "date": "",
            "time": "",
            "attendees": [],
            "location": None,
            "duration": "",
            "recurrence": None,
            "notes": None
        }
        demo = df.sample(n=min(5, len(df)), random_state=42)
        print("Testing synthetic augmentation and structured output via LLM:")
        for _, row in demo.iterrows():
            orig = row['event_text']
            syn = synthetic_augment_text(orig, num_variants=1)[0]
            print(f"Original:   {orig}")
            print(f"Synthetic: {syn}")
            prompt = render_chat({"event_text": syn}, skeleton)
            raw = call_model_api(prompt)
            js = sanitize_json_str(raw)
            structured = post_process_model_output(js)
            print("Structured:", structured)
            print()
        sys.exit(0)

    # --- Augmentation
    base = pathlib.Path(__file__).parent
    names = load_lookup(base / 'resources' / 'first_names.txt')
    locs = load_lookup(base / 'resources' / 'locations.txt')
    rec = load_lookup(base / 'resources' / 'recurrence.txt')

    augmented = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        augmented.append(row_dict)
        aug = augment_row(row, names, locs, rec)
        augmented.append(aug)
    df_aug = pd.DataFrame(augmented).drop_duplicates(subset=['event_text'])

    # --- Standardize augmented outputs (last step)
    df_aug['output'] = df_aug['output'].apply(lambda o: {
        **o,
        'action': o.get('action').strip().title() if o.get('action') else None,
        'date': standardize_date(o.get('date')),
        'time': standardize_time(o.get('time')),
        'attendees': standardize_attendees(o.get('attendees')),
        'location': o.get('location').strip().title() if o.get('location') else None,
        'duration': standardize_duration(o.get('duration')),
        'recurrence': o.get('recurrence'),
        'notes': o.get('notes')
    } if isinstance(o, dict) else o)

    # --- Label quality guard: drop invalid labels
    def valid_label(o):
        if not isinstance(o, dict): return False
        try:
            du.parse(o.get('date'), dayfirst=True)
            du.parse(o.get('time'))
            return True
        except:
            return False
    before = len(df_aug)
    df_aug = df_aug[df_aug['output'].apply(valid_label)]
    print(f"Dropped {before - len(df_aug)} rows due to invalid labels")

    # --- Synthetic augmentation with structured re-parsing
    print("Generating synthetic structured variants with LLM...")
    synthetic_rows = []
    skeleton = {
        "action": "",
        "date": "",
        "time": "",
        "attendees": [],
        "location": None,
        "duration": "",
        "recurrence": None,
        "notes": None
    }
    for _, row in df_aug.iterrows():
        if random.random() < 0.3:
            variants = synthetic_augment_text(row['event_text'], num_variants=1)
            for v in variants:
                # parse each variant into structured output
                prompt_json = render_chat({"event_text": v}, skeleton)
                raw = call_model_api(prompt_json)
                json_str = sanitize_json_str(raw)
                new_out = post_process_model_output(json_str)
                synthetic_rows.append({"event_text": v, "output": new_out})
    print(f"Generated {len(synthetic_rows)} synthetic rows")
    if synthetic_rows:
        df_aug = pd.concat([df_aug, pd.DataFrame(synthetic_rows)], ignore_index=True).drop_duplicates(subset=['event_text'])

    # --- Upsampling for rare fields
    threshold = 300
    for field in ['recurrence', 'notes']:
        subset = df_aug[df_aug['output'].apply(lambda o: bool(o.get(field)))]
        count = len(subset)
        print(f"{field} count: {count}")
        if 0 < count < threshold:
            need = threshold - count
            dup = subset.sample(n=need, replace=True, random_state=42)
            df_aug = pd.concat([df_aug, dup], ignore_index=True)
            print(f"Upsampled {field} by {need} examples")

    # --- Deduplication
    df_aug = df_aug.drop_duplicates(subset=['event_text'])

    # --- Train/Val split (stratified on rare fields)
    df_aug['has_rare'] = df_aug['output'].apply(lambda o: bool(o.get('recurrence') or o.get('notes')))
    train = df_aug.sample(frac=0.9, random_state=42)
    val = df_aug.drop(train.index)
    print(f"Train examples: {len(train)}, Val examples: {len(val)}")

    # --- Render chat
    # Default schema: empty strings, empty list for attendees, null for optional
    skeleton = {
        "action": "",
        "date": "",
        "time": "",
        "attendees": [],
        "location": None,
        "duration": "",
        "recurrence": None,
        "notes": None
    }
    train['chat'] = train.apply(lambda r: render_chat(r, skeleton), axis=1)
    val['chat']   = val.apply(lambda r: render_chat(r, skeleton), axis=1)

    # --- Validation pass: ensure outputs are serializable to JSON
    for df_ in (train, val):
        for idx, o in df_['output'].items():
            try:
                json.dumps(o, ensure_ascii=False)
            except Exception as e:
                raise ValueError(f"Output JSON not serializable at index {idx}: {e}")

    # --- Write final JSONL for fine-tuning
    def write_jsonl(df, filepath):
        with open(filepath, 'w', encoding='utf-8') as fout:
            for _, row in df.iterrows():
                comp = json.dumps(row['output'], ensure_ascii=False)
                record = {"prompt": row['chat'], "completion": comp}
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')

    out_dir = pathlib.Path(__file__).parent.parent / 'data'
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train, out_dir / 'curated_train.jsonl')
    write_jsonl(val,   out_dir / 'curated_val.jsonl')

if __name__ == '__main__':
    main()
