import json
import re
from datetime import datetime

INPUT_FILE = "../data/MartyMessages.txt"
OUTPUT_FILE = "../data/marty_cleaned.json"

# Normalize smart punctuation into plain ASCII
def normalize_text(text):
    replacements = {
        '\u2018': "'",  # left single quote
        '\u2019': "'",  # right single quote
        '\u201c': '"',  # left double quote
        '\u201d': '"',  # right double quote
        '\u2014': '--', # em dash
        '\u2026': '...', # ellipsis
    }
    for uni_char, replacement in replacements.items():
        text = text.replace(uni_char, replacement)
    return text

# Detect timestamp lines
def is_timestamp_line(line):
    return bool(re.match(r'^[A-Z][a-z]{2} \d{2}, \d{4}', line.strip()))

# Detect sender line
def is_sender_line(line):
    return line.strip() in ["Me", "+18572149625"]

# Parse the file
def parse_messages(filepath):
    messages = []
    current_message = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if is_timestamp_line(line):
                # If previous message exists, store it
                if "sender" in current_message and "text" in current_message:
                    messages.append(current_message)
                    current_message = {}

                # Store timestamp
                timestamp_str = line.split("(")[0].strip()
                try:
                    timestamp = datetime.strptime(timestamp_str, "%b %d, %Y %I:%M:%S %p")
                    current_message["timestamp"] = timestamp.isoformat()
                except ValueError:
                    continue

            elif is_sender_line(line):
                current_message["sender"] = "Sophie" if line == "Me" else "Marty"

            else:
                cleaned_line = normalize_text(line)
                if current_message.get("text"):
                    current_message["text"] += " " + cleaned_line
                else:
                    current_message["text"] = cleaned_line

    # Add last message
    if "sender" in current_message and "text" in current_message:
        messages.append(current_message)

    return messages

# Write to JSON
def save_as_json(data, outpath):
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    messages = parse_messages(INPUT_FILE)
    save_as_json(messages, OUTPUT_FILE)
    print(f"Parsed {len(messages)} messages into '{OUTPUT_FILE}'")
