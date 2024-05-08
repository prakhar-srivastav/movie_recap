import json
import re

def parse_srt(srt_content):
    """Parse the content of an SRT file and return a list of subtitles."""
    # Split the content into blocks
    blocks = re.split(r'\n\n+', srt_content.strip())
    subtitles = []
    for block in blocks:
        parts = block.split('\n', 2)
        if len(parts) >= 3:
            index = parts[0]
            time_range = parts[1]
            text = parts[2].replace('\n', ' ')
            subtitles.append({'index': index, 'time_range': time_range, 'text': text})
    return subtitles

def srt_to_json(srt_filename, json_filename):
    """Read an SRT file, parse it, and write the data to a JSON file."""
    with open(srt_filename, 'r', encoding='utf-8') as srt_file:
        content = srt_file.read()
    subtitles = parse_srt(content)
    with open(json_filename, 'w', encoding='utf-8') as json_file:
        json.dump(subtitles, json_file, indent=4)

# Example usage
srt_filename = 'extinction.srt'
json_filename = 'extinction.json'
srt_to_json(srt_filename, json_filename)