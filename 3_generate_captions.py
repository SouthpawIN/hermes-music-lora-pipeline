#!/usr/bin/env python3
"""
Step 3: Generate text captions for the audio dataset.
Uses YouTube transcripts (via youtube-transcript-api) + video metadata 
to create rich training prompts, replacing manual labeling.
"""
import os
import json
import glob
from youtube_transcript_api import YouTubeTranscriptApi

INPUT_DIR = "./dataset/raw"
OUTPUT_FILE = "./dataset/metadata.jsonl"

def extract_youtube_id(url: str) -> str:
    """Simple extraction of YouTube video ID from URL."""
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1]
    return ""

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    dataset_entries = []
    info_files = glob.glob(f"{INPUT_DIR}/*/*.info.json")
    
    print(f"Found {len(info_files)} metadata files. Extracting transcripts...")
    
    for info_file in info_files:
        with open(info_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        video_id = extract_youtube_id(data.get('webpage_url', ''))
        original_title = data.get('title', 'Unknown Title')
        description = data.get('description', '')
        
        # Clean filename to match processed audio
        safe_name = "".join(c for c in original_title if c.isalnum() or c in (' ', '_', '.', '-')).rstrip()
        safe_name = safe_name.replace(' ', '_')
        audio_path = f"dataset/processed/{safe_name}.wav"
        
        # Try to get transcript for rich captioning
        transcript_text = ""
        if video_id:
            try:
                # Fetch transcript (defaults to English, falls back gracefully)
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US'])
                transcript_text = " ".join([t['text'] for t in transcript_list])
                # Truncate to prevent overly long prompts
                transcript_text = transcript_text[:500] 
            except Exception:
                transcript_text = description[:200] # Fallback to description
        
        # Construct the training prompt
        # Format: [Genre/Style] + [Lyrical Content/Description]
        categories = data.get('categories', ['instrumental music'])
        genre = categories[0] if categories else "instrumental music"
        
        prompt = f"A {genre} track titled '{original_title}'. Audio content: {transcript_text}".strip()
        
        dataset_entries.append({
            "audio_path": audio_path,
            "prompt": prompt
        })
        
    # Write to JSONL
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in dataset_entries:
            f.write(json.dumps(entry) + '\n')
            
    print(f"Successfully generated {OUTPUT_FILE} with {len(dataset_entries)} entries.")

if __name__ == "__main__":
    main()