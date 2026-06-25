#!/usr/bin/env python3
"""
Step 3: Generate ACE-Step-compatible training metadata.
Takes yt-dlp .info.json files + processed audio and creates:
  - {song}.json (caption, bpm, keyscale, timesignature, language)
  - {song}.caption.txt (text caption for the track)
  - {song}.lyrics.txt (placeholder — will be auto-labeled by ACE-Step's LM)
  
This bridges the hermes-music-lora-pipeline to ACE-Step 1.5's built-in LoRA training.
"""
import os
import json
import glob

INPUT_DIR = "./dataset/raw"
PROCESSED_DIR = "./dataset/processed"
OUTPUT_DIR = "./dataset/acestep"  # ACE-Step training dataset

# Genre detection from title/artist for better captions
GENRE_MAP = {
    "ghostemane": "dark trap, horrorcore, underground hip-hop",
    "scarlxrd": "trap metal, aggressive rap, scream rap",
    "kendrick": "conscious hip-hop, west coast rap",
    "jid": "intelligent hip-hop, atlanta rap, melodic trap",
    "cal scruby": "underground hip-hop, lo-fi rap",
    "6lack": "alternative R&B, dark R&B, moody rap",
    "earthgang": "conscious hip-hop, atlanta rap, soulful",
    "j. cole": "conscious hip-hop, storytelling rap",
    "j cole": "conscious hip-hop, storytelling rap",
    "chief keef": "drill, chicago trap, aggressive rap",
    "a$ap": "cloud rap, hip-hop, experimental",
    "asap": "cloud rap, hip-hop, experimental",
    "schoolboy": "west coast hip-hop, gangsta rap",
    "scHoolboy": "west coast hip-hop, gangsta rap",
    "pouya": "underground rap, south florida trap",
    "token": "lyrical hip-hop, technical rap",
    "young m.a": "brooklyn drill, east coast rap",
    "joyner": "lyrical hip-hop, rapid-fire rap",
    "bone crusher": "crunk, southern hip-hop",
    "jasiah": "trap metal, aggressive rap",
    "akil": "underground hip-hop, classic rap",
}

def detect_genre(title: str, artist: str = "") -> str:
    """Detect genre from title/artist using our map."""
    combined = (title + " " + artist).lower()
    for key, genre in GENRE_MAP.items():
        if key.lower() in combined:
            return genre
    return "hip-hop, rap"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Copy processed WAVs to ACE-Step dataset dir
    processed_files = glob.glob(f"{PROCESSED_DIR}/*.wav")
    print(f"Found {len(processed_files)} processed audio files")
    
    for wav_file in processed_files:
        import shutil
        shutil.copy2(wav_file, OUTPUT_DIR)
    
    # Generate metadata from .info.json files
    info_files = glob.glob(f"{INPUT_DIR}/*.info.json")
    print(f"Found {len(info_files)} metadata files")
    
    matched = 0
    for info_file in info_files:
        with open(info_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        original_title = data.get('title', 'Unknown')
        artist = data.get('artist', '') or data.get('uploader', '')
        
        # Clean filename to match processed audio
        safe_name = "".join(c for c in original_title if c.isalnum() or c in (' ', '_', '.', '-')).rstrip()
        safe_name = safe_name.replace(' ', '_')
        
        # Check if matching WAV exists in output dir
        wav_path = os.path.join(OUTPUT_DIR, f"{safe_name}.wav")
        if not os.path.exists(wav_path):
            print(f"  ⚠ No matching WAV for: {safe_name}")
            continue
        
        matched += 1
        
        # Detect genre
        genre = detect_genre(original_title, artist)
        
        # Build caption
        duration = data.get('duration', 0)
        caption = f"A {genre} track"
        if artist:
            caption += f" by {artist}"
        caption += f". Titled '{original_title}'."
        
        # Write .json annotation (ACE-Step format)
        annotation = {
            "caption": caption,
            "language": "en",
        }
        json_path = os.path.join(OUTPUT_DIR, f"{safe_name}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(annotation, f, indent=2, ensure_ascii=False)
        
        # Write .caption.txt
        caption_path = os.path.join(OUTPUT_DIR, f"{safe_name}.caption.txt")
        with open(caption_path, 'w', encoding='utf-8') as f:
            f.write(caption)
        
        # Write placeholder .lyrics.txt (ACE-Step's auto-label will fill these)
        lyrics_path = os.path.join(OUTPUT_DIR, f"{safe_name}.lyrics.txt")
        if not os.path.exists(lyrics_path):
            with open(lyrics_path, 'w', encoding='utf-8') as f:
                f.write("[Instrumental]\n")  # Placeholder — ACE-Step LM will auto-label
        
        print(f"  ✅ {safe_name}: {genre}")
    
    print(f"\n{'='*50}")
    print(f"ACE-Step dataset ready at: {OUTPUT_DIR}")
    print(f"Matched: {matched}/{len(info_files)} tracks")
    print(f"\nNext steps:")
    print(f"  1. Launch ACE-Step Gradio: cd ~/projects/ACE-Step-1.5 && uv run acestep")
    print(f"  2. Go to LoRA Training tab")
    print(f"  3. Enter dataset path: {os.path.abspath(OUTPUT_DIR)}")
    print(f"  4. Click Scan → Auto Label → Preprocess → Train!")

if __name__ == "__main__":
    main()
