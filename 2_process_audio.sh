#!/bin/bash
# Step 2: Process raw audio for model training
# Standardizes to 44.1kHz, Mono, and applies loudness normalization
# Outputs to ACE-Step-compatible flat structure

INPUT_DIR="./dataset/raw"
OUTPUT_DIR="./dataset/processed"

mkdir -p "$OUTPUT_DIR"

echo "Processing audio files..."
find "$INPUT_DIR" -name "*.wav" | while read -r file; do
  filename=$(basename "$file" .wav)
  # Replace spaces/special chars for safe filenames
  safe_name=$(echo "$filename" | tr ' ' '_' | tr -cd 'a-zA-Z0-9_.-')

  echo "Processing: $safe_name.wav"

  # -ar 44100: resample to 44.1kHz
  # -ac 1: force mono channel  
  # -af loudnorm: normalize to -14 LUFS (standard for music generation models)
  ffmpeg -y -i "$file" \
    -ar 44100 -ac 1 \
    -af "loudnorm=I=-14:TP=-1.5:LRA=11" \
    "$OUTPUT_DIR/${safe_name}.wav" 2>/dev/null
done

echo "Audio processing complete. Output in $OUTPUT_DIR"
echo ""
echo "Processed files:"
ls -1 "$OUTPUT_DIR"/*.wav 2>/dev/null | wc -l
