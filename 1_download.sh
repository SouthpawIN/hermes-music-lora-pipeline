#!/bin/bash
# Step 1: Download audio and metadata from YouTube/Spotify playlists
# Requires: yt-dlp, ffmpeg

PLAYLIST_URL="${1:-https://youtube.com/playlist?list=YOUR_PLAYLIST_ID}"
OUTPUT_DIR="./dataset/raw"

echo "Downloading playlist: $PLAYLIST_URL"

# -x: extract audio
# --audio-format wav: save as uncompressed WAV for training
# -o: output template (playlist folder / title.wav)
# --write-info-json: save metadata (title, description, categories) for captioning
yt-dlp -x --audio-format wav \
  -o "$OUTPUT_DIR/%(playlist_title)s/%(title)s.%(ext)s" \
  --write-info-json \
  --embed-metadata \
  "$PLAYLIST_URL"

echo "Download complete. Metadata saved as .info.json files."