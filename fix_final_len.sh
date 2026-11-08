#!/bin/bash
# Re-mux a finished job's final mp4 so it is trimmed to min(video, audio) length (stream copy, no re-encode).
d=/home/administrator/imgsync/outputs/$1; f=$d/$1.mp4
v=$(ffprobe -v error -select_streams v:0 -show_entries stream=duration -of csv=p=0 "$f")
ffmpeg -nostdin -hide_banner -loglevel error -y -i "$f" -map 0 -c copy -t "$v" "$d/.fixed.mp4" && mv "$d/.fixed.mp4" "$f"
echo "$1: video=$v -> $(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f")s"
