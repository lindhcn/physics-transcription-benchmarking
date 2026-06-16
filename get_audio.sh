#!/usr/bin/env bash
# ./get_audio.sh <dataset_dir>
#     where <dataset_dir> is the path to the directory containing dataset.json
# need curl (or wget), ffmpeg

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Argument missing: Dataset directory"
    exit 1
fi

DATASET_DIR="${1%/}"          # strip trailing slash if present
DATASET_JSON="$DATASET_DIR/dataset.json"

if [[ ! -f "$DATASET_JSON" ]]; then
    echo "ERROR: $DATASET_JSON not found."
    exit 1
fi

# convert url
#   https://pirsa.org/10050070  →  https://streamer.perimeterinstitute.ca/mp4-med/10050070.mp4
pirsa_to_mp4() {
    local pirsa_url="$1"
    local talk_id
    talk_id=$(basename "$pirsa_url")           # e.g. 10050070
    echo "https://streamer.perimeterinstitute.ca/mp4-med/${talk_id}.mp4"
}

# convert HH:MM:SS to total seconds
to_seconds() {
    local h m s rest
    h="${1%%:*}"; rest="${1#*:}"; m="${rest%%:*}"; s="${rest#*:}"
    # printf %d strips leading zeros
    h=$(printf '%d' "$h"); m=$(printf '%d' "$m"); s=$(printf '%d' "$s")
    echo $(( h * 3600 + m * 60 + s ))
}

# required fields per entry:
#   .audio_file                  – output filename
#   .audio_info.source.link      – pirsa.org URL
#   .audio_info.source.start     – HH:MM:SS clip start
#   .audio_info.source.end       – HH:MM:SS clip end

entry_count=$(jq 'length' "$DATASET_JSON")

for i in $(seq 0 $(( entry_count - 1 ))); do
    audio_file=$(jq -r ".[$i].audio_file"                  "$DATASET_JSON")
    pirsa_link=$(jq -r ".[$i].audio_info.source.link"      "$DATASET_JSON")
    start_time=$(jq -r ".[$i].audio_info.source.start"     "$DATASET_JSON")
    end_time=$(jq -r ".[$i].audio_info.source.end"         "$DATASET_JSON")
    title=$(jq -r ".[$i].audio_info.title"                 "$DATASET_JSON")

    mp4_url=$(pirsa_to_mp4 "$pirsa_link")
    output_path="$DATASET_DIR/test_data/$audio_file"

    echo "[$((i+1))/$entry_count] $title"

    start_sec=$(to_seconds "$start_time")
    end_sec=$(to_seconds "$end_time")
    duration=$(( end_sec - start_sec ))

    ffmpeg -y \
        -ss "$start_time" \
        -i "$mp4_url" \
        -t "$duration" \
        -vn \
        -acodec pcm_s16le \
        -ar 16000 \
        -ac 1 \
        "$output_path" \
        2>&1 | grep -E "(Error|error|Warning|Output|Duration|Stream)" || true

    if [[ -f "$output_path" ]]; then
        size=$(du -sh "$output_path" | cut -f1)
    else
        echo "FAILED – output file not created."
        exit 1
    fi
done
echo "All audio files written to $DATASET_DIR/test_data/"
