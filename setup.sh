#!/bin/bash
set -e

apt-get update && apt-get install -y \
  ffmpeg \
  libsndfile1 \
  sox

pip install \
  "nemo_toolkit[asr]" \
  soundfile \
  numpy \
  openai-whisper
