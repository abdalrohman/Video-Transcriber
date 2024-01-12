# Video-Transcriber
[[Whisper Repo]](https://github.com/openai/whisper.git)
[[Colab]](https://colab.research.google.com/github/abdalrohman/Video-Transcriber/blob/main/notebooks/Evaluate_OpenAI_Whisper_Model.ipynb)

Effortlessly transcribe videos and create styled subtitles with this powerful wrapper for Whisper's ASR capabilities.

## Features
- Transcribe videos from various sources:
    - YouTube URLs
    - Local video files
    - Directories containing multiple videos
- Produce multiple subtitle formats:
    - SRT
    - VTT
    - TSV
    - JSON
    - Plain text
- Customize subtitle appearance with font and color options
- Leverage Whisper's advanced transcription features:
    - Language detection
    - Speech-to-text translation/transcription
    - Word level timestamps (experimental)

## Requirements
- Python 3.6 or above
- `yt-dlp`
- `moviepy`
- `whisper`
- `srt`
- `numpy`
- `torch`
- `ffmpeg` (install with `apt update && apt install ffmpeg`)

## Installation
1. Clone this repository:

    `git clone https://github.com/abdalrohman/Video-Transcriber.git`

2. Install the required dependencies:

    `pip install -r requirements.txt`

## Usage
`python video_transcriber.py [OPTIONS]`

Options:
- `--video_file [VIDEO_FILE]`: video file input can be either a video file or directory contains videos or youtube url
- `-v [VIDEO_OUTPUT_DIR]`: Directory to save downloaded YouTube videos. (default: Video).
- `-a [AUDIO_OUTPUT_DIR]`: Directory to save extracted audio files. (default: Audio).
- `-s [SRT_OUTPUT_DIR]`: Directory to save generated SRT files. (default: Srt).
- `--font_type [FONT_TYPE]`: Font for subtitles (default: "Arial").
- `--font_color [FONT_COLOR]`: Color for subtitles (default: "red").
- ... (See `python video_transcriber.py -h` for more options)

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
