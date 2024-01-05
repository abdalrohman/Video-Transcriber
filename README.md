# YouTube-Video-Transcriber
An automated transcription tool that downloads YouTube videos or uses local videos, transcribes them using OpenAI’s Whisper ASR system, and generates styled SRT files. Ideal for creating subtitles or transcribing lectures and speeches.

## Features
- Generate SRT files.
- Adjust the style of the SRT files.

## Requirements
- Python 3.6 or above
- `yt-dlp`
- `moviepy`
- `whisper`
- `srt`

Before running the script, make sure to install `ffmpeg` by running the following commands in your terminal:
```bash
apt update
apt install ffmpeg
```

## Usage
1. Clone this repository.
2. Install the required dependencies.
3. Run `python youtube_transcriber.py -y [YOUTUBE_URL]` to transcribe a YouTube video.
4. Run `python youtube_transcriber.py -l [LOCAL_VIDEO_DIR]` to transcribe local video files.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
