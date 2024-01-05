# Using yt-dlp to Download video from YouTube https://github.com/yt-dlp/yt-dlp
# Using moviepy library to convert video into audio https://github.com/Zulko/moviepy
# Using Whisper from OpenAI to transcribe any audio https://github.com/openai/whisper
# Using srt to composing srt file https://github.com/cdown/srt
import argparse
import os
import re
import sys
import time
from datetime import timedelta

import srt
import whisper
import yt_dlp
from loguru import logger
from moviepy.editor import VideoFileClip

current_dir = os.path.dirname(os.path.realpath(__file__))


class Log:
    def __init__(
            self, log_file_path=None, loglevel: str = "DEBUG"
    ) -> None:
        self.log_file_path = log_file_path
        self.level = loglevel
        self.configure_logger()

    def configure_logger(self) -> None:
        # Remove any existing handlers
        logger.remove()

        # Add a handler for stdout
        logger.add(
            sink=sys.stdout,
            format="<g>[{time:HH:mm}]</g> <level>{message}</level>",
            level=self.level,
        )

        # If log_file_path is not None, add a handler for the log file
        if self.log_file_path is not None:
            logger.add(
                sink=self.log_file_path,
                format="[{time:YYYY-MM-DD HH:mm:ss.SSS}] [{level}] [{file}:{line}] - {message}",
                level=self.level,
                rotation="24h",  # Rotate the log file every 24 hours
                enqueue=True,
            )


Log(os.path.join(current_dir, 'log.txt'))


# Create funtion to Download YouTube Video
def dld_ytb_video(url, video_folder):
    try:
        logger.info("Downloading youtube video...")
        ydl_opts = {
            'outtmpl': os.path.join(video_folder, '%(title)s.%(ext)s')
            # 'progress_hooks': [lambda d: print(f" Downloading {d['filename']}")],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info_dict)
            logger.info(f"Successfully downloaded youtube video into {filename}")
            return filename
    except Exception as e:
        logger.error(f"Error downloading video: {e}")
        return None


# Create Function To Convert Video Into Audio
def convert_to_audio(video_path, audio_folder):
    try:
        # Create the directory if it doesn't exist
        os.makedirs(audio_folder, exist_ok=True)
        logger.info("Extracting audio file...")
        clip = VideoFileClip(video_path)
        audio_path = os.path.join(audio_folder, re.sub(r'[^A-Za-z0-9]+', '_',
                                                       os.path.splitext(os.path.basename(video_path))[0]) + '.wav')
        clip.audio.write_audiofile(audio_path)

        if os.path.exists(audio_path):
            logger.info(f"Successfully extracted audio file to {audio_path}")
    except Exception as e:
        logger.exception(f"Error extracting audio: {e}")


# Create Function To Transcribe the audio file
# |  Size  | Parameters | English-only model | Multilingual model | Required VRAM | Relative speed |
# |:------:|:----------:|:------------------:|:------------------:|:-------------:|:--------------:|
# |  tiny  |    39 M    |     `tiny.en`      |       `tiny`       |     ~1 GB     |      ~32x      |
# |  base  |    74 M    |     `base.en`      |       `base`       |     ~1 GB     |      ~16x      |
# | small  |   244 M    |     `small.en`     |      `small`       |     ~2 GB     |      ~6x       |
# | medium |   769 M    |    `medium.en`     |      `medium`      |     ~5 GB     |      ~2x       |
# | large  |   1550 M   |        N/A         |      `large`       |    ~10 GB     |       1x       |
def generate_transcribe(input_audio, model_type):
    logger.info(f"Using OpenAi whisper with [{model_type}] model")
    logger.info(f"Transcribing [{input_audio}]")
    try:
        # Load the model based on the selected model type
        logger.info(f"- Loading {model_type} model...")
        model = whisper.load_model(model_type)

        # Transcribe the audio file
        logger.info(f"- Start transcribe the audio file please wait until finish the process...")
        result = model.transcribe(input_audio)

        return result
    except Exception as e:
        logger.exception(f"Error transcribing audio: {e}")


# Create Function to read segment information generated by whisper model and format it into srt compatible segments
# https://docs.fileformat.com/video/srt/#example-of-srt
# 1
# 00:05:00,400 --> 00:05:15,300
# This is an example of
# a subtitle.
def time_to_srt_format(segment_time):
    seconds = int(segment_time)
    microseconds = segment_time - seconds
    formatted_microseconds = "{:.3f}".format(microseconds)[2:]
    formatted_time = '0' + str(timedelta(seconds=seconds)) + ',' + formatted_microseconds
    return formatted_time


def create_srt_file(whisper_transcribe):
    # we need id, text, start and end from whisper_transcribe['segments']
    logger.info("Generating srt text format")
    segments = whisper_transcribe['segments']
    srt_list = []

    # loop over segments
    for segment in segments:
        # Get segment ID
        segment_id = segment['id']
        # Convert start time into srt format
        start_time = time_to_srt_format(segment['start'])
        # Convert end time into srt format
        end_time = time_to_srt_format(segment['end'])
        # Get Text from the segment
        text = segment['text']

        srt_subtitle_format = f"{segment_id}\n{start_time} --> {end_time}\n{text.strip()}\n"

        srt_list.append(srt_subtitle_format)
    return srt_list


def write_to_file(filename, text, mode="a", encoding="utf-8"):
    try:
        with open(filename, mode, encoding=encoding, buffering=1024 * 4) as file:  # Use 4KB buffering
            file.write(text)
    except IOError as e:
        logger.exception(f"Error writing to file: {e}")


# Create Function To change the font and color of the subtitles of the srt file generated (assume the srt file existing)
def adjust_srt_style(srt_file, font="Arial", color="red"):
    stylized_srt_file_file = f'{srt_file[:-4]}_with_style.srt'
    logger.info(f"Creating stylized srt file into {stylized_srt_file_file}")
    with open(srt_file, "r", encoding="utf-8") as f:
        subtitles = list(srt.parse(f.read()))
    # Loop through the subtitles and add the font and color tags to the content
    for sub in subtitles:
        sub.content = f"<font face='{font}' color='{color}'>{sub.content}</font>"
    # Open the output file and write the modified subtitles
    with open(stylized_srt_file_file, "w", encoding="utf-8") as f:
        f.write(srt.compose(subtitles))

    if os.path.exists(stylized_srt_file_file):
        logger.info(f"Successfully created {stylized_srt_file_file}")
    else:
        logger.error(f"Error creating the stylized srt file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Transcribe audio from a YouTube video or local video files, and generate styled SRT files.')
    parser.add_argument('-y', '--youtube_url',
                        help='URL of the YouTube video to download.')
    parser.add_argument('-l', '--local_video_dir',
                        help='Directory of local video files to transcribe. This option will be used if no YouTube URL is provided.')
    parser.add_argument('-w', '--whisper_model_type', default='tiny.en',
                        help='Type of Whisper ASR model to use for transcription. Options include "tiny.en", tiny, "base.en", "base", "small.en" "small", "medium.en", "medium", and "large".')
    parser.add_argument('-v', '--video_output_dir', default='Video',
                        help='Directory to save downloaded YouTube videos.')
    parser.add_argument('-a', '--audio_output_dir', default='Audio',
                        help='Directory to save extracted audio files.')
    parser.add_argument('-s', '--srt_output_dir', default='Srt',
                        help='Directory to save generated SRT files.')
    args = parser.parse_args()

    start_time = time.time()

    # Check if local video files are to be used
    if args.local_video_dir:
        video_folder_path = args.local_video_dir
        for video in os.listdir(video_folder_path):
            video_path = os.path.join(video_folder_path, video)
            convert_to_audio(video_path, args.audio_output_dir)
    else:
        video_path = dld_ytb_video(args.youtube_url, args.video_output_dir)
        convert_to_audio(video_path, args.audio_output_dir)

    # Create the SRT output directory if it doesn't exist
    os.makedirs(args.srt_output_dir, exist_ok=True)

    # Process each audio file in the audio output directory
    for audio_file in os.listdir(args.audio_output_dir):
        # Check if the file is an audio file
        if audio_file.endswith(('.wav', '.mp3')):
            # Get the full path of the audio file
            audio_file_path = os.path.join(args.audio_output_dir, audio_file)
            srt_filename = os.path.join(args.srt_output_dir, f'{os.path.splitext(audio_file)[0]}.srt')

            # Transcribe the audio file
            whisper_transcribe = generate_transcribe(audio_file_path, args.whisper_model_type)

            logger.info(f"Create srt file {srt_filename}")
            # Create the SRT format list
            srt_format_list = create_srt_file(whisper_transcribe)

            # Write each text in the SRT format list to the SRT file
            for text in srt_format_list:
                write_to_file(srt_filename, text)
            logger.info(f"Done creating srt file {srt_filename}")

            # Adjust the style of the SRT file
            adjust_srt_style(srt_filename)

    runtime = time.time() - start_time

    logger.info(f"Total Excution time: {runtime:3f} seconds")
