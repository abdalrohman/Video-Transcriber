# Using yt-dlp to Download video from YouTube https://github.com/yt-dlp/yt-dlp
# Using moviepy library to convert video into audio https://github.com/Zulko/moviepy
# Using Whisper from OpenAI to transcribe any audio https://github.com/openai/whisper
# Using srt to composing srt file https://github.com/cdown/srt
import argparse
import os
import re
import sys
import time
import traceback
import warnings

import numpy as np
import srt
import torch
import yt_dlp
from loguru import logger
from moviepy.editor import VideoFileClip
from whisper import available_models, load_model
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
from whisper.transcribe import transcribe
from whisper.utils import (
    get_writer,
    str2bool,
    optional_float,
    optional_int,
)

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


# Define a function to download YouTube videos
def dld_ytb_video(url: str, video_folder: str) -> str:
    """Download a YouTube video and return its file name.

    Args:
        url (str): The URL of the YouTube video.
        video_folder (str): The folder to save the video.

    Returns:
        str: The file name of the downloaded video, or None if an error occurred.
    """
    try:
        # Set the output template for the video file name
        ydl_opts = {
            "outtmpl": os.path.join(video_folder, "%(title)s.%(ext)s")
            # 'progress_hooks': [lambda d: print(f" Downloading {d['filename']}")],
        }
        # Create a YouTube downloader object
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract the information of the video
            info_dict = ydl.extract_info(url, download=True)
            # Prepare the file name of the video
            file_name = ydl.prepare_filename(info_dict)
            return file_name
    except Exception as e:
        logger.error(f"Error downloading video: {e}")
        return None


# Define a function to convert video into audio
def convert_to_audio(video_path: str, audio_folder: str) -> None:
    """Convert a video file into an audio file and save it in a folder.

    Args:
        video_path (str): The path to the video file.
        audio_folder (str): The folder to save the audio file.
    """
    try:
        # Create the directory if it doesn't exist
        os.makedirs(audio_folder, exist_ok=True)
        clip = VideoFileClip(video_path)
        # audio_path = os.path.join(audio_folder, re.sub(r'[^A-Za-z0-9]+', '_',
        #                                                os.path.splitext(os.path.basename(video_path))[0]) + '.wav')
        audio_path = os.path.join(audio_folder,
                                  f'{os.path.splitext(os.path.basename(video_path))[0]}.wav')

        clip.audio.write_audiofile(audio_path)

    except Exception as e:
        logger.exception(f"Error extracting audio: {e}")


# Define a function to change the font and color of the subtitles of an srt file
def adjust_srt_style(srt_file: str, font: str = "Arial", color: str = "red") -> None:
    """Change the font and color of the subtitles of an srt file and save it as a new file.

    Args:
        srt_file (str): The path to the original srt file.
        font (str, optional): The font name to use for the subtitles. Defaults to "Arial".
        color (str, optional): The color name to use for the subtitles. Defaults to "red".
    """
    stylized_srt_file_file = f'{srt_file[:-4]}_style.srt'
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


def cli():
    def valid_model_name(name):
        if name in available_models() or os.path.exists(name):
            return name
        raise ValueError(
            f"model should be one of {available_models()} or path to a model checkpoint"
        )

    parser = argparse.ArgumentParser(
        description='Transcribe video or local video files, and generate subtitles files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video_file', type=str,
                        help="video file input can be either a video file or directory contains videos or youtube url")
    parser.add_argument("--model", default="small", type=valid_model_name, help="name of the Whisper model to use")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="the path to save model files; uses ~/.cache/whisper by default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="device to use for PyTorch inference")

    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"],
                        help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default=None,
                        choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
                        help="language spoken in the audio, specify None to perform language detection")

    parser.add_argument("--verbose", type=str2bool, default=True,
                        help="whether to print out the progress and debug messages")
    parser.add_argument("--output_format", "-f", type=str, default="all",
                        choices=["txt", "vtt", "srt", "tsv", "json", "all"],
                        help="format of the output file; if not specified, all available formats will be produced")

    parser.add_argument("--temperature", type=float, default=0, help="temperature to use for sampling")
    parser.add_argument("--best_of", type=optional_int, default=5,
                        help="number of candidates when sampling with non-zero temperature")
    parser.add_argument("--beam_size", type=optional_int, default=5,
                        help="number of beams in beam search, only applicable when temperature is zero")
    parser.add_argument("--patience", type=float, default=None,
                        help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    parser.add_argument("--length_penalty", type=float, default=None,
                        help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")

    parser.add_argument("--suppress_tokens", type=str, default="-1",
                        help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
    parser.add_argument("--initial_prompt", type=str, default=None,
                        help="optional text to provide as a prompt for the first window.")
    parser.add_argument("--condition_on_previous_text", type=str2bool, default=True,
                        help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")
    parser.add_argument("--fp16", type=str2bool, default=True,
                        help="whether to perform inference in fp16; True by default")

    parser.add_argument("--temperature_increment_on_fallback", type=optional_float, default=0.2,
                        help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
    parser.add_argument("--compression_ratio_threshold", type=optional_float, default=2.4,
                        help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
    parser.add_argument("--logprob_threshold", type=optional_float, default=-1.0,
                        help="if the average log probability is lower than this value, treat the decoding as failed")
    parser.add_argument("--no_speech_threshold", type=optional_float, default=0.6,
                        help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")
    parser.add_argument("--word_timestamps", type=str2bool, default=False,
                        help="(experimental) extract word-level timestamps and refine the results based on them")
    parser.add_argument("--prepend_punctuations", type=str, default="\"\'“¿([{-",
                        help="if word_timestamps is True, merge these punctuation symbols with the next word")
    parser.add_argument("--append_punctuations", type=str, default="\"\'.。,，!！?？:：”)]}、",
                        help="if word_timestamps is True, merge these punctuation symbols with the previous word")
    parser.add_argument("--highlight_words", type=str2bool, default=False,
                        help="(requires --word_timestamps True) underline each word as it is spoken in srt and vtt")
    parser.add_argument("--max_line_width", type=optional_int, default=None,
                        help="(requires --word_timestamps True) the maximum number of characters in a line before breaking the line")
    parser.add_argument("--max_line_count", type=optional_int, default=None,
                        help="(requires --word_timestamps True) the maximum number of lines in a segment")
    parser.add_argument("--max_words_per_line", type=optional_int, default=None,
                        help="(requires --word_timestamps True, no effect with --max_line_width) the maximum number of words in a segment")
    parser.add_argument("--threads", type=optional_int, default=0,
                        help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")
    parser.add_argument("--clip_timestamps", type=str, default="0",
                        help="comma-separated list start,end,start,end,... timestamps (in seconds) of clips to process, where the last end timestamp defaults to the end of the file")
    parser.add_argument("--hallucination_silence_threshold", type=optional_float,
                        help="(requires --word_timestamps True) skip silent periods longer than this threshold (in seconds) when a possible hallucination is detected")

    parser.add_argument('-v', '--video_output_dir', default='Video',
                        help='Directory to save downloaded YouTube videos.')
    parser.add_argument('-a', '--audio_output_dir', default='Audio',
                        help='Directory to save extracted audio files.')
    parser.add_argument('-s', '--srt_output_dir', default='Srt',
                        help='Directory to save generated SRT files.')

    parser.add_argument('--font_type', default='Arial', help='')
    parser.add_argument('--font_color', default='red', help='')

    start_time = time.time()

    args = parser.parse_args().__dict__
    video_file: str = args.pop("video_file")
    model_name: str = args.pop("model")
    device: str = args.pop("device")
    model_dir: str = args.pop("model_dir")
    output_format: str = args.pop("output_format")
    video_output_dir: str = args.pop("video_output_dir")
    audio_output_dir: str = args.pop("audio_output_dir")
    srt_output_dir: str = args.pop("srt_output_dir")
    font_type: str = args.pop("font_type")
    font_color: str = args.pop("font_color")
    os.makedirs(video_output_dir, exist_ok=True)
    os.makedirs(audio_output_dir, exist_ok=True)
    os.makedirs(srt_output_dir, exist_ok=True)

    video_exts = [".mp4", ".avi", ".mov", ".wmv", ".mkv", ".webm"]
    audio_exts = [".wav", ".mp3"]

    if video_file:
        # Check if the input is a file, a directory, or a YouTube URL
        if os.path.isfile(video_file):
            # Convert the file to audio if it has a valid extension
            if video_file.lower().endswith(tuple(video_exts)):
                convert_to_audio(video_file, audio_output_dir)
        elif os.path.isdir(video_file):
            # Convert all the files in the directory to audio if they have valid extensions
            for video in os.listdir(video_file):
                if video.lower().endswith(tuple(video_exts)):
                    convert_to_audio(os.path.join(video_file, video), audio_output_dir)
        elif re.match(r"^https?://www\.youtube\.com/watch\?v=[\w-]+", video_file):
            # Download the YouTube video and convert it to audio
            video_path = dld_ytb_video(video_file, video_output_dir)
            convert_to_audio(video_path, audio_output_dir)

    if model_name.endswith(".en") and args["language"] not in {"en", "English"}:
        if args["language"] is not None:
            warnings.warn(
                f"{model_name} is an English-only model but receipted '{args['language']}'; using English instead."
            )
        args["language"] = "en"

    temperature = args.pop("temperature")
    if (increment := args.pop("temperature_increment_on_fallback")) is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
    else:
        temperature = [temperature]

    if (threads := args.pop("threads")) > 0:
        torch.set_num_threads(threads)

    model = load_model(model_name, device=device, download_root=model_dir)

    writer = get_writer(output_format, srt_output_dir)
    word_options = [
        "highlight_words",
        "max_line_count",
        "max_line_width",
        "max_words_per_line",
    ]
    if not args["word_timestamps"]:
        for option in word_options:
            if args[option]:
                parser.error(f"--{option} requires --word_timestamps True")
    if args["max_line_count"] and not args["max_line_width"]:
        warnings.warn("--max_line_count has no effect without --max_line_width")
    if args["max_words_per_line"] and args["max_line_width"]:
        warnings.warn("--max_words_per_line has no effect with --max_line_width")
    writer_args = {arg: args.pop(arg) for arg in word_options}

    for audio_file_name in os.listdir(audio_output_dir):
        if audio_file_name.lower().endswith(tuple(audio_exts)):
            audio_file = os.path.join(audio_output_dir, audio_file_name)
            try:
                result = transcribe(model, audio_file, **args)
                writer(result, audio_file, **writer_args)
            except Exception as e:
                traceback.print_exc()
                print(f"Skipping {audio_file} due to {type(e).__name__}: {str(e)}")

    # create stylized srt file
    for file in os.listdir(srt_output_dir):
        # ensure not restyle the styled file
        if file.endswith("_style.srt"):
            continue
        if file.endswith(".srt"):
            adjust_srt_style(os.path.join(srt_output_dir, file), font_type, font_color)

    runtime = time.time() - start_time

    hours, remainder = divmod(runtime, 3600)  # Calculate hours and remaining seconds
    minutes, seconds = divmod(remainder, 60)  # Calculate minutes and seconds

    if hours > 0:
        logger.info(f"Total Execution time: {hours:.0f} hours {minutes:02.0f} minutes {seconds:02.0f} seconds")
    elif minutes > 0:
        logger.info(f"Total Execution time: {minutes:02.0f} minutes {seconds:02.0f} seconds")
    else:
        logger.info(f"Total Execution time: {seconds:02.0f} seconds")


if __name__ == "__main__":
    cli()
