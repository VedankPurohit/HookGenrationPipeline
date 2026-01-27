import os
import json
import argparse
import ffmpeg
import yt_dlp
import subprocess
from dotenv import load_dotenv

# Import the new centralized function from the pipeline package
from pipeline.GetTranscript import generate_all_transcripts

# Load API keys from .env file
load_dotenv()

def has_audio_stream(video_path: str) -> bool:
    """Check if a video file contains audio streams."""
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', video_path
        ], capture_output=True, text=True, check=True)

        streams = json.loads(result.stdout).get('streams', [])
        return any(stream.get('codec_type') == 'audio' for stream in streams)
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
        return False

def download_and_extract_audio(url: str, video_path: str, audio_path: str):
    """Downloads a YouTube video and extracts its audio if they don't exist."""
    # TODO: The `yt-dlp` options could be externalized to `config.py` to allow for easier
    # customization of download quality or format without modifying the script.
    # Download video
    if not os.path.exists(video_path):
        print(f"-> Downloading video: {url}")
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=mp4]/best[ext=mp4]',
            'outtmpl': video_path,
            'noplaylist': True,
            'quiet': False,
            'merge_output_format': 'mp4'
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print("✅ Video downloaded.")
    else:
        print("-> Video already exists. Skipping download.")

    # Extract audio
    if not os.path.exists(audio_path):
        if has_audio_stream(video_path):
            print(f"-> Extracting audio from {video_path}")
            try:
                ffmpeg.input(video_path).output(
                    audio_path, acodec='pcm_s16le', ar='16000', ac=1
                ).run(quiet=True, overwrite_output=True)
                print("✅ Audio extracted.")
            except ffmpeg.Error as e:
                print(f"❌ FFMPEG ERROR during audio extraction: {e.stderr.decode()}")
                raise
        else:
            print("⚠️  Warning: Video contains no audio streams. Skipping audio extraction.")
            print("   Note: Audio-dependent features (transcription, emotion analysis) will not be available.")
            # Create an empty audio file as a placeholder
            with open(audio_path, 'wb') as f:
                pass  # Empty file
    else:
        print("-> Audio already exists. Skipping extraction.")

def main(project_name: str, youtube_url: str, download_only: bool):
    """
    Main function for the Preparation Stage.
    Sets up the project directory and generates all necessary source assets.
    """
    print(f"--- PREPARATION STAGE: Project '{project_name}' ---")

    # Define the directory structure
    project_dir = os.path.join("Output", project_name)
    source_assets_dir = os.path.join(project_dir, "source_assets")
    os.makedirs(source_assets_dir, exist_ok=True)
    os.makedirs(os.path.join(project_dir, "runs"), exist_ok=True)

    # Define file paths
    video_path = os.path.join(source_assets_dir, "source_video.mp4")
    audio_path = os.path.join(source_assets_dir, "source_audio.wav")

    # Step 1: Download video and extract audio
    # TODO: Add a more robust asset validation step here. For example, check if the
    # downloaded video has a valid duration or if the extracted audio is not silent.
    download_and_extract_audio(youtube_url, video_path, audio_path)

    # Step 2: Transcribe audio if not in download-only mode
    if not download_only:
        # Check if we have actual audio content (not just an empty placeholder)
        if os.path.getsize(audio_path) == 0:
            print("⚠️  Skipping transcription: No audio available in this video.")
            print("   Audio-dependent features (transcription, emotion analysis) are not available.")
        else:
            try:
                generate_all_transcripts(audio_path, source_assets_dir)
            except Exception as e:
                print(f"❌ ERROR: Transcription failed: {e}")
                return # Stop if transcription failed

    print(f"\n✅ Preparation complete for project '{project_name}'.")
    if not download_only:
        print("You can now generate a creative brief and run the main pipeline.")
    else:
        print("Remember to place your transcript.json, deepgram_raw_transcript.json, and llm_summary.txt in the source_assets directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare project assets for the video pipeline.")
    parser.add_argument("--project-name", type=str, required=True, help="A unique name for the project (e.g., 'sundar-pichai-interview').")
    parser.add_argument("--url", type=str, required=True, help="The YouTube URL of the source video.")
    parser.add_argument(
        "--download-only",
        action = "store_true",
        help = "if set then only the video will be downloaded. it is then expected of you to place othere required files at the correct location"
    )
    args = parser.parse_args()
    main(args.project_name, args.url, args.download_only)