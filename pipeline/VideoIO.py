import yt_dlp
import ffmpeg
import os
import shutil
import logging
from typing import List, Optional, Dict, Union, Any, Tuple
from collections import Counter

from .utils import timestamp_to_seconds

logger = logging.getLogger(f"pipeline.{__name__}")


def download_youtube_video(youtube_url: str, output_dir: str, filename: str = "original_video.mp4") -> Optional[str]:
    """
    Downloads a video from a given YouTube URL using yt-dlp.

    This function attempts to download the best quality MP4 video and audio streams
    and merge them into a single file. It skips the download if the target file
    already exists in the specified output directory.

    Args:
        youtube_url (str): The URL of the YouTube video to download.
        output_dir (str): The absolute path to the directory where the video file will be saved.
                          The directory will be created if it does not exist.
        filename (str): The desired filename for the downloaded video (e.g., "source_video.mp4").

    Returns:
        Optional[str]: The absolute path to the downloaded video file if successful,
                       or None if the download failed due to an error.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path: str = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        logger.info(f"☑️ Video '{output_path}' already exists. Skipping download.")
        return output_path

    logger.info(f"⬇️ Attempting to download: {youtube_url} to {output_path}")
    
    ydl_opts: Dict[str, Any] = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4', # Prioritize mp4 video and m4a audio
        'outtmpl': output_path, # Output template for the filename
        'noplaylist': True, # Do not download if the URL is part of a playlist
        'quiet': False, # Show progress and errors
        'merge_output_format': 'mp4' # Ensure the final merged file is mp4
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        logger.info(f"✅ Video downloaded successfully to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"❌ Error downloading video from {youtube_url}: {e}", exc_info=True)
        return None

def get_video_resolution(video_path: str) -> Optional[Tuple[int, int]]:
    """
    Get the resolution (width, height) of a video file.

    Args:
        video_path (str): Path to the video file

    Returns:
        Optional[Tuple[int, int]]: (width, height) tuple or None if probe fails
    """
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
        if video_stream:
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            return (width, height)
    except Exception as e:
        logger.warning(f"Could not probe resolution for {video_path}: {e}")
    return None

def concatenate_video_clips(video_clip_paths_list: List[str], final_output_path: str) -> Optional[str]:
    """
    Concatenates a list of video clips into a single output file.

    This function uses FFmpeg's robust 'concat' filter, which re-encodes streams
    to handle potential mismatches in timestamps, resolutions, or codecs between
    the input clips. This ensures a smooth and consistent final video.

    Args:
        video_clip_paths_list (List[str]): A list of absolute file paths for the video
                                           clips to be joined. The clips will be concatenated
                                           in the order they appear in this list.
        final_output_path (str): The absolute file path where the final concatenated video
                                 will be saved.

    Returns:
        Optional[str]: The absolute path to the final concatenated video if successful,
                       otherwise None if an error occurred or no clips were provided.
    """
    if not video_clip_paths_list:
        logger.warning("No video clips provided for concatenation. Returning None.")
        return None
    
    if len(video_clip_paths_list) == 1:
        logger.info(f"Only one clip provided. Copying '{video_clip_paths_list[0]}' to final output '{final_output_path}'.")
        try:
            shutil.copy(video_clip_paths_list[0], final_output_path)
            return final_output_path
        except Exception as e:
            logger.error(f"Error copying single clip to final output: {e}", exc_info=True)
            return None

    logger.info(f"🎬 Concatenating {len(video_clip_paths_list)} clips into {final_output_path} using FFmpeg concat filter...")

    # Check for resolution consistency and handle mismatches
    resolutions = []
    for path in video_clip_paths_list:
        res = get_video_resolution(path)
        if res:
            resolutions.append(res)
        else:
            logger.warning(f"Could not determine resolution for {path}, proceeding anyway")

    if resolutions:
        unique_resolutions = list(set(resolutions))
        if len(unique_resolutions) > 1:
            logger.warning(f"⚠️  Resolution mismatch detected! Found {len(unique_resolutions)} different resolutions: {unique_resolutions}")
            print(f"⚠️  Resolution mismatch: {unique_resolutions} - normalizing all clips")

            # Find the most common resolution, or use the first clip's resolution as standard
            res_counts = Counter(resolutions)
            standard_resolution = res_counts.most_common(1)[0][0]
            std_width, std_height = standard_resolution

            logger.info(f"📐 Normalizing all clips to {std_width}x{std_height}")

            # Create normalized input streams
            input_streams = []
            for path in video_clip_paths_list:
                clip_res = get_video_resolution(path)
                if clip_res and clip_res != standard_resolution:
                    # Scale this clip to match standard resolution
                    logger.debug(f"Scaling {os.path.basename(path)} from {clip_res[0]}x{clip_res[1]} to {std_width}x{std_height}")
                    scaled_stream = ffmpeg.input(path).filter('scale', std_width, std_height)
                    input_streams.append(scaled_stream)
                else:
                    # Resolution already matches or unknown, use as-is
                    input_streams.append(ffmpeg.input(path))
        else:
            logger.info(f"✅ All clips have consistent resolution: {unique_resolutions[0]}")
            input_streams = [ffmpeg.input(path) for path in video_clip_paths_list]
    else:
        # Could not probe resolutions, proceed without normalization
        logger.warning("Could not probe resolutions for clips, proceeding without normalization")
        input_streams = [ffmpeg.input(path) for path in video_clip_paths_list]

    video_parts = [stream.video for stream in input_streams]
    audio_parts = [stream.audio for stream in input_streams]

    # Concatenate video and audio streams separately
    concatenated_video = ffmpeg.concat(*video_parts, v=1, a=0) # v=1 for video, a=0 for no audio
    concatenated_audio = ffmpeg.concat(*audio_parts, v=0, a=1) # v=0 for no video, a=1 for audio

    try:
        (ffmpeg.output(
            concatenated_video,
            concatenated_audio,
            final_output_path,
            vcodec='libx264', # Video codec
            acodec='aac' # Audio codec
        )
        .overwrite_output() # Overwrite output file if it exists
        .run(capture_stdout=True, capture_stderr=True)) # Run FFmpeg command

        logger.info(f"✅ Successfully concatenated clips into: {final_output_path}")
        return final_output_path

    except ffmpeg.Error as e:
        stderr_info = e.stderr.decode('utf8', errors='ignore').strip().replace('\n', '\n   ')
        logger.error(f"❌ FFMPEG ERROR during final concatenation: {stderr_info}")
        return None
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred during concatenation: {e}", exc_info=True)
        return None

def extract_segment_reencode(
    input_video_path: str,
    start_ts: Union[str, float, int],
    end_ts: Union[str, float, int],
    output_segment_video_path: str,
    output_segment_audio_path: str,
    config_dict: Dict[str, Any]
) -> bool:
    """
    Extracts a video segment and re-encodes it into separate video and audio files.

    This two-step process ensures that the extracted video segment is a clean,
    independently usable file, and the audio is in a standardized WAV format
    suitable for further analysis (e.g., Voice Activity Detection).

    Args:
        input_video_path (str): The absolute path to the source video file.
        start_ts (Union[str, float, int]): The start timestamp of the segment to extract.
                                            Can be a string (e.g., "00:01:30"), float (seconds),
                                            or int (seconds).
        end_ts (Union[str, float, int]): The end timestamp of the segment to extract.
                                          Same format as `start_ts`.
        output_segment_video_path (str): The absolute path where the resulting video segment
                                         will be saved.
        output_segment_audio_path (str): The absolute path where the resulting audio segment
                                         (WAV format) will be saved.
        config_dict (Dict[str, Any]): The configuration dictionary, used for settings like
                                     'LOW_VRAM_MODE' and 'VAD_SAMPLE_RATE'.

    Returns:
        bool: True if both video and audio segments were successfully extracted and re-encoded,
              False otherwise.
    """
    # Check if segments already exist to avoid redundant processing
    if os.path.exists(output_segment_video_path) and os.path.exists(output_segment_audio_path):
        logger.debug(f"Segments '{os.path.basename(output_segment_video_path)}' and '{os.path.basename(output_segment_audio_path)}' already exist. Skipping extraction.")
        return True

    logger.info(f"-> Extracting segment from {start_ts} to {end_ts} from '{os.path.basename(input_video_path)}'...")
    try:
        # Convert timestamps to seconds for FFmpeg
        s_start: float = timestamp_to_seconds(start_ts)
        s_end: float = timestamp_to_seconds(end_ts)
        duration: float = s_end - s_start

        if duration <= 0:
            logger.error(f"Invalid duration ({duration:.2f}s) for segment [{s_start:.2f}s - {s_end:.2f}s]. Skipping extraction.")
            return False

        # TODO: The FFmpeg preset ('ultrafast') and CRF value (23) are hardcoded. These directly
    # impact the output quality vs. speed trade-off. Moving them to `config.py` would allow
    # users to choose between a quick, lower-quality preview and a slow, high-quality final render.
    # --- STEP 1: Create a clean video segment (with audio) ---
        # This step re-encodes the video to a consistent format and resolution if LOW_VRAM_MODE is enabled.
        output_params_video: Dict[str, Any] = {
            'vcodec': 'libx264', # H.264 video codec
            'preset': "ultrafast", # Fastest encoding preset
            'crf': 23, # Constant Rate Factor (quality setting, lower is better quality)
            'acodec': 'aac', # AAC audio codec
            'strict': 'experimental' # Required for AAC codec
        }
        if config_dict.get("LOW_VRAM_MODE", False):
            # Scale video down to target height to reduce processing load
            target_height = config_dict.get("LOW_VRAM_TARGET_HEIGHT", 720)
            output_params_video['vf'] = f'scale=-2:{target_height}' # -2 maintains aspect ratio

        ffmpeg.input(input_video_path, ss=s_start, t=duration)\
            .output(output_segment_video_path, **output_params_video)\
            .overwrite_output()\
            .run(capture_stdout=True, capture_stderr=True)

        # --- STEP 2: Extract WAV audio from the newly created video segment ---
        # This ensures the audio is perfectly synchronized with the extracted video segment.
        logger.debug(f"Extracting audio to {output_segment_audio_path} from video segment...")
        ffmpeg.input(output_segment_video_path)\
            .output(output_segment_audio_path, acodec='pcm_s16le', ar=config_dict["VAD_SAMPLE_RATE"], ac=1)\
            .overwrite_output()\
            .run(capture_stdout=True, capture_stderr=True)
        logger.debug(f"Audio extraction complete. File size: {os.path.getsize(output_segment_audio_path)} bytes.")

        # Final validation of created files
        if not os.path.exists(output_segment_video_path) or os.path.getsize(output_segment_video_path) == 0:
            raise IOError(f"Video segment file '{output_segment_video_path}' was not created or is empty.")
        if not os.path.exists(output_segment_audio_path) or os.path.getsize(output_segment_audio_path) == 0:
            raise IOError(f"Audio segment file '{output_segment_audio_path}' was not created or is empty.")
        
        logger.info(f"✅ Segment extracted and re-encoded successfully to '{os.path.basename(output_segment_video_path)}' and '{os.path.basename(output_segment_audio_path)}'.")
        return True

    except ffmpeg.Error as e:
        stderr_info = e.stderr.decode('utf8', errors='ignore').strip().replace('\n', '\n     ')
        logger.error(f"❌ FFMPEG ERROR during segment extraction: {stderr_info}")
        return False
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred during segment extraction: {e}", exc_info=True)
        return False
