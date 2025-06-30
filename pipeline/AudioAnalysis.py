import ffmpeg
import os
import wave
import webrtcvad
from pydub import AudioSegment
from pydub.silence import detect_silence
import logging
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(f"pipeline.{__name__}")

def get_speech_segments_from_vad(
    audio_path: str,
    sample_rate: int,
    frame_duration_ms: int,
    aggressiveness: int,
    silence_frames_thresh_factor: float,
    speech_frames_thresh_factor: float
) -> List[Tuple[int, int]]:
    """
    Detects speech segments in an audio file using WebRTC Voice Activity Detection (VAD).

    Args:
        audio_path (str): Path to the input audio file (must be mono, 16-bit, 16kHz WAV).
        sample_rate (int): The sample rate of the audio (e.g., 16000 Hz).
        frame_duration_ms (int): The duration of each VAD frame in milliseconds (e.g., 10, 20, or 30).
        aggressiveness (int): An integer between 0 and 3, 0 being the least aggressive
                              and 3 being the most aggressive in filtering out non-speech.
        silence_frames_thresh_factor (float): Factor to determine the minimum number of
                                              consecutive silent frames to consider a pause.
        speech_frames_thresh_factor (float): Factor to determine the minimum number of
                                             consecutive speech frames to consider speech.

    Returns:
        List[Tuple[int, int]]: A list of tuples, where each tuple (start_frame_idx, end_frame_idx)
                               represents a detected speech segment in VAD frames.
    """
    if not os.path.exists(audio_path):
        logger.error(f"Audio file for VAD not found: {audio_path}")
        return []

    try:
        with wave.open(audio_path, 'rb') as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != sample_rate:
                logger.error(f"Audio format error for VAD. Expected mono, 16-bit, {sample_rate}Hz. Got: channels={wf.getnchannels()}, sample_width={wf.getsampwidth()}, framerate={wf.getframerate()}")
                return []
            
            vad = webrtcvad.Vad()
            vad.set_mode(aggressiveness)
            
            bytes_per_vad_frame = (sample_rate * frame_duration_ms // 1000) * wf.getsampwidth()
            num_audio_samples_per_vad_frame = bytes_per_vad_frame // wf.getsampwidth()
            
            audio_frames_data = []
            while True:
                frame_data = wf.readframes(num_audio_samples_per_vad_frame)
                if len(frame_data) < bytes_per_vad_frame:
                    break # End of file or incomplete frame
                audio_frames_data.append(frame_data)
    except Exception as e:
        logger.error(f"Error opening/reading audio for VAD at {audio_path}: {e}", exc_info=True)
        return []

    if not audio_frames_data:
        logger.warning("No audio frames processed for VAD.")
        return []

    is_speech_flags = [vad.is_speech(f, sample_rate) if len(f) == bytes_per_vad_frame else False for f in audio_frames_data]
    
    speech_segments_vad_frames: List[Tuple[int, int]] = []
    speaking_state = False
    segment_start_idx = 0
    
    
    frames_per_second_vad = 1000 / frame_duration_ms if frame_duration_ms > 0 else 33.33 # Default to 30ms frame if 0
    vad_silence_thresh = int(silence_frames_thresh_factor * frames_per_second_vad)
    vad_speech_thresh = int(speech_frames_thresh_factor * frames_per_second_vad)
    
    consecutive_speech, consecutive_silence = 0, 0

    for i, is_speech_flag in enumerate(is_speech_flags):
        if is_speech_flag:
            consecutive_speech += 1
            consecutive_silence = 0
            if not speaking_state and consecutive_speech >= vad_speech_thresh:
                speaking_state = True
                segment_start_idx = i - (consecutive_speech - 1) # Start of the speech burst
        else:
            consecutive_silence += 1
            consecutive_speech = 0
            if speaking_state and consecutive_silence >= vad_silence_thresh:
                speaking_state = False
                speech_segments_vad_frames.append((segment_start_idx, i - consecutive_silence))
    
    # If still speaking at the end of the audio
    if speaking_state:
        speech_segments_vad_frames.append((segment_start_idx, len(is_speech_flags) - 1))
    
    logger.info(f"VAD detected {len(speech_segments_vad_frames)} speech segments from audio.")
    return speech_segments_vad_frames


def find_best_silence_boundary(
    video_path: str,
    anchor_time: float,
    direction: str,
    config: Dict[str, Any],
    temp_dir: str
) -> float:
    """
    Finds the best silence boundary near an anchor time in a given direction.

    This function extracts a small audio segment around the anchor time,
    detects silences within it, and returns a new timestamp that aligns
    with a silence boundary. If no suitable silence is found, it falls back
    to a buffered time.

    Args:
        video_path (str): The path to the source video file.
        anchor_time (float): The timestamp (in seconds) around which to search.
        direction (str): The direction to search for silence ('before' or 'after').
        config (Dict[str, Any]): Configuration dictionary containing:
                                 - 'SEARCH_WINDOW_SEC': How far to search for silence.
                                 - 'FALLBACK_BUFFER_SEC': Buffer to apply if no silence found.
                                 - 'MIN_SILENCE_LEN_MS': Minimum length of silence to detect.
                                 - 'SILENCE_THRESH_DBFS_OFFSET': dBFS offset for silence detection.
        temp_dir (str): Temporary directory to store extracted audio segments.

    Returns:
        float: The new timestamp (in seconds) snapped to a silence boundary,
               or a fallback time if no silence is found.
    """
    search_window = config['SEARCH_WINDOW_SEC']
    fallback_buffer = config['FALLBACK_BUFFER_SEC']

    if direction == 'before':
        context_start = max(0.0, anchor_time - search_window)
        context_end = anchor_time
    elif direction == 'after':
        context_start = anchor_time
        context_end = anchor_time + search_window
    else:
        logger.error(f"Invalid direction specified for silence boundary search: {direction}. Must be 'before' or 'after'.")
        return anchor_time # Return original anchor time on invalid direction

    context_duration = context_end - context_start
    if context_duration <= 0.01: # Handle very small or negative durations
        logger.debug(f"Context duration too small ({context_duration:.2f}s) for anchor {anchor_time:.2f}s, direction {direction}. Returning fallback.")
        return anchor_time - fallback_buffer if direction == 'before' else anchor_time + fallback_buffer

    temp_audio_path = os.path.join(temp_dir, f"context_audio_{direction}_{anchor_time:.2f}.wav")
    os.makedirs(temp_dir, exist_ok=True) # Ensure temp directory exists

    try:
        # Extract audio segment for analysis
        ffmpeg.input(video_path, ss=context_start, t=context_duration)\
            .output(temp_audio_path, acodec='pcm_s16le', ac=1, ar=16000)\
            .overwrite_output()\
            .run(quiet=True, capture_stdout=True, capture_stderr=True)

        audio = AudioSegment.from_wav(temp_audio_path)
        if len(audio) == 0:
            logger.debug(f"Extracted audio segment is empty for anchor {anchor_time:.2f}s. Returning fallback.")
            return anchor_time - fallback_buffer if direction == 'before' else anchor_time + fallback_buffer

        # Detect silence
        silence_thresh = audio.max_dBFS + config['SILENCE_THRESH_DBFS_OFFSET']
        silence_chunks = detect_silence(
            audio, min_silence_len=config['MIN_SILENCE_LEN_MS'], silence_thresh=silence_thresh
        )

        if not silence_chunks:
            logger.debug(f"No silence chunks found for anchor {anchor_time:.2f}s. Returning fallback.")
            return anchor_time - fallback_buffer if direction == 'before' else anchor_time + fallback_buffer

        # Determine the best boundary based on direction
        if direction == 'before':
            # Find the end of the silence chunk closest to (or before) the anchor
            # We want the latest possible silence end before the anchor
            best_boundary_ms = silence_chunks[-1][1] # Last silence chunk's end
        else: # direction == 'after'
            # Find the start of the silence chunk closest to (or after) the anchor
            # We want the earliest possible silence start after the anchor
            best_boundary_ms = silence_chunks[0][0] # First silence chunk's start
        
        new_timestamp = context_start + best_boundary_ms / 1000.0
        logger.debug(f"Found silence boundary at {new_timestamp:.2f}s for anchor {anchor_time:.2f}s ({direction}).")
        return new_timestamp

    except ffmpeg.Error as e:
        stderr_output = e.stderr.decode('utf-8', errors='ignore') if e.stderr else 'None'
        logger.warning(f"FFmpeg error during silence analysis for anchor {anchor_time:.2f}s ({direction}): {stderr_output}")
        return anchor_time - fallback_buffer if direction == 'before' else anchor_time + fallback_buffer
    except Exception as e:
        logger.warning(f"General exception during silence analysis for anchor {anchor_time:.2f}s ({direction}): {e}", exc_info=True)
        return anchor_time - fallback_buffer if direction == 'before' else anchor_time + fallback_buffer
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)