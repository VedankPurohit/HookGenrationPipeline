# pipeline/__init__.py

"""
This file initializes the 'pipeline' package, making it a single, cohesive library.

It imports the key functions and classes from the various submodules and exposes them
at the top level of the package. This creates a clean, unified public API.

Instead of a user needing to know the internal file structure (e.g., `from pipeline.io_video import ...`),
they can simply do `import pipeline` and access `pipeline.download_youtube_video`.

The `__all__` variable explicitly defines which names are part of this public API.
"""
import yt_dlp

# --- Configuration and Core Data ---
# Expose the main CONFIG dictionary and the loaded input data.
from .config import CONFIG, llm_generated_clips, Transcript

# --- Utility Functions ---
# Expose general-purpose helper functions.
from .utils import timestamp_to_seconds, calculate_iou, log_run_summary

from .LoggerSetup import setup_logging

# --- Transcript Parsing Module ---
# Expose functions for handling the transcript data.
from .TranscriptParser import flatten_whisperx_transcript, find_phrase_in_transcription

# --- I/O Module ---
# Expose functions for downloading, cutting, and joining video files.
from .VideoIO import download_youtube_video, extract_segment_reencode, concatenate_video_clips

# --- Audio Analysis Module ---
# Expose functions for analyzing audio content (VAD, silence).
from .AudioAnalysis import get_speech_segments_from_vad, find_best_silence_boundary

# --- Vision Analysis Module ---
# Expose all classes and functions related to computer vision analysis.
from .VisionAnalysis import (
    SimpleFaceTracker,
    detect_faces_in_video_segment,
    apply_face_tracking,
    calculate_mouth_corner_distance,
    get_mouth_activity_per_track,
    determine_active_speaker_ids,
    enforce_single_speaker_heuristics
)

# --- High-Level Processing Steps ---
from .CropProcessing import perform_asd_and_crop

# --- Editing Effects Module ---
# Expose the creative editing functions.
from .EditingEffects import (
    get_speaker_bbox_for_frame_cropping,
    create_speaker_focused_cropped_clip,
    find_filler_words_in_segment,
    refine_clip_by_removing_fillers
)

# --- Blueprint Creation Module ---
# Expose the main function for creating the production plan.
from .Blueprint import create_blueprint_from_llm


# Define the public API of the 'pipeline' package.
# When a user writes `from pipeline import *`, only these names will be imported.
__all__ = [
    # Config
    "CONFIG",
    "llm_generated_clips",
    "Transcript",
    # Utils
    "timestamp_to_seconds",
    "calculate_iou",
    "log_run_summary",
    "setup_logging"
    # Transcript
    "flatten_whisperx_transcript",
    "find_phrase_in_transcription",
    # I/O
    "download_youtube_video",
    "extract_segment_reencode",
    "concatenate_video_clips",
    # Audio
    "get_speech_segments_from_vad",
    "find_best_silence_boundary",
    # Vision
    "SimpleFaceTracker",
    "detect_faces_in_video_segment",
    "apply_face_tracking",
    "calculate_mouth_corner_distance",
    "get_mouth_activity_per_track",
    "determine_active_speaker_ids",
    "enforce_single_speaker_heuristics",
    # Editing
    "get_speaker_bbox_for_frame_cropping",
    "create_speaker_focused_cropped_clip",
    "find_filler_words_in_segment",
    "refine_clip_by_removing_fillers",
    # Blueprint
    "create_blueprint_from_llm",
    "perform_asd_and_crop"
]