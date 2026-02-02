# pipeline/config.py
import json

# --- Static Configuration ---
# All pipeline settings are stored in this dictionary.
CONFIG = {
    # --- Input & Output ---
    #"YOUTUBE_URL": 'https://youtu.be/IIPKMixTMfE?si=2vNst9fJPhmGcClX', # Change this to your video
    "BASE_OUTPUT_DIR": 'output/video_pipeline_output',
    "CLEAN_UP_BASE_OUTPUT_DIR_BEFORE_RUN": True,

    # --- Intelligent Editing Features ---
    "ENABLE_SILENCE_SNAPPING": True,
    "SILENCE_SNAP_CONFIG": {
        "SEARCH_WINDOW_SEC": 1.5,
        "MIN_SILENCE_LEN_MS": 300,
        "SILENCE_THRESH_DBFS_OFFSET": -30,
        "FALLBACK_BUFFER_SEC": 0.15
    },
    "ENABLE_FILLER_WORD_REMOVAL": True,
    "FILLER_WORD_LIST": {'um', 'uh', 'hmm', 'er', 'ah'},
    "PUNCH_IN_ZOOM_FACTOR": 1.08,

    # --- Pause Removal Settings ---
    "ENABLE_PAUSE_REMOVAL": False,
    "MIN_PAUSE_DURATION_SEC": 0.2,

    # --- Technical & Quality Settings ---
    "ENABLE_ASD_CROP": False, #switch for vertical crop and ASD
    "LOW_VRAM_MODE": True,
    "LOW_VRAM_TARGET_HEIGHT": 720,
    "CROP_ASPECT_RATIO_W": 9,
    "CROP_ASPECT_RATIO_H": 16,
    "CROP_SMOOTHING_ALPHA": 0.15,
    "VAD_AGGRESSIVENESS": 3,
    "VAD_SAMPLE_RATE": 16000,
    "VAD_FRAME_DURATION_MS": 30,
    "ENABLE_FORCED_SINGLE_SPEAKER_MODE": True,

    # --- Debugging ---
    "CREATE_DEBUG_VISUALIZATION_VIDEO_PER_SEGMENT": False,

    # --- Filenames (can be left as default) ---
    "SEGMENT_VIDEO_FILENAME": "segment.mp4",
    "SEGMENT_AUDIO_FILENAME": "segment_audio.wav",
    "CROPPED_SEGMENT_WITH_AUDIO_FILENAME": "cropped_segment_with_audio.mp4",
    "FINAL_CONCATENATED_VIDEO_FILENAME": "final_short_video.mp4",

    # --- External Tool Options ---
    "YT_DLP_OPTIONS": {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=mp4]/best[ext=mp4]',
        'noplaylist': True,
        'quiet': False,
        'merge_output_format': 'mp4'
    }
}