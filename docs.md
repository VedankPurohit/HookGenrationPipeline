# HookGen Technical Documentation

This document provides a detailed, code-level overview of the HookGen pipeline. It is intended for developers looking to understand the internal workings, architecture, and functionality of each module.

## Table of Contents

1.  [Entry Points](#1-entry-points)
    *   [`prepare.py`](#preparepy)
    *   [`main.py`](#mainpy)
2.  [The `pipeline` Package](#2-the-pipeline-package)
    *   [`config.py`](#configpy)
    *   [`LoggerSetup.py`](#loggersetuppy)
    *   [`utils.py`](#utilspy)
    *   [`VideoIO.py`](#videoiopy)
    *   [`GetTranscript.py`](#gettranscriptpy)
    *   [`TranscriptParser.py`](#transcriptparserpy)
    *   [`AudioAnalysis.py`](#audioanalysispy)
    *   [`VisionAnalysis.py`](#visionanalysispy)
    *   [`Blueprint.py`](#blueprintpy)
    *   [`EditingEffects.py`](#editingeffectspy)
    *   [`CropProcessing.py`](#cropprocessingpy)

---

## 1. Entry Points

These are the main scripts used to execute the pipeline.

### `prepare.py`

**Purpose**: Handles the initial asset gathering and processing stage. It takes a source URL, downloads the content, and generates the necessary transcripts for the production stage.

**Functions**:

*   `download_and_extract_audio(url, video_path, audio_path)`
    *   **Description**: Downloads a video from the given URL using `yt-dlp` and extracts its audio into a WAV file using `ffmpeg`.
    *   **Args**:
        *   `url` (str): The YouTube URL of the source video.
        *   `video_path` (str): The local path to save the downloaded video.
        *   `audio_path` (str): The local path to save the extracted audio.
    *   **Notes**: Skips operations if the output files already exist.

*   `main(project_name, youtube_url, download_only)`
    *   **Description**: The main orchestration function for the preparation stage. It sets up the project directory structure and calls the download and transcription functions.

### `main.py`

**Purpose**: The main entry point for the core video production pipeline. It orchestrates the process of turning source assets into a final edited video based on a creative brief.

**Functions**:

*   `run_production(...)`
    *   **Description**: Orchestrates the entire video processing workflow, from loading assets and models to generating a blueprint, processing clips, and assembling the final video.
    *   **Workflow**:
        1.  **Setup**: Defines directories, cleans up old run outputs, and initializes logging.
        2.  **Asset Loading**: Loads the source video, transcript, and creative brief. It can use a provided brief or generate one on the fly using an LLM.
        3.  **Model Loading**: Initializes the RetinaFace face detection model if cropping is enabled.
        4.  **Blueprint Creation**: Calls `pipeline.create_blueprint_from_llm` to get a precise shot list.
        5.  **Clip Processing Loop**: Iterates through the blueprint, processing each clip by calling `pipeline.extract_segment_reencode`, `pipeline.perform_asd_and_crop`, and `pipeline.refine_clip_by_removing_fillers`.
        6.  **Final Assembly**: Concatenates all processed clips into the final video.

---

## 2. The `pipeline` Package

This package contains all the core logic for the video processing pipeline.

### `config.py`

**Purpose**: Centralizes all static and dynamic configuration for the pipeline.

*   **`CONFIG` (dict)**: A dictionary holding all tunable parameters, file paths, and settings. This allows for easy adjustment of the pipeline's behavior without modifying the core logic.
*   **Dynamic Inputs**: The module also loads `inputs/creative_brief.json` and `inputs/transcript.json` at import time. This is a design choice for simplicity in this project, but in a larger system, this data would be passed explicitly.

### `LoggerSetup.py`

**Purpose**: Configures the application's logging system.

*   `setup_logging(run_output_dir)`
    *   **Description**: Configures a root logger that writes to both the console (INFO level) and a dedicated log file (DEBUG level) for each run. This ensures that all events are captured for debugging and reproducibility.

### `utils.py`

**Purpose**: Contains general-purpose utility functions used across the pipeline.

*   `timestamp_to_seconds(ts_str)`: Converts a timestamp string (e.g., "HH:MM:SS.ss") or number into total seconds.
*   `calculate_iou(boxA, boxB)`: Calculates the Intersection over Union (IoU) for two bounding boxes, a key metric for the face tracker.
*   `log_run_summary(...)`: Saves a JSON summary of the run's configuration and creative brief for reproducibility.

### `VideoIO.py`

**Purpose**: Handles fundamental input/output operations for video and audio files using `yt-dlp` and `ffmpeg-python`.

*   `download_youtube_video(...)`: Downloads a video from YouTube.
*   `extract_segment_reencode(...)`: Extracts a precise video segment by re-encoding it. It produces separate, synchronized video and audio files (WAV), which is crucial for reliable analysis.
*   `concatenate_video_clips(...)`: Joins a list of video clips into a single file using FFmpeg's `concat` filter for robustness.

### `GetTranscript.py`

**Purpose**: Handles the entire transcription process by interfacing with the Deepgram API.

*   `_transcribe_with_deepgram(audio_path)`: Sends an audio file to Deepgram and returns the raw JSON response.
*   `_create_llm_summary(deepgram_response)`: Formats the transcript into a token-efficient, speaker-separated summary suitable for an LLM.
*   `_parse_to_whisperx_format(deepgram_response)`: Converts the Deepgram output into the WhisperX-style format that the rest of the pipeline expects. This acts as a compatibility layer.
*   `generate_all_transcripts(audio_path, output_dir)`: The main orchestrator that calls the above functions to produce all necessary transcript files.

### `TranscriptParser.py`

**Purpose**: Provides utilities for parsing and manipulating transcript data.

*   `flatten_whisperx_transcript(...)`: Converts the nested segment/word structure of a transcript into a single flat list of words, which is much easier to search.
*   `find_phrase_in_transcription(...)`: Finds the precise start and end time of a sentence in the transcript by performing a cleaned, case-insensitive search on the flattened word list.

### `AudioAnalysis.py`

**Purpose**: Provides functions for analyzing audio streams to detect speech and silence.

*   `get_speech_segments_from_vad(...)`: Uses WebRTC Voice Activity Detection (VAD) to identify periods of speech in an audio file, returning a list of speech segment timestamps.
*   `find_best_silence_boundary(...)`: Given a timestamp, this function searches in a small window around it to find the nearest silence boundary. This is used for "silence snapping" to make cuts feel more natural.

### `VisionAnalysis.py`

**Purpose**: Handles all computer vision tasks, primarily focused on face detection and tracking.

*   **`SimpleFaceTracker` (class)**: A custom tracker that uses Intersection over Union (IoU) to assign and maintain a unique ID for each face across frames. It can handle temporary disappearances.
*   `detect_faces_in_video_segment(...)`: Uses the RetinaFace model to detect all faces in every frame of a video segment.
*   `apply_face_tracking(...)`: Instantiates `SimpleFaceTracker` and uses it to process the raw face detections, assigning persistent IDs.
*   `get_mouth_activity_per_track(...)`: A clever heuristic for determining who is speaking. It calculates the distance between mouth corners for each tracked face in every frame. A higher variance in this distance over time implies mouth movement (i.e., talking).
*   `determine_active_speaker_ids(...)`: The core of Active Speaker Detection. It correlates the speech segments from VAD (`AudioAnalysis`) with the mouth movement variance from `get_mouth_activity_per_track` to determine the most likely speaker ID for each frame.
*   `enforce_single_speaker_heuristics(...)`: Cleans up the speaker ID data by filling in small gaps and ensuring a single, consistent speaker is identified, preventing jarring camera switches.

### `Blueprint.py`

**Purpose**: Creates the final, precise production plan.

*   `create_blueprint_from_llm(...)`: Takes the high-level clip ideas from the creative brief and refines them. It uses `find_phrase_in_transcription` to get initial timestamps and, if enabled, `find_best_silence_boundary` to snap the start/end times to natural pauses.

### `EditingEffects.py`

**Purpose**: Applies programmatic editing effects to enhance the video.

*   `create_speaker_focused_cropped_clip(...)`: Generates the dynamically cropped video. It calculates the ideal crop center for each frame based on the active speaker's face and applies an exponential moving average to smooth the virtual "camera" movement. It writes frames to an FFmpeg pipe for efficiency.
*   `refine_clip_by_removing_fillers(...)`: Constructs a complex FFmpeg filter graph to cut out filler word segments. To hide the jump cuts, it applies a slight zoom effect to alternating segments.
*   `find_filler_words_in_segment(...)`: Identifies filler words in a segment's transcript and returns their relative start/end times for cutting.

### `CropProcessing.py`

**Purpose**: A high-level orchestration module for the ASD and cropping workflow.

*   `perform_asd_and_crop(...)`: This function is the main entry point for the smart cropping feature. It orchestrates the entire sequence:
    1.  Detects faces (`VisionAnalysis`).
    2.  Tracks faces (`VisionAnalysis`).
    3.  Detects speech (`AudioAnalysis`).
    4.  Analyzes mouth movement (`VisionAnalysis`).
    5.  Determines the active speaker (`VisionAnalysis`).
    6.  Generates the final cropped clip (`EditingEffects`).
    *   **Fallback Logic**: Crucially, if any step fails or if the feature is disabled, it gracefully returns the original, uncropped video path, ensuring the main pipeline does not halt.
