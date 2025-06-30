'''
This module is a prototype for a future feature: Emotion-Aware Editing.

**Feature Objective:**
To analyze the emotional content of speech in video segments and use this data
to make more intelligent editing decisions. For example, the pipeline could
automatically select clips with high emotional intensity (excitement, happiness)
for a more engaging hook, or apply different visual effects based on the
detected emotion.

**Current Status:**
This module contains the core logic for analyzing an audio segment and
determining its dominant emotion using a pre-trained Hugging Face model.
The implementation was paused because the analysis proved to be time-consuming
and was deferred in favor of refining the core video processing pipeline.

**Implementation Details:**
- Uses the 'ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition' model from Hugging Face.
- Extracts a specific time-bound segment from a larger audio file using FFmpeg.
- Processes the segment through the audio-classification pipeline.
- Returns the dominant emotion and a confidence score.

**To-Do for Full Integration:**
1.  Integrate this analysis into the main `run_production` loop in `main.py`.
2.  Store the emotion data in the `final_blueprint` for each clip.
3.  Develop logic in `Blueprint.py` or a new module to use emotion data for
    clip selection or to tag clips for specific editing effects.
4.  Add configuration options in `config.py` to enable/disable this feature
    and set thresholds for emotional intensity.
'''

import os
import logging
import tempfile
import ffmpeg
import torch
from transformers import pipeline
from typing import Dict, Optional, Any

# It's good practice to have a dedicated logger for each module.
logger = logging.getLogger(f"pipeline.{__name__}")

# Global variable to hold the initialized pipeline, preventing re-loading on every call.
# This is a common pattern for expensive-to-load models.
_emotion_recognition_pipeline: Optional[Any] = None

def _initialize_emotion_pipeline() -> Optional[Any]:
    """
    Initializes the Hugging Face pipeline for emotion recognition.

    This function checks if the pipeline has already been loaded and, if not,
    initializes it. It intelligently selects between GPU and CPU.

    Returns:
        Optional[Any]: The initialized pipeline object, or None if initialization fails.
    """
    global _emotion_recognition_pipeline
    if _emotion_recognition_pipeline is not None:
        return _emotion_recognition_pipeline

    try:
        # Determine the device to use (GPU if available, otherwise CPU)
        device_id = 0 if torch.cuda.is_available() else -1
        device_name = "GPU" if device_id == 0 else "CPU"
        logger.info(f"Initializing emotion recognition pipeline for the first time on {device_name}...")

        # The user should ideally provide their token via an environment variable
        # or a secure secrets management system.
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.warning("Hugging Face token (HF_TOKEN) not found in environment variables. "
                           "Proceeding without authentication. Model access may be restricted.")

        _emotion_recognition_pipeline = pipeline(
            "audio-classification",
            model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
            token=hf_token,
            device=device_id
        )
        logger.info("✅ Emotion recognition pipeline initialized successfully.")
        return _emotion_recognition_pipeline

    except ImportError:
        logger.error("The 'transformers' or 'torch' library is not installed. "
                       "Please install them with 'pip install transformers torch'.")
        return None
    except Exception as e:
        logger.error(f"❌ Failed to initialize Hugging Face pipeline: {e}", exc_info=True)
        return None


def get_emotion_intensity_for_segment(
    audio_file_path: str,
    start_time: float,
    end_time: float
) -> Optional[Dict[str, Any]]:
    """
    Analyzes a specific audio segment to determine its dominant emotion and intensity.

    Args:
        audio_file_path (str): The path to the source audio file (e.g., 'source_audio.wav').
        start_time (float): The start timestamp of the segment in seconds.
        end_time (float): The end timestamp of the segment in seconds.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing the dominant emotion ('emotion')
                                  and its confidence score ('intensity'). Returns None if
                                  the analysis fails.
                                  Example: {'emotion': 'happy', 'intensity': 0.89}
    """
    # Step 1: Ensure the analysis pipeline is ready.
    emotion_pipeline = _initialize_emotion_pipeline()
    if not emotion_pipeline:
        logger.error("Emotion analysis pipeline is not available. Aborting analysis.")
        return None

    duration = end_time - start_time
    if duration <= 0.1: # Segments that are too short may fail analysis.
        logger.warning(f"Segment duration is too short ({duration:.2f}s) for emotion analysis. Skipping.")
        return None

    # Step 2: Use a temporary file to store the extracted audio segment.
    # This is safer than in-memory solutions for larger files.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmpfile:
        temp_audio_path = tmpfile.name

        try:
            # Step 3: Extract the precise audio segment using FFmpeg.
            (
                ffmpeg.input(audio_file_path, ss=start_time, t=duration)
                .output(temp_audio_path, acodec='pcm_s16le', ar=16000, ac=1)
                .run(quiet=True, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            )

            # Step 4: Perform the emotion analysis on the extracted segment.
            results = emotion_pipeline(temp_audio_path)

            # The pipeline returns a list of dictionaries. We want the one with the highest score.
            if not results:
                logger.warning("Emotion analysis returned no results for the segment.")
                return None

            dominant_emotion = max(results, key=lambda x: x['score'])
            return {
                "emotion": dominant_emotion['label'],
                "intensity": dominant_emotion['score']
            }

        except ffmpeg.Error as e:
            stderr = e.stderr.decode(errors='ignore')
            logger.error(f"FFmpeg error during audio segment extraction for emotion analysis: {stderr}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during emotion analysis: {e}", exc_info=True)
            return None

if __name__ == '__main__':
    # This block demonstrates how the function would be used in a real script.
    # It requires a sample audio file to exist at the specified path.
    print("--- EmotionAnalysis Module Demonstration ---")

    # Create a dummy logger and handler for demonstration if no root logger is set up.
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # In a real scenario, these would come from your project's assets.
    SAMPLE_AUDIO = "Output/my-interview-project/source_assets/source_audio.wav"
    # This is a mock transcript structure for demonstration.
    mock_transcript = {
        "segments": [
            {"start": 10.0, "end": 15.0, "text": "This was a really exciting moment for us."},
            {"start": 25.0, "end": 28.0, "text": "And that was quite surprising."},
            {"start": 40.0, "end": 45.0, "text": "I felt very happy with the outcome."}
        ]
    }

    if not os.path.exists(SAMPLE_AUDIO):
        print(f"\nWARNING: Sample audio file not found at '{SAMPLE_AUDIO}'.")
        print("Cannot run live demonstration. Please run 'prepare.py' for a project first.")
    else:
        print(f"\nAnalyzing segments from '{SAMPLE_AUDIO}'...")
        for i, segment in enumerate(mock_transcript["segments"]):
            start = segment["start"]
            end = segment["end"]
            print(f"\nAnalyzing segment {i+1} ({start}s - {end}s): "{segment['text']}"")

            emotion_data = get_emotion_intensity_for_segment(SAMPLE_AUDIO, start, end)

            if emotion_data:
                segment["emotion"] = emotion_data["emotion"]
                segment["intensity"] = f"{emotion_data['intensity']:.4f}"
                print(f"  -> Detected Emotion: {segment['emotion']} (Intensity: {segment['intensity']})")
            else:
                print("  -> Emotion analysis failed for this segment.")

        print("\n--- Modified Transcript with Emotion Data ---")
        import json
        print(json.dumps(mock_transcript, indent=2))
