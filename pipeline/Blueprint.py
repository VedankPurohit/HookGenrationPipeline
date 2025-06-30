import os
import shutil
import logging
from typing import List, Dict, Any, Tuple, Optional

from .TranscriptParser import flatten_whisperx_transcript, find_phrase_in_transcription
from .AudioAnalysis import find_best_silence_boundary

logger = logging.getLogger(f"pipeline.{__name__}")

def create_blueprint_from_llm(
    llm_clips: List[Dict[str, Any]],
    full_transcript: Dict[str, Any],
    config: Dict[str, Any],
    video_path: str
) -> List[Dict[str, Any]]:
    """
    Generates a precise production blueprint (shot list) from LLM-suggested clip ideas.

    This function refines the start and end times of each clip by:
    1. Finding the exact phrase specified by the LLM within the full transcript.
    2. Optionally snapping these times to nearby silence boundaries for smoother transitions.

    Args:
        llm_clips (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
                                          represents a clip idea from the LLM. Each dict
                                          is expected to have a 'text_include' key and
                                          optionally a 'why_this_clip' key.
        full_transcript (Dict[str, Any]): The full, detailed transcript object, typically
                                         from a service like Deepgram, in a WhisperX-compatible format.
        config (Dict[str, Any]): The main configuration dictionary for the pipeline,
                                 containing settings for output directories and silence
                                 snapping parameters.
        video_path (str): The absolute path to the source video file, used for audio analysis.

    Returns:
        List[Dict[str, Any]]: A list of finalized clip data dictionaries. Each dictionary
                              contains precise 'original_start' and 'original_end' timestamps,
                              along with the 'text_include' and 'why_this_clip' (if provided).
                              Returns an empty list if no valid clips can be generated.
    """
    
    logger.info("🎬 Refining LLM Hook with Precise Timestamps...")

    # Flatten the transcript once for efficient phrase searching
    flat_transcript: List[Dict[str, Any]] = flatten_whisperx_transcript(full_transcript)
    logger.debug(f"Flattened transcript contains {len(flat_transcript)} words.")

    final_blueprint: List[Dict[str, Any]] = []
    temp_dir_for_analysis: str = os.path.join(config["BASE_OUTPUT_DIR"], "_temp_snap_analysis")
    os.makedirs(temp_dir_for_analysis, exist_ok=True)

    for i, llm_clip in enumerate(llm_clips):
        text_to_find: str = llm_clip.get('text_include', '')
        if not text_to_find:
            logger.warning(f"LLM clip {i+1} has no 'text_include' specified. Skipping this clip.")
            continue

        logger.info(f"⚙️ Refining LLM clip {i+1}: '{text_to_find[:70]}...'")

        # Find the core start and end times of the phrase in the transcript
        core_start: Optional[float]
        core_end: Optional[float]
        core_start, core_end = find_phrase_in_transcription(flat_transcript, text_to_find)

        if core_start is None or core_end is None:
            logger.warning(f"❌ Could not find phrase: '{text_to_find}' in transcript. This clip will be SKIPPED.")
            continue

        final_start: float = core_start
        final_end: float = core_end

        # Apply silence snapping if enabled in the configuration
        if config.get("ENABLE_SILENCE_SNAPPING", False):
            logger.debug(f"🔇 Snapping enabled. Finding best pauses around [{core_start:.2f}s - {core_end:.2f}s]...")
            
            snap_config: Dict[str, Any] = config["SILENCE_SNAP_CONFIG"]
            
            # Snap the start time to a silence boundary before the core start
            final_start = find_best_silence_boundary(video_path, core_start, 'before', snap_config, temp_dir_for_analysis)
            
            # Snap the end time to a silence boundary after the core end
            final_end = find_best_silence_boundary(video_path, core_end, 'after', snap_config, temp_dir_for_analysis)
            
            logger.debug(f"✨ Snapped timestamps: [{final_start:.2f}s - {final_end:.2f}s]")

        # Construct the clip data dictionary
        clip_data: Dict[str, Any] = {
            "why_this_clip": llm_clip.get('why_this_clip', ''), # Optional explanation from LLM
            "original_start": final_start,
            "original_end": final_end,
            "text_include": text_to_find,
        }
        final_blueprint.append(clip_data)

    # Clean up temporary directory used for silence analysis
    if os.path.exists(temp_dir_for_analysis):
        shutil.rmtree(temp_dir_for_analysis)
    
    logger.info(f"✅ Blueprint refinement complete. Final shot list has {len(final_blueprint)} clips.")
    return final_blueprint