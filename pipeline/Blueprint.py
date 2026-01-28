import os
import shutil
import logging
from typing import List, Dict, Any, Tuple, Optional

from .TranscriptParser import flatten_whisperx_transcript, find_phrase_in_transcription
from .AudioAnalysis import find_best_silence_boundary

logger = logging.getLogger(f"pipeline.{__name__}")

def detect_overlapping_clips(clips: List[Dict[str, Any]], tolerance: float = 0.1) -> List[List[int]]:
    """
    Detects overlapping clips based on their time ranges.

    Args:
        clips: List of clip dictionaries with 'original_start' and 'original_end' keys
        tolerance: Time tolerance in seconds for considering clips as overlapping (default 0.1s)

    Returns:
        List of lists, where each inner list contains indices of overlapping clips
    """
    overlaps = []
    for i in range(len(clips)):
        for j in range(i + 1, len(clips)):
            clip_a = clips[i]
            clip_b = clips[j]

            # Check if clips overlap (with tolerance)
            if (clip_a['original_end'] > clip_b['original_start'] + tolerance and
                clip_b['original_end'] > clip_a['original_start'] + tolerance):
                overlaps.append([i, j])

    return overlaps

def merge_overlapping_clips(clips: List[Dict[str, Any]], overlap_groups: List[List[int]]) -> List[Dict[str, Any]]:
    """
    Merges overlapping clips by combining consecutive overlapping segments into single clips.

    This function iteratively merges overlapping clips until no overlaps remain.
    Clips are sorted by start time, and consecutive clips that overlap are merged
    into a single continuous clip.

    Args:
        clips: List of clip dictionaries with 'original_start' and 'original_end' keys
        overlap_groups: Initial overlap groups (used for logging, actual merging re-detects)

    Returns:
        List of merged clips with all overlaps resolved
    """
    if not clips:
        return clips

    # Sort clips by start time for sequential processing
    sorted_clips = sorted(clips, key=lambda c: c['original_start'])
    
    logger.info(f"🔄 Processing overlaps among {len(sorted_clips)} clips...")

    # Iteratively merge until no overlaps remain
    iteration = 0
    max_iterations = 50  # Safety limit
    
    while iteration < max_iterations:
        iteration += 1
        merged_this_round = False
        new_clips = []
        i = 0
        
        while i < len(sorted_clips):
            current_clip = sorted_clips[i].copy()
            
            # Look ahead and merge all consecutive overlapping clips
            while i + 1 < len(sorted_clips):
                next_clip = sorted_clips[i + 1]
                
                # Check if next clip overlaps with current (starts before current ends)
                if next_clip['original_start'] < current_clip['original_end']:
                    # Merge: extend current clip to cover next clip
                    original_end = current_clip['original_end']
                    current_clip['original_end'] = max(current_clip['original_end'], next_clip['original_end'])
                    
                    # Combine text
                    current_text = current_clip.get('text_include', '')
                    next_text = next_clip.get('text_include', '')
                    if next_text and next_text not in current_text:
                        current_clip['text_include'] = f"{current_text} | {next_text}"
                    
                    # Combine explanations
                    current_why = current_clip.get('why_this_clip', '')
                    next_why = next_clip.get('why_this_clip', '')
                    if next_why and next_why not in current_why:
                        current_clip['why_this_clip'] = f"{current_why} + {next_why}"
                    
                    logger.debug(f"   Merged clip ending at {original_end:.2f}s with clip starting at {next_clip['original_start']:.2f}s")
                    merged_this_round = True
                    i += 1  # Skip the merged clip
                else:
                    break  # No overlap with next clip
            
            new_clips.append(current_clip)
            i += 1
        
        sorted_clips = new_clips
        
        if not merged_this_round:
            break
    
    clips_removed = len(clips) - len(sorted_clips)
    if clips_removed > 0:
        logger.info(f"✅ Overlap merging complete. Merged {clips_removed} clips into adjacent segments.")
        print(f"✅ Overlap resolution: {clips_removed} clips merged, {len(sorted_clips)} clips remaining")
    else:
        logger.info(f"✅ No additional merges needed after sorting.")
        print(f"✅ Clips sorted and validated, {len(sorted_clips)} clips remaining")

    return sorted_clips

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
    2. If phrase search fails, falls back to the LLM's provided timestamps.
    3. Optionally snapping these times to nearby silence boundaries for smoother transitions.

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
        sentence_context: str = llm_clip.get('sentence_context', '')
        
        if not text_to_find:
            logger.warning(f"LLM clip {i+1} has no 'text_include' specified. Skipping this clip.")
            continue

        # Use sentence_context for matching if available and text_to_find is short
        search_text = text_to_find
        if sentence_context and len(text_to_find.split()) < 5:
            search_text = sentence_context
            logger.debug(f"Using sentence_context for matching (text_include too short: {len(text_to_find.split())} words)")

        logger.info(f"⚙️ Refining LLM clip {i+1}: '{text_to_find[:70]}...'")

        # Find the core start and end times of the phrase in the transcript
        core_start: Optional[float]
        core_end: Optional[float]
                # Get LLM's timestamp hints for proximity-based search
        llm_start = llm_clip.get('original_start')
        llm_end = llm_clip.get('original_end')
        
        core_start, core_end = find_phrase_in_transcription(
            flat_transcript, search_text,
            hint_start=llm_start, hint_end=llm_end
        )

        # If phrase search fails, try to use LLM's provided timestamps as fallback
        if core_start is None or core_end is None:
            llm_start = llm_clip.get('original_start')
            llm_end = llm_clip.get('original_end')
            
            if llm_start is not None and llm_end is not None:
                # Use LLM timestamps with a small buffer for safety
                buffer = 0.15  # 150ms buffer
                core_start = max(0, llm_start - buffer)
                core_end = llm_end + buffer
                logger.warning(f"⚠️ Phrase not found, using LLM timestamps as fallback: [{core_start:.2f}s - {core_end:.2f}s]")
            else:
                logger.warning(f"❌ Could not find phrase and no LLM timestamps available. Skipping clip {i+1}.")
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
            "sentence_context": sentence_context,  # Preserve for debug overlay
        }
        final_blueprint.append(clip_data)

    # --- STAGE 4: DETECT AND RESOLVE OVERLAPPING CLIPS ---
    logger.info(f"🔍 Checking for overlapping clips among {len(final_blueprint)} processed clips...")

    # Detect overlapping clips
    overlap_groups = detect_overlapping_clips(final_blueprint)
    print(f"🔍 Overlap detection: Found {len(overlap_groups)} overlapping clip groups")

    if overlap_groups:
        logger.warning(f"⚠️  Detected {len(overlap_groups)} overlapping clip groups that need merging")
        for idx, group in enumerate(overlap_groups):
            clip_times = [(final_blueprint[i]['original_start'], final_blueprint[i]['original_end'])
                         for i in group]
            logger.info(f"   Group {idx + 1}: Clips {group} with times {clip_times}")
            print(f"   Group {idx + 1}: Clips {[i+1 for i in group]} overlap at times {clip_times}")

        # Merge overlapping clips
        logger.info("🔄 Starting overlap resolution process...")
        final_blueprint = merge_overlapping_clips(final_blueprint, overlap_groups)
    else:
        logger.info("✅ No overlapping clips detected - proceeding with original blueprint")
        print("✅ No overlapping clips found - all clips are properly sequenced")

    # Clean up temporary directory used for silence analysis
    if os.path.exists(temp_dir_for_analysis):
        shutil.rmtree(temp_dir_for_analysis)

    logger.info(f"✅ Blueprint refinement complete. Final shot list has {len(final_blueprint)} clips.")
    print(f"✅ Final blueprint: {len(final_blueprint)} clips ready for processing")
    return final_blueprint