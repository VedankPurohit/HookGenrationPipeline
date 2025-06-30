"""
This module contains high-level, multi-step processing functions that orchestrate
calls to lower-level modules like analysis and editing.
"""
import os
import gc
import torch
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional

# Import all the low-level functions this high-level step will need
from .VisionAnalysis import (
    detect_faces_in_video_segment, apply_face_tracking, get_mouth_activity_per_track,
    determine_active_speaker_ids, enforce_single_speaker_heuristics
)
from .AudioAnalysis import get_speech_segments_from_vad
from .EditingEffects import create_speaker_focused_cropped_clip

logger = logging.getLogger(f"pipeline.{__name__}")

def perform_asd_and_crop(
    seg_vid_path: str,
    seg_aud_path: str,
    segment_dir: str,
    config: Dict[str, Any],
    detector_model_instance: Any
) -> str:
    """
    Orchestrates the complete Active Speaker Detection (ASD) and smart cropping workflow.

    This function takes a raw video segment, analyzes it to identify the active speaker
    in each frame, and then generates a vertically cropped video that dynamically follows
    the speaker. It integrates face detection, tracking, voice activity detection, and
    mouth activity analysis to achieve this.
    
    If ASD and cropping are disabled in the configuration, or if any critical step fails,
    the function will gracefully fall back and return the path to the original, uncropped
    video segment, ensuring the pipeline can continue.

    Args:
        seg_vid_path (str): Absolute path to the input video segment file.
        seg_aud_path (str): Absolute path to the synchronized audio file for the segment.
        segment_dir (str): The absolute path to the directory where all intermediate and
                           final output files for this specific segment should be saved.
        config (Dict[str, Any]): The main configuration dictionary containing various
                                 parameters for ASD, cropping, and memory management.
        detector_model_instance (Any): An initialized instance of the face detection model
                                       (e.g., RetinaFace model). This is passed to avoid
                                       reloading the model for each segment.

    Returns:
        str: The absolute file path of the resulting video clip. This will be the path
             to the cropped video if the process is successful and enabled, or the path
             to the original segment video if ASD/cropping is disabled or encounters an error.
    """
    if not config.get("ENABLE_ASD_CROP", False):
        logger.info("ASD and Cropping is disabled in config. Skipping this step.")
        return seg_vid_path

    logger.info("🚀 Starting Active Speaker Detection (ASD) & Cropping process...")
    try:
        # 1. Analyze video frames and detect faces
        # Returns: list of frames, list of raw face detections per frame, video FPS
        video_frames: List[np.ndarray]
        faces_raw: List[List[Dict[str, Any]]]
        vid_fps: float
        video_frames, faces_raw, vid_fps = detect_faces_in_video_segment(
            seg_vid_path, 
            low_vram_mode=config.get("LOW_VRAM_MODE", False),
            detector_model_instance=detector_model_instance,
            config=config
        )
        if not video_frames or not faces_raw:
            logger.warning("⚠️ No video frames or face detections found. Aborting ASD/Crop for this segment.")
            return seg_vid_path

        # 2. Perform face tracking to assign consistent IDs to detected faces
        faces_tracked: List[List[Dict[str, Any]]] = apply_face_tracking(faces_raw, vid_fps)
        if not faces_tracked:
            logger.warning("⚠️ Face tracking failed or found no persistent faces. Aborting ASD/Crop.")
            return seg_vid_path

        # 3. Perform Voice Activity Detection (VAD) on the audio segment
        
        speech_segments: List[Tuple[int, int]] = get_speech_segments_from_vad(
            seg_aud_path, 
            config["VAD_SAMPLE_RATE"], 
            config["VAD_FRAME_DURATION_MS"], 
            config["VAD_AGGRESSIVENESS"], 
            0.3, # silence_frames_thresh_factor
            0.1  # speech_frames_thresh_factor
        )
        if not speech_segments:
            logger.warning("⚠️ No speech segments detected in audio. Aborting ASD/Crop.")
            return seg_vid_path

        # 4. Analyze mouth activity for each tracked face
        mouth_activity: Dict[int, List[Optional[float]]] = get_mouth_activity_per_track(faces_tracked)
        if not mouth_activity:
            logger.warning("⚠️ No mouth activity detected for tracked faces. Aborting ASD/Crop.")
            return seg_vid_path

        # 5. Determine the active speaker by correlating VAD and mouth activity
        speaker_ids: List[Optional[int]] = determine_active_speaker_ids(
            speech_segments, 
            mouth_activity, 
            len(video_frames), 
            vid_fps, 
            config["VAD_FRAME_DURATION_MS"]
        )
        
        # 6. Apply heuristics to enforce a single, consistent speaker if configured
        if config.get("ENABLE_FORCED_SINGLE_SPEAKER_MODE", False):
            logger.debug("Applying forced single speaker mode heuristics.")
            speaker_ids = enforce_single_speaker_heuristics(
                speaker_ids, 
                faces_tracked, 
                video_frames[0].shape[1], # Pass original frame width
                vid_fps
            )

        # 7. Create the speaker-focused cropped video based on the analysis
        cropped_clip_path: Optional[str] = create_speaker_focused_cropped_clip(
            video_frames, 
            faces_tracked, 
            speaker_ids, 
            vid_fps, 
            seg_aud_path, 
            segment_dir,
            target_aspect_w=config.get("CROP_ASPECT_RATIO_W", 9),
            target_aspect_h=config.get("CROP_ASPECT_RATIO_H", 16),
            smoothing_alpha=config.get("CROP_SMOOTHING_ALPHA", 0.15)
        )
        
        # 8. Return the new path if successful, otherwise fall back to original
        if cropped_clip_path and os.path.exists(cropped_clip_path):
            logger.info(f"✅ ASD & Cropping successful. Output: {os.path.basename(cropped_clip_path)}")
            return cropped_clip_path
        else:
            logger.warning("⚠️ Cropping failed or returned an invalid path. Falling back to original segment.")
            return seg_vid_path

    except Exception as e:
        logger.error(f"❌ An unexpected error occurred during ASD & Cropping: {e}", exc_info=True)
        return seg_vid_path # Always fallback on any error to prevent pipeline halt
    finally:
        # 9. Clean up memory, especially important in low VRAM environments
        if config.get("LOW_VRAM_MODE", False):
            logger.debug("Attempting to clear memory in low VRAM mode.")
            # Explicitly delete large objects if they exist in the local scope
            for var in ['video_frames', 'faces_raw', 'faces_tracked', 'mouth_activity', 'speaker_ids']:
                if var in locals():
                    del locals()[var]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.debug("Memory cleanup attempted.")