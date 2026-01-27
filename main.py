# main.py
"""
Main entry point for the video production pipeline.

This script orchestrates the production stage, assuming the preparation stage
(running prepare.py) has already been completed.

Workflow:
1.  Parses command-line arguments for project and run configuration.
2.  Sets up a dedicated output directory for the current run.
3.  Initializes a comprehensive logging system.
4.  Loads pre-prepared source assets (video, transcript, creative brief).
5.  Initializes necessary models (e.g., face detection).
6.  Generates a production blueprint by aligning the creative brief with the transcript.
7.  Processes each clip defined in the blueprint through the pipeline.
8.  Assembles the processed clips into a final video.
"""

import os
import time
import shutil
import argparse
import json
import logging
from typing import Dict, Any, List

import pipeline
from pipeline import setup_logging, log_run_summary
from retinaface import RetinaFace
from MoreFeatures.LLM.simpleLlm import generate_clips

def run_production(project_name: str, run_name: str, use_gpu: bool, no_crop: bool, use_custom_clips: str, custom_instructs: str, use_template: str) -> None:
    """
    Executes the main video production pipeline.

    Orchestrates the entire video processing workflow, from loading source assets
    and models to generating a production blueprint, processing individual clips,
    and assembling the final video.

    Args:
        project_name: The name of the project folder in the 'output' directory.
        run_name: A unique name for the current production run.
        use_gpu: If True, attempts to use the GPU for face detection.
        no_crop: If True, disables Active Speaker Detection and vertical cropping.
        use_custom_clips: Path to a pre-existing JSON file for clip definitions.
        custom_instructs: Custom instructions for the 'general' LLM template.
        use_template: The LLM template to use for clip generation.
    """
    # --- STAGE 0: SETUP AND VALIDATION ---
    CONFIG: Dict[str, Any] = pipeline.CONFIG

    # Define key directories
    project_dir: str = os.path.join("output", project_name)
    source_assets_dir: str = os.path.join(project_dir, "source_assets")
    run_dir: str = os.path.join(project_dir, "runs", run_name)
    CONFIG["BASE_OUTPUT_DIR"] = run_dir
    
    CONFIG["ENABLE_ASD_CROP"] = not no_crop
    
    # Ensure a clean run by removing previous outputs
    if os.path.exists(run_dir):
        print(f"-> Found existing run '{run_name}'. Deleting old outputs.")
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)
    
    # Configure run-specific logging
    setup_logging(run_dir)
    logger = logging.getLogger('pipeline.main')
    
    logger.info("--- PRODUCTION STAGE ---")
    logger.info(f"Project: {project_name}")
    logger.info(f"Run:     {run_name}")
    logger.info(f"GPU Enabled: {use_gpu}")
    logger.info(f"ASD Cropping Enabled: {CONFIG['ENABLE_ASD_CROP']}")
    logger.info("------------------------")

    # TODO: Consider creating a dedicated 'Context' or 'State' object to pass to pipeline
    # functions, rather than passing individual objects like `face_detector_model`. This
    # would make the function signatures cleaner and the data flow more explicit.
    
    # --- STAGE 1: LOAD SOURCE ASSETS AND MODELS ---
    logger.info("--- STAGE 1: Loading Source Assets & Models ---")
    try:
        source_video_path = os.path.join(source_assets_dir, "source_video.mp4")
        transcript_path = os.path.join(source_assets_dir, "transcript.json")
        llm_transcript_path = os.path.join(source_assets_dir, "llm_summary.txt")

        if use_custom_clips:
            creative_brief_path = use_custom_clips
            if not os.path.exists(creative_brief_path):
                logger.critical(f"Error: Custom clips file not found at {creative_brief_path}.")
                return
            logger.info(f"Using custom clips from: {creative_brief_path}")
        else:
            generated_clips_path = os.path.join(run_dir, "generated_clips.json")
            template_name = use_template
            if custom_instructs:
                template_name = "general"

            logger.info(f"Generating clips using template: '{template_name}'")
            if template_name == "general":
                if custom_instructs:
                    logger.info(f"With custom instructions: \"{custom_instructs}\"")
                else:
                    logger.info("No custom instructions provided; a random prompt will be selected.")

            generate_clips(
                input_filepath=llm_transcript_path,
                output_filepath=generated_clips_path,
                template_name=template_name,
                custom_instructions=custom_instructs,
            )
            creative_brief_path = generated_clips_path

            if not os.path.exists(creative_brief_path):
                logger.critical(f"Generated clips file not found at {creative_brief_path}. LLM generation may have failed.")
                return
            logger.info(f"Generated clips saved to: {creative_brief_path}")

        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        
        with open(creative_brief_path, 'r', encoding='utf-8') as f:
            creative_brief = json.load(f)

        if not os.path.exists(source_video_path):
            raise FileNotFoundError(f"Source video not found at {source_video_path}. Please run 'prepare.py' first.")

        # Load face detection model if cropping is enabled
        face_detector_model = None
        if not no_crop:
            logger.info("Initializing face detection model (RetinaFace)...")
            if not use_gpu:
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                logger.info("Running face detection on CPU.")
            else:
                logger.info("Attempting to run face detection on GPU.")
            face_detector_model = RetinaFace.build_model()
            logger.info("✅ Face detection model loaded successfully.")

        logger.info("✅ All source assets and models loaded successfully.")
        log_run_summary(run_dir, CONFIG, creative_brief)

    except FileNotFoundError as e:
        logger.critical(f"A required asset is missing: {e}. Ensure 'prepare.py' was run for project '{project_name}'.")
        return
    except json.JSONDecodeError as e:
        logger.critical(f"Error parsing JSON file (transcript or creative brief): {e}. Please check file integrity.")
        return
    except Exception as e:
        logger.critical(f"Failed to load assets or models due to an unexpected error: {e}", exc_info=True)
        return

    # --- STAGE 2: CREATE PRODUCTION BLUEPRINT ---
    logger.info("--- STAGE 2: Creating Production Blueprint ---")
    final_blueprint = pipeline.create_blueprint_from_llm(creative_brief, transcript_data, CONFIG, source_video_path)
    if not final_blueprint:
        logger.critical("Blueprint generation failed. No clips to process. Aborting.")
        return

    # --- STAGE 3: PER-CLIP PROCESSING LOOP ---
    processed_clip_paths = []
    flat_transcript = pipeline.flatten_whisperx_transcript(transcript_data)

    for i, clip_info in enumerate(final_blueprint):
        start_ts = clip_info['original_start']
        end_ts = clip_info['original_end']
        logger.info(f"--- Processing Clip {i+1}/{len(final_blueprint)}: [{start_ts:.2f}s - {end_ts:.2f}s] ---")

        segment_dir = os.path.join(run_dir, f"segment_{i:02d}")
        os.makedirs(segment_dir, exist_ok=True)

        # Extract raw video and audio for the current clip
        seg_vid_path = os.path.join(segment_dir, CONFIG["SEGMENT_VIDEO_FILENAME"])
        seg_aud_path = os.path.join(segment_dir, CONFIG["SEGMENT_AUDIO_FILENAME"])
        if not pipeline.extract_segment_reencode(source_video_path, start_ts, end_ts, seg_vid_path, seg_aud_path, CONFIG):
            logger.warning(f"Failed to extract segment for clip {i+1}. Skipping.")
            continue

        # Perform Active Speaker Detection (ASD) and smart vertical cropping
        clip_to_refine = pipeline.perform_asd_and_crop(seg_vid_path, seg_aud_path, segment_dir, CONFIG, detector_model_instance=face_detector_model)

        # Refine the clip by removing filler words
        final_clip_path = clip_to_refine
        if CONFIG["ENABLE_FILLER_WORD_REMOVAL"]:
            filler_words_cut_list = pipeline.find_filler_words_in_segment(flat_transcript, start_ts, CONFIG)
            
            refined_output_path = os.path.join(segment_dir, 'refined_clip.mp4')
            refined_clip_result = pipeline.refine_clip_by_removing_fillers(clip_to_refine, filler_words_cut_list, refined_output_path, CONFIG)
            
            if refined_clip_result:
                final_clip_path = refined_output_path
            else:
                logger.warning(f"Filler word removal failed for clip {i+1}. Using unrefined clip.")
        
        # TODO: Integrate features from the MoreFeatures directory.
        # - EmotionAnalysis: Analyze `seg_aud_path` to get emotion data for the clip.
        # - B_Roll_Integration: Use transcript keywords from `clip_info` to find and insert B-roll.
        # - Sound_Effect_Engine: Add transition sounds between clips during concatenation.

        if final_clip_path and os.path.exists(final_clip_path):
            logger.info(f"✅ Finalized clip for concatenation: {os.path.basename(final_clip_path)}")
            processed_clip_paths.append(final_clip_path)
        else:
            logger.error(f"Processed clip path is invalid or does not exist for clip {i+1}. Skipping.")

    # --- STAGE 4: FINAL ASSEMBLY ---
    if not processed_clip_paths:
        logger.error("No clips were successfully processed. Aborting final assembly.")
        return

    logger.info("--- STAGE 4: Assembling Final Video ---")
    final_video_path = os.path.join(run_dir, CONFIG["FINAL_CONCATENATED_VIDEO_FILENAME"])
    final_video = pipeline.concatenate_video_clips(processed_clip_paths, final_video_path)
    
    if final_video:
        logger.info("🎉🎉🎉 PIPELINE COMPLETE! 🎉🎉🎉")
        logger.info(f"Final video available at: {final_video}")
    else:
        logger.error("❌ Final video assembly failed. Check logs for details.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the video production pipeline.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--project-name",
        type=str,
        required=True,
        help="The name of the project folder in the 'output' directory."
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=f"run_{int(time.time())}",
        help="A unique name for this production run (e.g., 'hook_v1_fast_cuts'). Defaults to a timestamp."
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable GPU acceleration for face detection."
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Disable vertical cropping and Active Speaker Detection (ASD)."
    )
    parser.add_argument(
        "--use-custom-clips",
        type=str,
        help="Path to a custom JSON file with clip definitions, skipping LLM generation."
    )
    parser.add_argument(
        "--custom-instructs",
        type=str,
        help="Custom instructions for the 'general' clip generation template."
    )
    parser.add_argument(
        "--use-template",
        type=str,
        choices=['best', 'rapidfire', 'general', 'emotional', 'keytakeaway', 'controversial', 'shorts'],
        default='rapidfire',
        help="Template for LLM-based clip generation. Defaults to 'rapidfire'."
    )
    args = parser.parse_args()

    start_time = time.time()
    run_production(
        project_name=args.project_name,
        run_name=args.run_name,
        use_gpu=args.use_gpu,
        no_crop=args.no_crop,
        use_custom_clips=args.use_custom_clips,
        custom_instructs=args.custom_instructs,
        use_template=args.use_template
    )
    end_time = time.time()
    
    logging.getLogger('pipeline').info(f"Total execution time: {end_time - start_time:.2f} seconds.")
