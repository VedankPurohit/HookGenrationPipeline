import os
import shutil
import cv2
import numpy as np
import ffmpeg
import logging
from typing import List, Dict, Optional, Any, Tuple

logger = logging.getLogger(f"pipeline.{__name__}")

def get_speaker_bbox_for_frame_cropping(active_speaker_id: Optional[int], faces_in_frame_list: List[Dict[str, Any]]) -> Optional[List[float]]:
    """
    Retrieves the bounding box coordinates for a specific active speaker within a list of detected faces in a single frame.

    Args:
        active_speaker_id (Optional[int]): The unique ID of the active speaker to find. If None, the function returns None.
        faces_in_frame_list (List[Dict[str, Any]]): A list of dictionaries, where each dictionary represents a detected
                                                     face in the current frame and contains at least an 'id' and 'bbox' key.

    Returns:
        Optional[List[float]]: A list of four floats [x1, y1, x2, y2] representing the bounding box
                               of the active speaker, or None if the active speaker is not found or their ID is None.
    """
    if active_speaker_id is None:
        return None
    for face_data in faces_in_frame_list:
        if face_data.get('id') == active_speaker_id:
            return face_data.get('bbox')
    return None

def create_speaker_focused_cropped_clip(
    video_frames_list: List[np.ndarray],
    all_faces_with_ids_per_frame_list: List[List[Dict[str, Any]]],
    final_active_speaker_ids_list: List[Optional[int]],
    video_fps: float,
    segment_audio_file_path: str,
    output_dir_for_segment: str,
    cropped_no_audio_filename: str = "cropped_no_audio.mp4",
    cropped_with_audio_filename: str = "cropped_with_audio.mp4",
    target_aspect_w: int = 9,
    target_aspect_h: int = 16,
    smoothing_alpha: float = 0.15
) -> Optional[str]:
    """
    Generates a vertically cropped video that dynamically follows the active speaker.

    This function performs the following steps:
    1. Calculates the ideal horizontal center for the crop based on the active speaker's face position.
    2. Applies a smoothing filter to the calculated centers to ensure fluid camera movement.
    3. Processes each video frame, cropping it to the target aspect ratio and writing it to an FFmpeg pipe.
    4. Merges the cropped video stream (from the pipe) with the original audio segment to create the final output.

    Args:
        video_frames_list (List[np.ndarray]): A list of video frames as NumPy arrays (H, W, 3 BGR).
        all_faces_with_ids_per_frame_list (List[List[Dict[str, Any]]]): A list where each element corresponds
                                                                       to a frame and contains a list of detected
                                                                       faces with their assigned tracking IDs.
        final_active_speaker_ids_list (List[Optional[int]]): A list where each element is the ID of the active
                                                             speaker for that specific frame, or None if no speaker is active.
        video_fps (float): The frames per second of the input video.
        segment_audio_file_path (str): The absolute path to the audio file corresponding to the video segment.
        output_dir_for_segment (str): The directory where the output cropped video file will be saved.
        cropped_no_audio_filename (str): The filename for the intermediate cropped video (video-only).
        cropped_with_audio_filename (str): The filename for the final cropped video with audio.
        target_aspect_w (int): The width component of the desired output aspect ratio (e.g., 9 for 9:16).
        target_aspect_h (int): The height component of the desired output aspect ratio (e.g., 16 for 9:16).
        smoothing_alpha (float): The alpha value for exponential moving average smoothing of camera movement.
                                 A lower value results in smoother, more delayed movement.

    Returns:
        Optional[str]: The absolute path to the final cropped video file with audio if successful,
                       or None if a critical error occurs during processing.
    """
    if not video_frames_list:
        logger.error("No video frames provided for cropping. Aborting.")
        return None

    orig_H, orig_W = video_frames_list[0].shape[:2]
    num_frames = len(video_frames_list)

    # Calculate target crop dimensions
    crop_h_output = orig_H
    crop_w_output = int(crop_h_output * (target_aspect_w / target_aspect_h))
    if crop_w_output % 2 != 0:
        crop_w_output += 1 # Ensure width is an even number for codec compatibility

    if crop_w_output > orig_W:
        crop_w_output = orig_W if orig_W % 2 == 0 else orig_W - 1 # Ensure crop width does not exceed original and is even
    
    logger.info(f"Cropping Dimensions: Original: {orig_W}x{orig_H}. Target: {crop_w_output}x{crop_h_output}")

    ideal_centers_x = np.zeros(num_frames)
    last_known_good_target_center_x = orig_W / 2.0 # Initialize with frame center

    # TODO: The fallback logic when a speaker is not found could be more sophisticated.
    # Instead of just picking the closest or largest face, it could analyze the past few frames
    # to predict the speaker's likely position, leading to smoother camera work.
    # Determine ideal horizontal center for each frame
    for i in range(num_frames):
        current_target_center_x: Optional[float] = None
        active_id: Optional[int] = final_active_speaker_ids_list[i]
        faces_in_frame: List[Dict[str, Any]] = all_faces_with_ids_per_frame_list[i]
        speaker_bbox: Optional[List[float]] = get_speaker_bbox_for_frame_cropping(active_id, faces_in_frame)

        if speaker_bbox:
            current_target_center_x = (speaker_bbox[0] + speaker_bbox[2]) / 2.0
        else:
            # Fallback if active speaker not found in frame
            if faces_in_frame:
                # If other faces exist, pick the one closest to the frame center or largest
                best_fallback_center_x: Optional[float] = None
                min_dist: float = float('inf')
                for face_data in faces_in_frame:
                    if face_data.get('bbox'):
                        fx1, _, fx2, _ = face_data['bbox']
                        f_center_x = (fx1 + fx2) / 2.0
                        dist_to_frame_center = abs(f_center_x - (orig_W / 2.0))
                        if dist_to_frame_center < min_dist:
                            min_dist = dist_to_frame_center
                            best_fallback_center_x = f_center_x
                current_target_center_x = best_fallback_center_x if best_fallback_center_x is not None else last_known_good_target_center_x
            else:
                # If no faces at all, maintain last known good center or frame center
                current_target_center_x = last_known_good_target_center_x
        
        ideal_centers_x[i] = current_target_center_x
        last_known_good_target_center_x = current_target_center_x # Update for next iteration

    # Apply exponential moving average for smooth camera movement
    smooth_centers_x = np.zeros(num_frames)
    if num_frames > 0:
        smooth_centers_x[0] = ideal_centers_x[0]
        for i in range(1, num_frames):
            smooth_centers_x[i] = (smoothing_alpha * ideal_centers_x[i] + (1 - smoothing_alpha) * smooth_centers_x[i-1])

    final_video_with_audio_path: str = os.path.join(output_dir_for_segment, cropped_with_audio_filename)
    video_only_output_path: str = os.path.join(output_dir_for_segment, cropped_no_audio_filename) # Define this path

    # Setup FFmpeg pipe for video input
    # This allows us to feed frames directly from memory to FFmpeg
    video_input_pipe = (
        ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{crop_w_output}x{crop_h_output}', r=video_fps)
        .output(video_only_output_path, vcodec='libx264', pix_fmt='yuv420p', preset='ultrafast', crf=23)
        .overwrite_output()
        .run_async(pipe_stdin=True, quiet=True)
    )

    logger.info(f"Processing {num_frames} frames for cropping and writing to pipe...")
    for i in range(num_frames):
        original_frame: np.ndarray = video_frames_list[i]
        
        # Calculate crop coordinates
        crop_x1: float = np.clip(smooth_centers_x[i] - (crop_w_output / 2.0), 0, orig_W - crop_w_output)
        crop_x1_int: int = int(round(crop_x1))
        crop_x2_int: int = crop_x1_int + crop_w_output
        
        # Perform the crop
        cropped_slice: np.ndarray = original_frame[0:crop_h_output, crop_x1_int:crop_x2_int]
        
        # Resize if the cropped slice dimensions don't match exactly (due to rounding or edge cases)
        if cropped_slice.shape[1] != crop_w_output or cropped_slice.shape[0] != crop_h_output:
            cropped_slice = cv2.resize(cropped_slice, (crop_w_output, crop_h_output))
        
        # Write the processed frame to the FFmpeg pipe
        video_input_pipe.stdin.write(cropped_slice.tobytes())
    
    video_input_pipe.stdin.close()
    video_input_pipe.wait()

    logger.info(f"✅ Cropped video (video-only) saved to {video_only_output_path}.")

    # Merge the video-only output with the audio segment
    if not os.path.exists(segment_audio_file_path) or os.path.getsize(segment_audio_file_path) == 0:
        logger.error(f"Audio file '{segment_audio_file_path}' not found or is empty. Cannot merge audio. Returning video-only path.")
        return video_only_output_path # Return video-only path as fallback
    
    logger.info(f"🔊 Adding audio from '{segment_audio_file_path}' to create '{final_video_with_audio_path}'")
    try:
        input_video_stream = ffmpeg.input(video_only_output_path)
        input_audio_stream = ffmpeg.input(segment_audio_file_path)
        
        (ffmpeg.output(input_video_stream.video, input_audio_stream.audio, final_video_with_audio_path,
                       vcodec='copy', acodec='aac', strict='experimental')
         .overwrite_output().run(quiet=True, capture_stdout=True, capture_stderr=True))
        
        logger.info(f"✅ Final cropped clip with audio saved: {final_video_with_audio_path}")
        return final_video_with_audio_path
    except ffmpeg.Error as e:
        stderr_output = e.stderr.decode('utf8', errors='ignore').strip().replace('\n', '\n     ')
        logger.error(f"❌ FFmpeg error merging audio: {stderr_output}")
        return video_only_output_path # Fallback to video-only if merge fails
    except Exception as e_gen:
        logger.error(f"Unexpected error merging audio: {e_gen}", exc_info=True)
        return video_only_output_path # Fallback to video-only on general error

# TODO: Implement subtitle generation and burn-in.
# 1. Create a function `generate_srt_from_transcript` that takes the `flat_transcript`,
#    a clip's start/end times, and generates a standard .srt file.
# 2. Create a function `burn_in_subtitles` that uses FFmpeg's `subtitles` filter
#    to burn the .srt file into the video, with customizable styles from `config.py`.
# 3. Integrate these steps into the main processing loop in `main.py` after filler word removal.

def refine_clip_by_removing_fillers(
    input_clip_path: str,
    cut_list: List[Dict[str, Any]],
    output_clip_path: str,
    config: Dict[str, Any]
) -> Optional[str]:
    """
    Removes filler words from a video clip by creating jump cuts with a "punch-in" zoom effect.

    This function constructs a complex FFmpeg filter graph to:
    1. Segment the input video, excluding the specified filler word intervals.
    2. Apply a slight zoom (punch-in) to alternating kept segments to visually hide the cuts.
    3. Concatenate the processed video and audio segments into a single output file.

    Args:
        input_clip_path (str): The absolute path to the input video clip to be refined.
        cut_list (List[Dict[str, Any]]): A list of dictionaries, where each dictionary specifies
                                         the 'start' and 'end' time (relative to the input clip)
                                         of a filler word segment to be removed.
        output_clip_path (str): The absolute path where the refined video clip will be saved.
        config (Dict[str, Any]): The configuration dictionary, used to retrieve the
                                 'PUNCH_IN_ZOOM_FACTOR' setting.

    Returns:
        Optional[str]: The absolute path to the refined video clip if successful, otherwise None.
    """
    if not cut_list:
        logger.info("No filler words found in this segment. Skipping refinement and copying original clip.")
        try:
            shutil.copy(input_clip_path, output_clip_path)
            return output_clip_path
        except Exception as e:
            logger.error(f"Error copying original clip to output path {output_clip_path}: {e}", exc_info=True)
            return None

    logger.info(f"Building FFmpeg command to remove {len(cut_list)} filler segment(s) from {os.path.basename(input_clip_path)}...")

    try:
        probe = ffmpeg.probe(input_clip_path)
        duration = float(probe['format']['duration'])
        video_info = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
        if not video_info:
            raise ValueError(f"No video stream found in {input_clip_path}")
        width = int(video_info['width'])
        height = int(video_info['height'])
    except Exception as e:
        logger.error(f"Could not probe clip properties for '{input_clip_path}'. Error: {e}", exc_info=True)
        return None

    sorted_cuts = sorted(cut_list, key=lambda x: x['start'])
    segments_to_keep: List[Dict[str, float]] = []
    last_end_time: float = 0.0

    # Determine segments to keep by inverting the cut list
    for cut in sorted_cuts:
        if cut['start'] > last_end_time:
            segments_to_keep.append({'start': last_end_time, 'end': cut['start']})
        last_end_time = max(last_end_time, cut['end']) # Ensure last_end_time always progresses
    if last_end_time < duration:
        segments_to_keep.append({'start': last_end_time, 'end': duration})

    if not segments_to_keep:
        logger.warning("Cut list resulted in an empty video after processing. Aborting refinement.")
        return None

    input_stream = ffmpeg.input(input_clip_path)
    video_streams = []
    audio_streams = []
    zoom_factor: float = config.get("PUNCH_IN_ZOOM_FACTOR", 1.08)

    for i, segment in enumerate(segments_to_keep):
        # Trim video and audio segments
        v = input_stream.video.trim(start=segment['start'], end=segment['end']).setpts('PTS-STARTPTS')
        a = input_stream.audio.filter('atrim', start=segment['start'], end=segment['end']).filter('asetpts', 'PTS-STARTPTS')

        # Apply zoom effect to alternating segments
        if i % 2 != 0: # Apply zoom to every second segment
            # Scale up and then crop back to original dimensions to create punch-in effect
            v = v.filter('scale', f'iw*{zoom_factor}', '-1') \
                 .filter('crop', f'iw/{zoom_factor}', f'ih/{zoom_factor}')

        # Ensure output resolution matches original and set pixel aspect ratio
        v = v.filter('scale', width, height).filter('setsar', 1)

        video_streams.append(v)
        audio_streams.append(a)

    # Concatenate all processed video and audio streams
    concatenated_video = ffmpeg.concat(*video_streams, v=1, a=0)
    concatenated_audio = ffmpeg.concat(*audio_streams, v=0, a=1)

    try:
        (ffmpeg.output(
            concatenated_video,
            concatenated_audio,
            output_clip_path,
            vcodec='libx264',
            acodec='aac',
            strict='experimental'
        )
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True, quiet=True))
        
        logger.info(f"✅ Successfully refined clip and saved to '{os.path.basename(output_clip_path)}'")
        return output_clip_path
    except ffmpeg.Error as e:
        stderr = e.stderr.decode(errors='ignore').strip().replace('\n', '\n     ')
        logger.error(f"❌ FFMPEG ERROR during clip refinement: {stderr}")
        return None
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred during clip refinement: {e}", exc_info=True)
        return None

def find_filler_words_in_segment(
    segment_words: List[Dict[str, Any]],
    segment_start_time: float,
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Identifies and extracts filler words from a list of word-level transcript data.

    This function iterates through the provided word list and checks each word against
    a predefined list of filler words from the configuration. For each identified
    filler word, it calculates its start and end time relative to the beginning of
    the current video segment.

    Args:
        segment_words (List[Dict[str, Any]]): A list of word dictionaries from the transcript,
                                             each expected to have 'word', 'start', and 'end' keys.
        segment_start_time (float): The absolute start time (in seconds) of the current video segment.
                                    This is used to convert absolute word timestamps to relative timestamps.
        config (Dict[str, Any]): The configuration dictionary, which must contain the 'FILLER_WORD_LIST'.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents a filler word
                              to be cut. Each dictionary contains 'start' (relative to segment),
                              'end' (relative to segment), and 'text' (the filler word itself).
    """

    cut_list: List[Dict[str, Any]] = []
    filler_word_set: set = set(config.get("FILLER_WORD_LIST", []))
    if not filler_word_set:
        logger.debug("FILLER_WORD_LIST is empty in config. No filler words will be identified.")
        return []

    for word_info in segment_words:
        word: str = word_info.get('word', '').lower().strip(".,?! ") # Strip spaces too
        if word in filler_word_set:
            relative_start: float = word_info['start'] - segment_start_time
            relative_end: float = word_info['end'] - segment_start_time
            cut_list.append({
                "start": relative_start,
                "end": relative_end,
                "text": word_info['word']
            })

    if cut_list:
        filler_texts = [c['text'] for c in cut_list]
        logger.info(f"Found {len(cut_list)} filler word(s) to remove: {filler_texts}")
    return cut_list
