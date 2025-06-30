import os
import cv2
import time
import math
import collections
import numpy as np
import logging
from retinaface import RetinaFace
from typing import List, Dict, Any, Optional, Tuple, OrderedDict

from .utils import calculate_iou

logger = logging.getLogger(f"pipeline.{__name__}")


class SimpleFaceTracker:
    """
    A simple, straightforward face tracker based on Intersection over Union (IoU).

    # TODO: Explore more advanced tracking algorithms. While IoU is robust for this
    # use case, algorithms like SORT or DeepSORT could provide better handling of
    # longer occlusions or more complex scenes if the project requirements expand.

    This tracker assigns a unique ID to each detected face. In subsequent frames,
    it uses the IoU between previously tracked faces and new detections to maintain
    the correct IDs. It can also handle temporary disappearances of faces up to a
    configurable number of frames.

    Attributes:
        next_object_id (int): The next available ID to be assigned to a new face.
        objects (OrderedDict): Stores the most recent data (bbox, landmarks, etc.)
                               of currently tracked faces, keyed by their ID.
        disappeared (OrderedDict): Stores the number of consecutive frames a
                                   tracked face has been missing, keyed by ID.
        max_disappeared (int): The maximum number of frames a face can be missing
                               before it is deregistered.
        iou_threshold (float): The minimum IoU required to match a new detection
                               with an existing tracked face.
    """
    def __init__(self, max_disappeared_frames: int = 10, iou_threshold: float = 0.3):
        """
        Initializes the SimpleFaceTracker.

        Args:
            max_disappeared_frames (int): The number of consecutive frames an object
                                          can be missing before it's deregistered.
            iou_threshold (float): The minimum IoU score to consider two bounding
                                   boxes as a match.
        """
        self.next_object_id = 0
        self.objects: OrderedDict[int, Dict[str, Any]] = collections.OrderedDict()
        self.disappeared: OrderedDict[int, int] = collections.OrderedDict()
        self.max_disappeared = max_disappeared_frames
        self.iou_threshold = iou_threshold

    def register(self, new_face_data: Dict[str, Any]) -> None:
        """
        Registers a new face, assigning it a unique ID.

        Args:
            new_face_data (Dict[str, Any]): A dictionary containing the face's
                                            bounding box and other metadata.
        """
        new_face_data_copy = new_face_data.copy()
        new_face_data_copy['id'] = self.next_object_id
        self.objects[self.next_object_id] = new_face_data_copy
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id: int) -> None:
        """
        Deregisters a face, removing it from tracking.

        Args:
            object_id (int): The ID of the face to remove.
        """
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, current_frame_detections_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Updates the tracker with new face detections from a single frame.

        This method matches new detections to existing tracked objects,
        registers new faces, and deregisters faces that have disappeared.

        Args:
            current_frame_detections_list (List[Dict[str, Any]]): A list of face
                dictionaries detected in the current frame.

        Returns:
            List[Dict[str, Any]]: A list of face dictionaries for the current
                frame, updated with their assigned tracker IDs.
        """
        if len(current_frame_detections_list) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return []

        if len(self.objects) == 0:
            for detection in current_frame_detections_list:
                self.register(detection)
            updated_detections_for_frame = []
            for det in current_frame_detections_list:
                for obj_id, tracked_obj_data in self.objects.items():
                    if np.array_equal(det['bbox'], tracked_obj_data['bbox']):
                        updated_detections_for_frame.append(tracked_obj_data.copy())
                        break
            return updated_detections_for_frame

        object_ids_list = list(self.objects.keys())
        previous_tracked_bboxes = [self.objects[obj_id]['bbox'] for obj_id in object_ids_list]
        current_detection_bboxes = [det['bbox'] for det in current_frame_detections_list]

        if not previous_tracked_bboxes or not current_detection_bboxes:
            iou_matrix = np.array([])
        else:
            iou_matrix = np.array([[calculate_iou(prev_bbox, curr_bbox)
                                    for curr_bbox in current_detection_bboxes]
                                   for prev_bbox in previous_tracked_bboxes])

        used_prev_object_indices, used_current_detection_indices = set(), set()
        if iou_matrix.size > 0:
            rows, cols = iou_matrix.shape
            sorted_iou_indices = sorted([(r, c) for r in range(rows) for c in range(cols)],
                                        key=lambda x: iou_matrix[x[0], x[1]], reverse=True)
            for prev_obj_idx, curr_det_idx in sorted_iou_indices:
                if prev_obj_idx in used_prev_object_indices or curr_det_idx in used_current_detection_indices:
                    continue
                if iou_matrix[prev_obj_idx, curr_det_idx] > self.iou_threshold:
                    object_id_to_update = object_ids_list[prev_obj_idx]
                    updated_detection_data = current_frame_detections_list[curr_det_idx].copy()
                    updated_detection_data['id'] = object_id_to_update
                    self.objects[object_id_to_update] = updated_detection_data
                    self.disappeared[object_id_to_update] = 0
                    used_prev_object_indices.add(prev_obj_idx)
                    used_current_detection_indices.add(curr_det_idx)

        unmatched_prev_indices = set(range(len(object_ids_list))) - used_prev_object_indices
        for prev_idx in unmatched_prev_indices:
            obj_id = object_ids_list[prev_idx]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self.deregister(obj_id)

        unmatched_curr_indices = set(range(len(current_detection_bboxes))) - used_current_detection_indices
        for curr_idx in unmatched_curr_indices:
            self.register(current_frame_detections_list[curr_idx])

        output_detections_for_frame = []
        for original_detection_data in current_frame_detections_list:
            found_corresponding_tracked_object = False
            for tracked_id, tracked_data_with_id in self.objects.items():
                if np.array_equal(original_detection_data['bbox'], tracked_data_with_id['bbox']):
                    output_detections_for_frame.append(tracked_data_with_id.copy())
                    found_corresponding_tracked_object = True
                    break
            if not found_corresponding_tracked_object:
                temp_det = original_detection_data.copy()
                temp_det['id'] = None
                output_detections_for_frame.append(temp_det)
                # print(f"Debug: Original detection (bbox: {temp_det['bbox']}) not found in tracker.objects, added with ID None.")

        return output_detections_for_frame


def detect_faces_in_video_segment(
    video_path: str,
    detection_confidence_threshold: float = 0.7,
    low_vram_mode: bool = False,
    detector_model_instance: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[List[np.ndarray], List[List[Dict[str, Any]]], float]:
    """
    Detects faces in all frames of a video segment using the provided RetinaFace model.

    This function now relies on a pre-initialized model instance being passed in,
    ensuring that the device (CPU/GPU) is controlled by the main entry point.

    Args:
        video_path: The path to the input video file.
        detection_confidence_threshold: The minimum confidence score for a valid face.
        low_vram_mode: Flag to enable memory-saving measures.
        detector_model_instance: A pre-built RetinaFace model instance.
        config: The pipeline configuration dictionary.

    Returns:
        A tuple containing the list of frames, a list of face detections per frame,
        and the video's FPS.
    """
    if not detector_model_instance:
        logger.critical("CRITICAL: Face detector model was not provided to detect_faces_in_video_segment. Aborting.")
        return [], [], 0.0

    if not os.path.exists(video_path):
        logger.error(f"Video file not found at {video_path} for face detection.")
        return [], [], 0.0
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file {video_path} with OpenCV.")
        return [], [], 0.0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"🎞️ Video Properties for Face Detection: FPS={video_fps if video_fps else 'N/A'}, Total Frames={total_frames_video}")

    # Determine the device being used for the log message
    device = "GPU" if os.environ.get('CUDA_VISIBLE_DEVICES', '0') != '-1' else "CPU"
    logger.info(f"🤖 Starting face detection on {device} for {total_frames_video} frames...")

    all_frames_face_data = []
    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)

        try:
            # Use the passed-in model instance directly
            detected_faces_dict = RetinaFace.detect_faces(frame, threshold=detection_confidence_threshold, model=detector_model_instance)
        except Exception as e:
            logger.error(f"Error in RetinaFace.detect_faces on frame {frame_count}: {e}", exc_info=True)
            all_frames_face_data.append([])
            frame_count += 1
            continue
        
        current_frame_faces = []
        if isinstance(detected_faces_dict, dict):
            for _fkey, finfo in detected_faces_dict.items():
                current_frame_faces.append({
                    'bbox': [int(c) for c in finfo['facial_area']],
                    'score': finfo['score'],
                    'landmarks': finfo['landmarks'],
                    'id': None
                })
        all_frames_face_data.append(current_frame_faces)
        frame_count += 1
             
    cap.release()
    logger.info(f"✅ Face detection complete for {frame_count} frames.")
    return frames, all_frames_face_data, video_fps

def apply_face_tracking(
    all_per_frame_detections_input: List[List[Dict[str, Any]]],
    video_fps: float,
    max_disappeared_factor: float = 0.5,
    iou_thresh: float = 0.3
) -> List[List[Dict[str, Any]]]:
    """
    Applies tracking to a sequence of face detections across multiple frames.

    Args:
        all_per_frame_detections_input (List[List[Dict[str, Any]]]): A list where each
            element is a list of face detections for a single frame.
        video_fps (float): The frames per second of the video, used to calculate
                           the disappearance threshold.
        max_disappeared_factor (float): A factor multiplied by FPS to determine how
                                        many frames a face can be missing before being
                                        deregistered.
        iou_thresh (float): The IoU threshold for matching faces between frames.

    Returns:
        List[List[Dict[str, Any]]]: The same list of detections, but with the
                                    'id' field populated by the tracker.
    """
    all_tracked_faces_per_frame = []
    max_disappeared_frames = int(video_fps * max_disappeared_factor) if video_fps > 0 else 10
    tracker = SimpleFaceTracker(max_disappeared_frames=max_disappeared_frames, iou_threshold=iou_thresh)
    num_frames = len(all_per_frame_detections_input)
    logger.info(f"🛤️ Starting face tracking for {num_frames} frames (max_disappeared={max_disappeared_frames})...")
    print(f"Starting face tracking for {num_frames} frames (max_disappeared={max_disappeared_frames})...")
    for frame_idx, current_detections_in_frame in enumerate(all_per_frame_detections_input):
        detections_for_tracker_update = [det.copy() for det in current_detections_in_frame]
        tracked_detections_for_this_frame = tracker.update(detections_for_tracker_update)
        all_tracked_faces_per_frame.append(tracked_detections_for_this_frame)
        if frame_idx > 0 and video_fps > 0 and frame_idx % (int(video_fps) * 10) == 0:
            logger.debug(f"  ...processed tracking for frame {frame_idx}/{num_frames}...")
            print(f"  Processed tracking for frame {frame_idx}/{num_frames}...")
    logger.info("✅ Face tracking complete.")
    print("Face tracking complete.")
    return all_tracked_faces_per_frame


def calculate_mouth_corner_distance(landmarks_dict: Dict[str, List[float]]) -> Optional[float]:
    """
    Calculates the Euclidean distance between the left and right mouth corners.

    Args:
        landmarks_dict (Dict[str, List[float]]): A dictionary of facial landmarks
                                                 from RetinaFace.

    Returns:
        Optional[float]: The distance, or None if landmarks are not available.
    """
    if 'mouth_left' in landmarks_dict and 'mouth_right' in landmarks_dict:
        ml = np.array(landmarks_dict['mouth_left'])
        mr = np.array(landmarks_dict['mouth_right'])
        return np.linalg.norm(ml - mr)
    return None

def get_mouth_activity_per_track(all_faces_with_ids_per_frame_list: List[List[Dict[str, Any]]]) -> Dict[int, List[Optional[float]]]:
    """
    Calculates the mouth corner distance for each tracked face in every frame.

    This creates a time series of mouth-width measurements for each unique face ID,
    which can be used to infer speaking activity.

    Args:
        all_faces_with_ids_per_frame_list (List[List[Dict[str, Any]]]): The list of
            tracked face detections for all frames.

    Returns:
        Dict[int, List[Optional[float]]]: A dictionary where keys are track IDs and
            values are lists of mouth distances for each frame.
    """
    num_video_frames = len(all_faces_with_ids_per_frame_list)
    track_mouth_distances: Dict[int, List[Optional[float]]] = collections.defaultdict(lambda: [None] * num_video_frames)
    for frame_idx, faces_in_frame_list in enumerate(all_faces_with_ids_per_frame_list):
        for face_data_dict in faces_in_frame_list:
            track_id = face_data_dict.get('id')
            if track_id is not None:
                landmarks = face_data_dict.get('landmarks')
                if landmarks:
                    track_mouth_distances[track_id][frame_idx] = calculate_mouth_corner_distance(landmarks)
    return dict(track_mouth_distances)


def determine_active_speaker_ids(
    vad_speech_segments: List[Tuple[int, int]],
    track_mouth_activity_dict: Dict[int, List[Optional[float]]],
    num_video_frames: int,
    video_fps: float,
    vad_frame_duration_ms: int,
    # TODO: The variance threshold is a fixed value. This could be made adaptive.
    # For example, it could be normalized based on the average mouth size of the speaker,
    # making it more robust for faces that are further from the camera.
    min_mouth_movement_variance: float = 0.5
) -> List[Optional[int]]:
    """
    Determines the active speaker for each frame based on audio and visual cues.

    It correlates VAD (Voice Activity Detection) segments with the variance in
    mouth movement for each tracked face to identify the most likely speaker.

    Args:
        vad_speech_segments (List[Tuple[int, int]]): List of (start, end) frame indices from VAD.
        track_mouth_activity_dict (Dict[int, List[Optional[float]]]): Mouth distances per track ID.
        num_video_frames (int): Total number of frames in the video.
        video_fps (float): Frames per second of the video.
        vad_frame_duration_ms (int): Duration of a single VAD frame in milliseconds.
        min_mouth_movement_variance (float): Minimum mouth distance variance to be
                                             considered as speaking.

    Returns:
        List[Optional[int]]: A list where each element is the ID of the active
                             speaker for that frame, or None if no one is speaking.
    """
    active_speaker_per_video_frame_list: List[Optional[int]] = [None] * num_video_frames
    if not track_mouth_activity_dict:
        logger.warning("No face tracks with mouth activity for ASD.")
        print("No face tracks with mouth activity for ASD."); return active_speaker_per_video_frame_list

    vad_frames_per_second = 1000.0 / vad_frame_duration_ms if vad_frame_duration_ms > 0 else 33.33
    video_frames_per_vad_frame = (video_fps / vad_frames_per_second) if video_fps > 0 and vad_frames_per_second > 0 else 1.0

    for vad_start_frame_idx, vad_end_frame_idx in vad_speech_segments:
        video_start_frame = math.floor(vad_start_frame_idx * video_frames_per_vad_frame)
        video_end_frame = math.ceil(vad_end_frame_idx * video_frames_per_vad_frame)
        video_start_frame = max(0, video_start_frame)
        video_end_frame = min(num_video_frames - 1, video_end_frame)
        if video_start_frame >= video_end_frame:
            continue

        best_speaker_id_for_segment, max_variance_for_segment = None, -1.0
        for track_id, mouth_dists_for_track in track_mouth_activity_dict.items():
            segment_mouth_dists = [d for d in mouth_dists_for_track[video_start_frame : video_end_frame + 1] if d is not None]
            min_samples_for_variance = int(video_fps * 0.2) if video_fps > 0 else 5
            if len(segment_mouth_dists) > min_samples_for_variance:
                variance = np.var(segment_mouth_dists)
                if variance > max_variance_for_segment:
                    max_variance_for_segment = variance
                    best_speaker_id_for_segment = track_id
        
        if best_speaker_id_for_segment is not None and max_variance_for_segment >= min_mouth_movement_variance:
            for i in range(video_start_frame, video_end_frame + 1):
                if i < len(active_speaker_per_video_frame_list) and active_speaker_per_video_frame_list[i] is None:
                    active_speaker_per_video_frame_list[i] = best_speaker_id_for_segment
    
    num_speaker_frames = sum(1 for s in active_speaker_per_video_frame_list if s is not None)
    logger.info(f"✅ Active speaker determination complete. Found speakers in {num_speaker_frames}/{num_video_frames} frames.")
    print(f"Active speaker determination complete. Found speakers in {num_speaker_frames}/{num_video_frames} frames.")
    return active_speaker_per_video_frame_list


def enforce_single_speaker_heuristics(
    original_asd_speaker_ids_list: List[Optional[int]],
    all_faces_with_ids_per_frame_list: List[List[Dict[str, Any]]],
    video_frame_width: int,
    video_fps: float,
    max_hold_frames_factor: float = 0.75
) -> List[Optional[int]]:
    """
    Applies heuristics to force a single, consistent speaker throughout the clip.

    This function cleans up the active speaker data by filling gaps, holding onto
    the last known speaker for a short duration, and selecting the most likely
    speaker based on centrality and size when the primary speaker is lost.

    Args:
        original_asd_speaker_ids_list (List[Optional[int]]): The initial list of speaker IDs per frame.
        all_faces_with_ids_per_frame_list (List[List[Dict[str, Any]]]): All tracked faces for each frame.
        video_frame_width (int): The width of the video frame.
        video_fps (float): The video's frames per second.
        max_hold_frames_factor (float): Factor to determine how long to "hold" a speaker ID.

    Returns:
        List[Optional[int]]: The cleaned list of speaker IDs for each frame.
    """
    num_video_frames = len(original_asd_speaker_ids_list)
    if num_video_frames == 0 or len(all_faces_with_ids_per_frame_list) != num_video_frames:
        logger.warning("Mismatch/empty lists for enforce_single_speaker. Skipping.")
        print("Warning: Mismatch/empty lists for enforce_single_speaker. Skipping."); return list(original_asd_speaker_ids_list)

    processed_speaker_ids = list(original_asd_speaker_ids_list)
    last_known_valid_and_visible_speaker_id, frames_since_last_valid_speaker = None, 0
    max_hold_frames = int(video_fps * max_hold_frames_factor) if video_fps > 0 else 15
    frame_center_x = float(video_frame_width) / 2.0 if video_frame_width and video_frame_width > 0 else None

    logger.info("Enforcing 'Forced Single Speaker Mode' heuristics...")
    print("Enforcing 'Forced Single Speaker Mode' heuristics...")

    for i in range(num_video_frames):
        current_faces_in_frame = all_faces_with_ids_per_frame_list[i]
        original_asd_id = original_asd_speaker_ids_list[i]
        speaker_is_assigned = original_asd_id is not None
        assigned_speaker_is_visible = False
        if speaker_is_assigned:
            for face_data in current_faces_in_frame:
                if face_data.get('id') == original_asd_id:
                    assigned_speaker_is_visible = True
                    break
        if speaker_is_assigned and assigned_speaker_is_visible:
            processed_speaker_ids[i] = original_asd_id
            last_known_valid_and_visible_speaker_id = original_asd_id
            frames_since_last_valid_speaker = 0
            continue

        frames_since_last_valid_speaker += 1
        if not current_faces_in_frame:
            processed_speaker_ids[i] = None
            continue

        chosen_fallback_id = None
        if last_known_valid_and_visible_speaker_id is not None and frames_since_last_valid_speaker <= max_hold_frames:
            for face_data in current_faces_in_frame:
                if face_data.get('id') == last_known_valid_and_visible_speaker_id:
                    chosen_fallback_id = last_known_valid_and_visible_speaker_id
                    break
        if chosen_fallback_id is None:
            best_central_id, min_dist_center = None, float('inf')
            largest_area_id, max_area = None, -1
            for face_data in current_faces_in_frame:
                face_id = face_data.get('id')
                if face_id is None: continue
                x1, y1, x2, y2 = face_data['bbox']
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    largest_area_id = face_id
                if frame_center_x is not None:
                    face_cx = (x1 + x2) / 2.0
                    dist = abs(face_cx - frame_center_x)
                    if dist < min_dist_center:
                        min_dist_center = dist
                        best_central_id = face_id
            chosen_fallback_id = best_central_id if best_central_id is not None else largest_area_id

        processed_speaker_ids[i] = chosen_fallback_id
        if chosen_fallback_id is not None:
             last_known_valid_and_visible_speaker_id = chosen_fallback_id
             frames_since_last_valid_speaker = 0
             
    for i in range(num_video_frames):
        if processed_speaker_ids[i] is None and all_faces_with_ids_per_frame_list[i]:
            for face_data in all_faces_with_ids_per_frame_list[i]:
                if face_data.get('id') is not None:
                    processed_speaker_ids[i] = face_data.get('id')
                    break
                    
    logger.info("✅ Forced Single Speaker Mode heuristics applied.")
    print("Forced Single Speaker Mode heuristics applied.")
    return processed_speaker_ids