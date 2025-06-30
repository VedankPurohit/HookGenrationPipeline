import logging
import json
import os
from typing import Union, List, Dict, Any

logger = logging.getLogger(f"pipeline.{__name__}")

def timestamp_to_seconds(ts_str: Union[str, float, int]) -> float:
    """
    Converts a timestamp string or numerical value into total seconds.

    This function supports various timestamp formats:
    - "HH:MM:SS.ss" (e.g., "01:05:30.123")
    - "MM:SS.ss" (e.g., "05:30.123")
    - "SS.ss" (e.g., "30.123")
    - A raw number (float or int) representing seconds.

    Args:
        ts_str (Union[str, float, int]): The timestamp to convert. Can be a string
                                         in one of the specified formats, or a float/int.

    Returns:
        float: The total number of seconds represented by the timestamp.

    Raises:
        TypeError: If the input `ts_str` is not a string, float, or integer.
        ValueError: If the string format is not recognized or cannot be parsed.
    """
    try:
        # If it's already a number, just return it as a float.
        return float(ts_str)
    except (ValueError, TypeError):
        # If it's a string, parse it.
        pass

    if not isinstance(ts_str, str):
         raise TypeError(f"Invalid type for timestamp: Expected str, float, or int, but got {type(ts_str)}")

    parts: List[str] = ts_str.split(':')
    seconds: float = 0.0

    if len(parts) == 3:
        # Format: HH:MM:SS.ss
        try:
            seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        except ValueError as e:
            raise ValueError(f"Invalid HH:MM:SS.ss format for '{ts_str}': {e}") from e
    elif len(parts) == 2:
        # Format: MM:SS.ss
        try:
            seconds = int(parts[0]) * 60 + float(parts[1])
        except ValueError as e:
            raise ValueError(f"Invalid MM:SS.ss format for '{ts_str}': {e}") from e
    elif len(parts) == 1:
        # Format: SS.ss
        try:
            seconds = float(parts[0])
        except ValueError as e:
            raise ValueError(f"Invalid SS.ss format for '{ts_str}': {e}") from e
    else:
        raise ValueError(f"Invalid timestamp format: '{ts_str}'. Expected HH:MM:SS.ss, MM:SS.ss, or SS.ss.")
    
    return seconds


def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    """
    Calculates the Intersection over Union (IoU) for two bounding boxes.

    IoU is a metric used to quantify the overlap between two bounding boxes.
    It is defined as the area of intersection divided by the area of union.

    Args:
        boxA (List[float]): The coordinates of the first bounding box, formatted as [x1, y1, x2, y2].
                            x1, y1 are the top-left corner coordinates, and x2, y2 are the bottom-right.
        boxB (List[float]): The coordinates of the second bounding box, formatted as [x1, y1, x2, y2].

    Returns:
        float: The IoU score, a value between 0.0 and 1.0. A value of 1.0 indicates
               perfect overlap, while 0.0 indicates no overlap.
    """
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA: float = max(boxA[0], boxB[0])
    yA: float = max(boxA[1], boxB[1])
    xB: float = min(boxA[2], boxB[2])
    yB: float = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    # Ensure the intersection is positive (i.e., boxes actually overlap)
    interArea: float = max(0.0, xB - xA) * max(0.0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea: float = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea: float = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the Intersection over Union
    # Add a small epsilon (1e-6) to the denominator to prevent division by zero
    # in cases where both box areas are zero or only touch at a point/line.
    iou: float = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    
    return iou

def log_run_summary(run_output_dir: str, config: Dict[str, Any], creative_brief: List[Dict[str, Any]]) -> None:
    """
    Saves a JSON summary of the current run's configuration and creative brief.

    This summary is useful for reproducibility and debugging, providing a snapshot
    of the parameters and inputs used for a specific video generation run.

    Args:
        run_output_dir (str): The absolute path to the main output directory for the current run.
                              The 'run_summary.json' file will be saved here.
        config (Dict[str, Any]): The configuration dictionary used for the run, containing
                                 various pipeline settings and parameters.
        creative_brief (List[Dict[str, Any]]): The creative brief data (e.g., LLM output)
                                              that guided the clip selection and processing.
    """
    summary: Dict[str, Any] = {
        "run_configuration": config,
        "creative_brief": creative_brief
    }
    summary_path: str = os.path.join(run_output_dir, "run_summary.json")
    try:
        # Ensure the output directory exists before writing the file
        os.makedirs(run_output_dir, exist_ok=True)
        with open(summary_path, 'w', encoding='utf-8') as f:
            # Use a default lambda to handle non-serializable types like sets, converting them to lists.
            # Also handles other non-serializable types by converting them to their string representation.
            json.dump(summary, f, indent=4, default=lambda o: list(o) if isinstance(o, set) else str(o))
        logger.info(f"📋 Run summary saved to {summary_path}")
    except Exception as e:
        logger.error(f"❌ Could not save run summary to {summary_path}: {e}", exc_info=True)