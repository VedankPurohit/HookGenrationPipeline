import logging
import os
import sys

def setup_logging(run_output_dir: str):
    """
    Configures the logging system for the entire application.

    This sets up a logger that outputs:
    - INFO level and above to the console (for clean user feedback).
    - DEBUG level and above to a dedicated log file inside the run's output directory.

    Args:
        run_output_dir (str): The absolute path to the run's output directory,
                              where the log file will be created.
    """
    # Define the log file path
    log_file_path = os.path.join(run_output_dir, 'pipeline.log')
    
    # Get the root logger for the 'pipeline' package
    logger = logging.getLogger('pipeline')
    logger.setLevel(logging.DEBUG)  # Set the lowest level to capture all messages

    # Clear existing handlers to prevent duplicate logs if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # --- Create Console Handler (for high-level feedback) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # --- Create File Handler (for detailed debugging) ---
    try:
        # Ensure the output directory exists before creating the log file
        os.makedirs(run_output_dir, exist_ok=True)
        # Specify UTF-8 encoding to handle all characters, including emojis
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging initialized. Detailed log file at: {log_file_path}")
    except Exception as e:
        logger.error(f"Failed to set up file logging to {log_file_path}: {e}")
        logger.info("Proceeding without file logging. All logs will go to console.")