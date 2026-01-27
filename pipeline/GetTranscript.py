import os
import json
import logging
from typing import Dict, List, Any

from dotenv import load_dotenv
from deepgram import DeepgramClient, PrerecordedOptions, FileSource

# Load environment variables at the module level
load_dotenv(dotenv_path=".env")
logger = logging.getLogger(f"pipeline.{__name__}")


def _transcribe_with_deepgram(audio_path: str, max_retries: int = 3) -> Dict[str, Any]:
    """
    Transcribes the given audio file using the Deepgram API with retry logic for timeouts.

    Args:
        audio_path (str): The path to the audio file to transcribe.
        max_retries (int): Maximum number of retry attempts for network timeouts.

    Returns:
        Dict[str, Any]: The raw JSON response from Deepgram as a dictionary.

    Raises:
        ValueError: If the DEEPGRAM_API_KEY is not set.
        Exception: For any other errors during the API call.
    """
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise ValueError("DEEPGRAM_API_KEY not found in environment variables.")

    # Check file size and warn if very large
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    if file_size_mb > 100:
        logger.warning(f"⚠️  Audio file is very large ({file_size_mb:.1f}MB). This may cause timeouts.")
    elif file_size_mb > 25:
        logger.info(f"📊 Large audio file detected ({file_size_mb:.1f}MB). Using extended timeout settings.")

    logger.info("Initializing Deepgram client...")
    deepgram = DeepgramClient(api_key)

    with open(audio_path, "rb") as file:
        buffer_data = file.read()
    payload: FileSource = {"buffer": buffer_data}

    # TODO: The Deepgram model ('nova-2') and other options (language, etc.) are hardcoded.
    # These should be moved to `config.py` to allow users to easily switch to other models
    # or languages (e.g., 'whisper-large') without altering the source code.
    options = PrerecordedOptions(
        model="nova-2", language="en", smart_format=True, punctuate=True,
        diarize=True, utterances=True, filler_words=True
    )

    # Configure extended timeout for large file uploads (inspired by working implementation)
    import httpx
    timeout = httpx.Timeout(
        600.0,  # total timeout (10 minutes for large files)
        connect=60.0,  # connection timeout
        read=300.0,  # read timeout
        write=300.0,  # write timeout (important for large file uploads)
    )

    last_exception = None
    for attempt in range(max_retries):
        try:
            logger.info(f"Requesting transcription from Deepgram (attempt {attempt + 1}/{max_retries})...")
            response = deepgram.listen.rest.v("1").transcribe_file(payload, options, timeout=timeout)
            logger.info("✅ Successfully received transcription from Deepgram.")
            return response.to_dict()

        except Exception as e:
            last_exception = e
            error_msg = str(e).lower()

            # Check if this is a timeout or network error that we should retry
            if any(keyword in error_msg for keyword in ['timeout', 'connection', 'network', 'write operation']):
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 30  # Progressive backoff: 30s, 60s, 90s
                    logger.warning(f"⚠️  Network/timeout error (attempt {attempt + 1}/{max_retries}): {e}")
                    logger.info(f"⏳ Retrying in {wait_time} seconds...")
                    import time
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"❌ All {max_retries} attempts failed due to network/timeout errors")
            else:
                # Non-retryable error (API key, file format, etc.)
                logger.error(f"❌ Non-retryable error during transcription: {e}")
                break

    # If we get here, all retries failed
    raise last_exception


def _create_llm_summary(deepgram_response: Dict[str, Any]) -> str:
    """
    Creates a token-efficient summary from a Deepgram response, suitable for LLMs.

    Args:
        deepgram_response (Dict[str, Any]): The parsed JSON from a Deepgram API call.

    Returns:
        str: A formatted string containing the speaker-separated transcript.
    """
    logger.info("📄 Creating token-efficient summary for LLM...")
    try:
        utterances = deepgram_response.get('results', {}).get('utterances', [])
        if not utterances:
            logger.warning("No utterances found in the response to summarize.")
            return ""
    except (KeyError, TypeError):
        logger.error("Could not find 'results.utterances' in the Deepgram response.")
        return ""

    summary_lines = [
        f"SPEAKER_{u.get('speaker', 99):02d} ({u.get('start'):.2f}s - {u.get('end'):.2f}s): {u.get('transcript', '')}"
        for u in utterances
    ]
    logger.info(f"✅ LLM Summary created with {len(summary_lines)} speaker turns.")
    return "\n".join(summary_lines)


def _parse_to_whisperx_format(deepgram_response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Converts a Deepgram response to the WhisperX format for pipeline compatibility.

    Args:
        deepgram_response (Dict[str, Any]): The parsed JSON from a Deepgram API call.

    Returns:
        List[Dict[str, Any]]: A list of segment dictionaries in WhisperX format.
    """
    logger.info("🔄 Parsing Deepgram response into WhisperX-compatible format...")
    try:
        words_list = deepgram_response['results']['channels'][0]['alternatives'][0]['words']
        if not words_list:
            logger.warning("No words found in the response for WhisperX parsing.")
            return []
    except (KeyError, TypeError, IndexError):
        logger.error("Could not find word-level data in Deepgram response for parsing.")
        return []

    final_segments = []
    current_segment = None
    for word_data in words_list:
        speaker_id = f"SPEAKER_{word_data.get('speaker', 99):02d}"
        word_text = word_data.get('punctuated_word', word_data.get('word', ''))
        word_info = {
            "word": word_text, "start": word_data.get('start', 0.0),
            "end": word_data.get('end', 0.0), "score": word_data.get('confidence', 0.0),
            "speaker": speaker_id
        }
        if current_segment is None or current_segment["speaker"] != speaker_id:
            if current_segment:
                current_segment["text"] = current_segment["text"].strip()
                final_segments.append(current_segment)
            current_segment = {
                "text": "", "start": word_info["start"], "end": word_info["end"],
                "speaker": speaker_id, "words": []
            }
        current_segment["words"].append(word_info)
        current_segment["text"] += word_text + " "
        current_segment["end"] = word_info["end"]
    if current_segment:
        current_segment["text"] = current_segment["text"].strip()
        final_segments.append(current_segment)

    logger.info(f"✅ Parsing complete. Created {len(final_segments)} WhisperX-compatible segments.")
    return final_segments


def generate_all_transcripts(audio_path: str, output_dir: str):
    """
    Orchestrates the entire transcription process for a given audio file.

    This function will:
    1. Transcribe the audio file using Deepgram.
    2. Save the raw Deepgram JSON response.
    3. Generate and save a human-readable LLM summary.
    4. Generate and save a WhisperX-compatible transcript for the pipeline.

    Args:
        audio_path (str): The path to the source audio file (e.g., 'source_audio.wav').
        output_dir (str): The directory to save all transcript assets.
    """
    logger.info(f"--- Starting Full Transcription Process for {audio_path} ---")
    raw_transcript_path = os.path.join(output_dir, "deepgram_raw_transcript.json")
    llm_summary_path = os.path.join(output_dir, "llm_summary.txt")
    final_transcript_path = os.path.join(output_dir, "transcript.json")

    try:
        # Step 1: Transcribe with retry logic for network timeouts
        deepgram_response = _transcribe_with_deepgram(audio_path, max_retries=3)
        with open(raw_transcript_path, "w", encoding="utf-8") as f:
            json.dump(deepgram_response, f, indent=4)
        logger.info(f"💾 Raw Deepgram transcript saved to {raw_transcript_path}")

        # Step 2: Create and save LLM summary
        llm_summary = _create_llm_summary(deepgram_response)
        with open(llm_summary_path, 'w', encoding="utf-8") as f:
            f.write(llm_summary)
        logger.info(f"💾 LLM summary saved to {llm_summary_path}")

        # Step 3: Create and save WhisperX-compatible format
        whisperx_data = _parse_to_whisperx_format(deepgram_response)
        with open(final_transcript_path, 'w', encoding="utf-8") as f:
            json.dump(whisperx_data, f, indent=2)
        logger.info(f"💾 WhisperX-compatible transcript saved to {final_transcript_path}")

    except Exception as e:
        logger.critical(f"❌ A critical error occurred during the transcription process: {e}", exc_info=True)
        # Re-raise the exception to halt the preparation script if transcription fails
        raise
