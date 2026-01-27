import re
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(f"pipeline.{__name__}")

def flatten_whisperx_transcript(transcript_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts the nested WhisperX-like transcript structure into a flat list of individual words.

    The input `transcript_data` is expected to be a list of segment dictionaries,
    where each segment contains a 'words' key, which is itself a list of word dictionaries.
    Each word dictionary is expected to have 'word', 'start', and 'end' keys.

    This function iterates through all segments and extracts their 'words' lists,
    combining them into a single, flat list. It also ensures that each word string
    is stripped of leading/trailing whitespace for consistency.

    Args:
        transcript_data (List[Dict[str, Any]]): The raw transcript data, typically parsed
                                                 from a JSON file, structured as a list of
                                                 segment dictionaries.

    Returns:
        List[Dict[str, Any]]: A flattened list of word dictionaries. Each dictionary
                              in the returned list will contain 'word' (str), 'start' (float),
                              and 'end' (float) keys, representing the word's text and its
                              absolute start and end timestamps in the original audio/video.
    """
    all_words: List[Dict[str, Any]] = []
    for segment in transcript_data:
        # Ensure 'words' key exists and is not None before extending
        if 'words' in segment and segment['words'] is not None:
            all_words.extend(segment['words'])
    
    # Strip whitespace from each word for cleaner processing downstream
    for word_info in all_words:
        if 'word' in word_info and isinstance(word_info['word'], str):
            word_info['word'] = word_info['word'].strip()
    
    logger.debug(f"Flattened transcript into {len(all_words)} words.")
    return all_words

def find_phrase_in_transcription(
    transcription_words: List[Dict[str, Any]],
    phrase_to_find: str,
    hint_start: Optional[float] = None,
    hint_end: Optional[float] = None,
    search_window: float = 30.0
) -> Tuple[Optional[float], Optional[float]]:
    """
    Finds the start and end time of a specific text phrase within a flat list of transcribed words.

    This function performs a case-insensitive search and ignores common punctuation
    (commas, periods, question marks, exclamation marks) when matching the phrase.
    
    When hint timestamps are provided, it searches within a window around those timestamps
    first before falling back to a full transcript search.
    
    If exact match fails, it attempts partial matching using the first N words
    to find the start time, and the last N words to find the end time.

    Args:
        transcription_words: A flat list of word dictionaries with 'word', 'start', and 'end' keys.
        phrase_to_find: The text phrase to search for within the transcript.
        hint_start: Optional timestamp hint for where the phrase should start.
        hint_end: Optional timestamp hint for where the phrase should end.
        search_window: Seconds to search around the hint timestamps (default 30s).

    Returns:
        Tuple containing (start_timestamp, end_timestamp) of the found phrase,
        or (None, None) if not found.
    """
    def clean_text(text: str) -> str:
        """Remove punctuation and normalize whitespace."""
        cleaned = re.sub(r'\.{2,}', ' ', text)  # Replace ... with space
        cleaned = re.sub(r'[.,?!;:\'"()\[\]{}]', '', cleaned)  # Remove punctuation
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
        return cleaned.lower().strip()
    
    def find_word_sequence(words_subset: List[Dict[str, Any]], phrase_words: List[str]) -> Tuple[Optional[float], Optional[float]]:
        """Try to find a sequence of words in a subset of the transcript."""
        num_phrase_words = len(phrase_words)
        num_words = len(words_subset)
        
        if num_phrase_words > num_words:
            return None, None
        
        for i in range(num_words - num_phrase_words + 1):
            window_dicts = words_subset[i : i + num_phrase_words]
            
            if not all('word' in w and isinstance(w['word'], str) for w in window_dicts):
                continue
            
            window_words = [clean_text(w['word']) for w in window_dicts]
            
            if window_words == phrase_words:
                if 'start' in words_subset[i] and 'end' in words_subset[i + num_phrase_words - 1]:
                    return words_subset[i]['start'], words_subset[i + num_phrase_words - 1]['end']
        
        return None, None
    
    def search_in_window(words: List[Dict[str, Any]], phrase_words: List[str], 
                         window_start: float, window_end: float) -> Tuple[Optional[float], Optional[float]]:
        """Search for phrase within a specific time window."""
        # Filter words to those within the window
        words_in_window = [w for w in words if 'start' in w and window_start <= w['start'] <= window_end]
        if not words_in_window:
            return None, None
        return find_word_sequence(words_in_window, phrase_words)
    
    # Clean the phrase
    clean_phrase = clean_text(phrase_to_find)
    phrase_words = [w for w in clean_phrase.split() if w]

    if not phrase_words:
        logger.warning(f"Phrase '{phrase_to_find}' is empty or invalid after cleaning. Skipping search.")
        return None, None
    
    # If phrase is very short (1-2 words), require hint timestamps to avoid false matches
    if len(phrase_words) <= 2 and hint_start is None:
        logger.warning(f"Phrase '{phrase_to_find}' is too short ({len(phrase_words)} words) without timestamp hints. Skipping to avoid false matches.")
        return None, None

    # If we have hint timestamps, search within a window around them first
    if hint_start is not None:
        window_start = max(0, hint_start - search_window)
        window_end = (hint_end or hint_start) + search_window
        
        logger.debug(f"Searching within window [{window_start:.1f}s - {window_end:.1f}s] based on LLM hints")
        
        # Try exact match within window
        start_time, end_time = search_in_window(transcription_words, phrase_words, window_start, window_end)
        if start_time is not None:
            logger.info(f"🎯 Found phrase '{phrase_to_find}' from {start_time:.2f}s to {end_time:.2f}s.")
            return start_time, end_time
        
        # Try partial match within window (first few words)
        min_words = min(4, len(phrase_words))
        for n in range(len(phrase_words), min_words - 1, -1):
            prefix_words = phrase_words[:n]
            start_time, _ = search_in_window(transcription_words, prefix_words, window_start, window_end)
            if start_time is not None:
                # Estimate end based on phrase length
                estimated_end = start_time + len(phrase_words) * 0.4
                logger.info(f"🎯 Found phrase '{phrase_to_find[:50]}...' via {n}-word prefix from {start_time:.2f}s to {estimated_end:.2f}s.")
                return start_time, estimated_end
    
    # Fall back to full transcript search (only for phrases with 3+ words)
    if len(phrase_words) >= 3:
        logger.debug(f"Searching full transcript for {len(phrase_words)}-word phrase")
        start_time, end_time = find_word_sequence(transcription_words, phrase_words)
        if start_time is not None:
            logger.info(f"🎯 Found phrase '{phrase_to_find}' from {start_time:.2f}s to {end_time:.2f}s.")
            return start_time, end_time
        
        # Try partial match on full transcript
        min_words = min(4, len(phrase_words) // 2, len(phrase_words))
        for n in range(len(phrase_words), min_words - 1, -1):
            prefix_words = phrase_words[:n]
            start_time, _ = find_word_sequence(transcription_words, prefix_words)
            if start_time is not None:
                estimated_end = start_time + len(phrase_words) * 0.4
                logger.info(f"🎯 Found phrase '{phrase_to_find[:50]}...' via {n}-word prefix from {start_time:.2f}s to {estimated_end:.2f}s.")
                return start_time, estimated_end

    logger.info(f"Phrase '{phrase_to_find[:70]}' not found in the transcription.")
    return None, None