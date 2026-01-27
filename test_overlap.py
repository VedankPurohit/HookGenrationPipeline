#!/usr/bin/env python3
"""
Quick test of overlap detection and merging functions
"""

from typing import List, Dict, Any

def detect_overlapping_clips(clips: List[Dict[str, Any]], tolerance: float = 0.1) -> List[List[int]]:
    """Detects overlapping clips based on their time ranges."""
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
    """Merges overlapping clips using a priority-based strategy."""
    merged_clips = clips.copy()
    indices_to_remove = set()

    print(f"🔄 Processing {len(overlap_groups)} overlap groups...")

    for group_idx, overlap_indices in enumerate(overlap_groups):
        # Skip if any clip in this group has already been marked for removal
        if any(idx in indices_to_remove for idx in overlap_indices):
            continue

        # Find the clip with the earliest start time (primary clip)
        primary_idx = min(overlap_indices, key=lambda idx: clips[idx]['original_start'])
        primary_clip = clips[primary_idx]

        # Find the latest end time among all overlapping clips
        max_end_time = max(clips[idx]['original_end'] for idx in overlap_indices)

        # Extend the primary clip to cover the overlap
        original_end = primary_clip['original_end']
        merged_clips[primary_idx]['original_end'] = max_end_time

        # Mark other clips in the group for removal
        for idx in overlap_indices:
            if idx != primary_idx:
                indices_to_remove.add(idx)

        # Log the merge operation
        overlap_duration = max_end_time - original_end
        print(f"🔀 Merged overlap group {group_idx + 1}: Extended clip {primary_idx + 1} by {overlap_duration:.2f}s "
              f"(from {original_end:.2f}s to {max_end_time:.2f}s)")

        # Combine text and explanations from merged clips
        all_texts = [clips[idx].get('text_include', '') for idx in overlap_indices]
        all_explanations = [clips[idx].get('why_this_clip', '') for idx in overlap_indices]

        merged_clips[primary_idx]['text_include'] = ' | '.join(filter(None, all_texts))
        merged_clips[primary_idx]['why_this_clip'] = ' + '.join(filter(None, all_explanations))

    # Remove the overlapping clips that were merged
    merged_clips = [clip for idx, clip in enumerate(merged_clips) if idx not in indices_to_remove]

    print(f"✅ Overlap merging complete. Removed {len(indices_to_remove)} duplicate clips.")

    return merged_clips

if __name__ == "__main__":
    # Test data with overlapping clips
    test_clips = [
        {'original_start': 0.0, 'original_end': 5.0, 'text_include': 'clip 1', 'why_this_clip': 'reason 1'},
        {'original_start': 3.0, 'original_end': 8.0, 'text_include': 'clip 2', 'why_this_clip': 'reason 2'},  # overlaps with clip 1
        {'original_start': 10.0, 'original_end': 15.0, 'text_include': 'clip 3', 'why_this_clip': 'reason 3'},  # no overlap
        {'original_start': 12.0, 'original_end': 18.0, 'text_include': 'clip 4', 'why_this_clip': 'reason 4'},  # overlaps with clip 3
    ]

    print("Original clips:")
    for i, clip in enumerate(test_clips):
        print(f"  Clip {i+1}: {clip['original_start']:.1f}s - {clip['original_end']:.1f}s: {clip['text_include']}")

    print("\nTesting overlap detection...")
    overlaps = detect_overlapping_clips(test_clips)
    print(f"🔍 Found {len(overlaps)} overlapping groups: {overlaps}")

    if overlaps:
        print("\nTesting merge functionality...")
        merged = merge_overlapping_clips(test_clips, overlaps)

        print(f"\n✅ Final result: {len(merged)} clips after merging")
        for i, clip in enumerate(merged):
            print(f"  Clip {i+1}: {clip['original_start']:.1f}s - {clip['original_end']:.1f}s")
            print(f"    Text: {clip['text_include']}")
            print(f"    Why: {clip['why_this_clip']}")

    print("\n✅ Test completed successfully!")

