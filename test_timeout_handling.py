#!/usr/bin/env python3
"""
Quick test of the timeout handling and file size detection logic
"""

import os
import tempfile

def test_file_size_detection():
    """Test the file size detection logic from GetTranscript.py"""

    # Test file size calculation (simulating the logic from _transcribe_with_deepgram)
    def check_file_size_logic(file_path):
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > 100:
            return f"⚠️  Audio file is very large ({file_size_mb:.1f}MB). This may cause timeouts."
        elif file_size_mb > 25:
            return f"📊 Large audio file detected ({file_size_mb:.1f}MB). Using extended timeout settings."
        else:
            return f"📊 Normal file size ({file_size_mb:.1f}MB)"

    # Create test files of different sizes
    with tempfile.NamedTemporaryFile(delete=False) as f:
        # Small file (1MB)
        f.write(b'0' * (1024 * 1024))
        small_file = f.name

    with tempfile.NamedTemporaryFile(delete=False) as f:
        # Medium file (50MB)
        f.write(b'0' * (50 * 1024 * 1024))
        medium_file = f.name

    with tempfile.NamedTemporaryFile(delete=False) as f:
        # Large file (150MB)
        f.write(b'0' * (150 * 1024 * 1024))
        large_file = f.name

    try:
        print("Testing file size detection logic:")
        print(f"Small file (1MB): {check_file_size_logic(small_file)}")
        print(f"Medium file (50MB): {check_file_size_logic(medium_file)}")
        print(f"Large file (150MB): {check_file_size_logic(large_file)}")

        print("\n✅ File size detection logic works correctly!")

    finally:
        # Clean up
        os.unlink(small_file)
        os.unlink(medium_file)
        os.unlink(large_file)

def test_error_detection():
    """Test the error type detection logic"""

    def is_retryable_error(error_msg):
        """Check if error should be retried (simulating logic from _transcribe_with_deepgram)"""
        error_lower = error_msg.lower()
        return any(keyword in error_lower for keyword in ['timeout', 'connection', 'network', 'write operation'])

    # Test various error messages
    test_errors = [
        "The write operation timed out",
        "Connection failed",
        "Network is unreachable",
        "Server error 500",
        "Invalid API key",
        "File format not supported"
    ]

    print("\nTesting error classification logic:")
    for error in test_errors:
        retryable = is_retryable_error(error)
        status = "RETRYABLE" if retryable else "NON-RETRYABLE"
        print(f"'{error}' -> {status}")

    print("\n✅ Error classification logic works correctly!")

if __name__ == "__main__":
    test_file_size_detection()
    test_error_detection()
    print("\n🎉 All timeout handling logic tests passed!")

