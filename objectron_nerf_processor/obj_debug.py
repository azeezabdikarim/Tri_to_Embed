#!/usr/bin/env python3
"""
Debug script to examine Objectron annotation file structure
"""

import struct
import sys
from pathlib import Path

def examine_annotation_file(file_path):
    """Examine the structure of an annotation file."""
    print(f"Examining: {file_path}")
    print(f"File size: {file_path.stat().st_size} bytes")
    
    with open(file_path, 'rb') as f:
        data = f.read()
    
    print("\n=== RAW DATA SAMPLE ===")
    print(f"First 200 bytes (hex): {data[:200].hex()}")
    print(f"First 200 bytes (repr): {repr(data[:200])}")
    
    print("\n=== LOOKING FOR TEXT PATTERNS ===")
    # Look for common text patterns
    text_patterns = ['bike', 'book', 'bottle', 'camera', 'cereal_box', 'chair', 'cup', 'laptop', 'shoe']
    for pattern in text_patterns:
        if pattern.encode('utf-8') in data:
            print(f"Found text: '{pattern}'")
    
    print("\n=== PROTOBUF MESSAGE STRUCTURE ===")
    # Try to parse protobuf structure
    offset = 0
    message_count = 0
    
    while offset < len(data) and message_count < 10:  # Limit to first 10 messages
        try:
            if offset + 4 > len(data):
                break
                
            # Try reading message length
            msg_len = struct.unpack('<I', data[offset:offset + 4])[0]
            print(f"Message {message_count}: offset={offset}, length={msg_len}")
            
            if msg_len > len(data) - offset - 4 or msg_len < 0:
                print(f"  Invalid message length: {msg_len}")
                break
            
            offset += 4
            message_data = data[offset:offset + msg_len]
            
            # Look for patterns in this message
            print(f"  Message data sample: {message_data[:50].hex()}")
            
            # Look for float patterns (potential coordinates)
            float_count = 0
            for i in range(0, min(len(message_data) - 4, 100), 4):
                try:
                    val = struct.unpack('<f', message_data[i:i + 4])[0]
                    if -100 < val < 100 and abs(val) > 1e-6:  # Reasonable coordinate range
                        float_count += 1
                except:
                    pass
            print(f"  Potential coordinates found: {float_count}")
            
            offset += msg_len
            message_count += 1
            
        except Exception as e:
            print(f"  Error parsing message: {e}")
            break
    
    print(f"\nParsed {message_count} messages total")

if __name__ == "__main__":
    # Examine a few annotation files
    annotation_files = list(Path("data/temp/annotations").glob("*.pbdata"))
    
    if not annotation_files:
        print("No annotation files found. Run the main processor first to download some samples.")
        sys.exit(1)
    
    for i, file_path in enumerate(annotation_files[:3]):  # Examine first 3 files
        print("="*80)
        examine_annotation_file(file_path)
        print()