#!/usr/bin/env python3
"""Analyze debug_failed images and try barcode decoding strategies.

This script imports the IDCardTracker (safe) and runs try_read_barcode_variants
on each enhanced image in debug_failed/. It prints results and copies any
images where a barcode-like string was found to debug_success/ for inspection.
"""
import os
import shutil
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from main import IDCardTracker

DEBUG_DIR = repo_root / 'debug_failed'
SUCCESS_DIR = repo_root / 'debug_success'

def main():
    tracker = IDCardTracker(save_frames=False)

    if not DEBUG_DIR.exists():
        print('No debug_failed directory found; run tracker with DEBUG_SAVE_FAILED=1 first')
        return

    os.makedirs(SUCCESS_DIR, exist_ok=True)

    enhanced_files = sorted([p for p in DEBUG_DIR.iterdir() if p.name.startswith('enhanced_') and p.suffix.lower() in ('.jpg','.png')])
    total = len(enhanced_files)
    print(f'Found {total} enhanced images to analyze')

    successes = 0
    for p in enhanced_files:
        img_path = str(p)
        try:
            res = tracker.try_read_barcode_variants(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))
        except Exception as e:
            res = []

        if res:
            successes += 1
            print(f'OK: {p.name} -> {res}')
            # copy original warped and enhanced for inspection
            base = p.name.replace('enhanced_', '')
            warped = DEBUG_DIR / ('warped_' + base)
            dest_enh = SUCCESS_DIR / p.name
            shutil.copy2(p, dest_enh)
            if warped.exists():
                shutil.copy2(warped, SUCCESS_DIR / warped.name)
        else:
            print(f'NO : {p.name}')

    print(f'Analysis complete: {successes}/{total} had candidate barcodes')

if __name__ == '__main__':
    # avoid importing cv2 at module top-level in case environment differs
    import cv2
    main()
