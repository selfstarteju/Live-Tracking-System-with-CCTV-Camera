#!/usr/bin/env python3
"""Seed demo attendance data for the Live-Tracking project.

This script copies a few images from the `output/` folder into `scans/` folders
and inserts attendance + scan_images records into the SQLite database using
the helper functions from `database.py`.

Run from the project root like:
    python3 scripts/seed_demo.py
"""
import os
import shutil
import random
import sys
from datetime import datetime, timedelta

# Ensure project root is on sys.path so we can import local modules when
# running this script from the scripts/ directory.
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import database as db


def ensure_dirs(root: str):
    os.makedirs(os.path.join(root, 'scans'), exist_ok=True)


def pick_frames(output_dir: str, n: int = 10):
    files = [f for f in os.listdir(output_dir) if f.lower().endswith('.jpg')]
    files = sorted(files)
    if not files:
        raise RuntimeError(f'No frames found in {output_dir}')
    # Return full paths
    return [os.path.join(output_dir, f) for f in files][:n]


def seed_demo():
    project_root = os.path.dirname(os.path.dirname(__file__))
    output_dir = os.path.join(project_root, 'output')
    scans_root = os.path.join(project_root, 'scans')

    ensure_dirs(project_root)

    db.init_db()

    frames = pick_frames(output_dir, n=50)

    demo_people = [
        {'barcode': 'ID1001', 'name': 'Alice Demo', 'color': 'green'},
        {'barcode': 'ID1002', 'name': 'Bob Demo', 'color': 'yellow'},
        {'barcode': 'ID1003', 'name': 'Charlie Demo', 'color': 'red'},
        {'barcode': 'ID1004', 'name': 'Dana Demo', 'color': 'blue'},
    ]

    created = 0

    now = datetime.now()

    for person in demo_people:
        # Create a few attendance entries per person
        for j in range(3):
            # random recent timestamp within last 3 days
            ts = now - timedelta(days=random.randint(0, 3), hours=random.randint(0, 23), minutes=random.randint(0, 59))
            ts_str = ts.strftime('%Y%m%d_%H%M%S')

            scan_folder = os.path.join(scans_root, f"{person['barcode']}_{ts_str}")
            os.makedirs(scan_folder, exist_ok=True)

            # pick a random frame to represent the full frame
            frame_src = random.choice(frames)
            full_frame_dst = os.path.join(scan_folder, 'full_frame.jpg')
            shutil.copy2(frame_src, full_frame_dst)

            # For demo, reuse the same image as card image
            card_image_dst = os.path.join(scan_folder, 'id_card.jpg')
            shutil.copy2(frame_src, card_image_dst)

            # No face images for the demo (empty list)
            face_image_paths = []

            # Insert into DB
            success = db.log_attendance_with_images(
                barcode=person['barcode'],
                name=person['name'],
                color=person['color'],
                timestamp=ts,
                full_frame_path=full_frame_dst,
                card_image_path=card_image_dst,
                face_image_paths=face_image_paths
            )

            if success:
                created += 1

    print(f"Seeded {created} demo attendance records.")


if __name__ == '__main__':
    try:
        seed_demo()
    except Exception as e:
        print('Seeding failed:', e)
        raise
