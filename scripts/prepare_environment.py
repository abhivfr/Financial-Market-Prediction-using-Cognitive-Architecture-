#!/usr/bin/env python
# scripts/prepare_environment.py

import os
import shutil
from datetime import datetime
import argparse
import glob

def ensure_dir(directory):
    """Ensure a directory exists, create if not"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")

def archive_files(source_dir, target_dir, file_pattern="*.pth"):
    """Move files from source_dir to target_dir if they match pattern"""
    # Ensure source and target directories exist
    if not os.path.exists(source_dir):
        print(f"Source directory does not exist: {source_dir}")
        return 0
    
    ensure_dir(target_dir)
    
    # Find files matching pattern
    pattern = os.path.join(source_dir, file_pattern)
    files = glob.glob(pattern)
    
    # Archive files
    count = 0
    for file in files:
        filename = os.path.basename(file)
        target_file = os.path.join(target_dir, filename)
        try:
            shutil.move(file, target_file)
            print(f"Archived: {filename}")
            count += 1
        except Exception as e:
            print(f"Error archiving {filename}: {e}")
    
    return count

def prepare_environment(archive=True):
    """Prepare environment for training and validation"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create archive directories
    archive_base = "archive"
    ensure_dir(archive_base)
    ensure_dir(os.path.join(archive_base, "models"))
    ensure_dir(os.path.join(archive_base, "checkpoints"))
    ensure_dir(os.path.join(archive_base, "logs"))
    
    # Create validation directories
    validation_base = "validation"
    ensure_dir(validation_base)
    ensure_dir(os.path.join(validation_base, "baselines"))
    ensure_dir(os.path.join(validation_base, "results"))
    ensure_dir(os.path.join(validation_base, "plots"))
    
    # Archive existing models and logs if requested
    archived_count = 0
    if archive:
        print("\nArchiving existing models and logs...")
        archived_count += archive_files("models", os.path.join(archive_base, "models"))
        archived_count += archive_files("checkpoints", os.path.join(archive_base, "checkpoints"))
        archived_count += archive_files(".", os.path.join(archive_base, "logs"), "train_output*.txt")
        print(f"Archived {archived_count} files in total")
    
    # Ensure model directories exist (may have been created above)
    ensure_dir("models")
    ensure_dir("checkpoints")
    ensure_dir("logs")
    
    print(f"\nEnvironment prepared at {timestamp}")
    print(f"Archived {archived_count} files")
    print("Ready for training and validation")
    
    return {
        "timestamp": timestamp,
        "archived_count": archived_count
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare environment for training and validation")
    parser.add_argument("--no-archive", action="store_true", help="Skip archiving existing models and logs")
    args = parser.parse_args()
    
    prepare_environment(archive=not args.no_archive)
