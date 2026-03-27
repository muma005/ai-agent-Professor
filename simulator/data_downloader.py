"""
Data Downloader — Downloads competition data via Kaggle API.

This module handles downloading competition data from Kaggle and preparing
it for the simulator. It's separate from the data_splitter to keep concerns
isolated.

Usage:
    from simulator.data_downloader import download_competition
    
    # Download a competition
    data_dir = download_competition("spaceship-titanic")
    
    # The data is now at simulator/data/spaceship-titanic/raw/
"""

import json
import zipfile
import subprocess
import hashlib
from pathlib import Path
from typing import Optional

from simulator.competition_registry import CompetitionEntry


def download_competition(
    entry: CompetitionEntry,
    force: bool = False,
) -> Path:
    """
    Download competition data via Kaggle API.
    
    Args:
        entry: Competition entry with kaggle_id
        force: If True, re-download even if data exists
    
    Returns:
        Path to downloaded data directory
    """
    target_dir = entry.get_data_dir() / "raw"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if data already exists
    if not force and any(target_dir.iterdir()):
        print(f"[downloader] Data already present at {target_dir}. Skipping download.")
        return target_dir
    
    print(f"[downloader] Downloading {entry.kaggle_id}...")
    
    # Try kaggle CLI first
    kaggle_exe = _find_kaggle_cli()
    
    if kaggle_exe:
        result = subprocess.run(
            [kaggle_exe, "competitions", "download", "-c", entry.kaggle_id, "-p", str(target_dir)],
            capture_output=True,
            text=True,
        )
        
        if result.returncode == 0:
            # Extract zip files
            for p in target_dir.glob("*.zip"):
                print(f"[downloader] Extracting {p.name}...")
                with zipfile.ZipFile(p, 'r') as zf:
                    zf.extractall(target_dir)
                p.unlink()  # Remove zip after extraction
            
            print(f"[downloader] Downloaded to {target_dir}")
            return target_dir
        else:
            print(f"[downloader] Kaggle CLI failed: {result.stderr}")
    
    # Fall back to manual download
    print(
        f"[downloader] Automatic download failed or unavailable.\n"
        f"Please download manually from:\n"
        f"  https://www.kaggle.com/competitions/{entry.kaggle_id}/data\n"
        f"and extract to: {target_dir}"
    )
    
    return target_dir


def _find_kaggle_cli() -> Optional[Path]:
    """Find the kaggle CLI executable."""
    import sys
    import os
    
    # Try common locations
    candidates = [
        Path(sys.executable).parent / "kaggle",
        Path(sys.executable).parent / "kaggle.exe",
        Path(os.environ.get("HOME", "")) / ".local" / "bin" / "kaggle",
    ]
    
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    
    # Try PATH
    import shutil
    kaggle_path = shutil.which("kaggle")
    if kaggle_path:
        return Path(kaggle_path)
    
    return None


def verify_data_integrity(
    entry: CompetitionEntry,
    expected_hash: Optional[str] = None,
) -> bool:
    """
    Verify downloaded data integrity.
    
    Args:
        entry: Competition entry
        expected_hash: Expected MD5 hash of train file (optional)
    
    Returns:
        True if data is valid
    """
    data_dir = entry.get_data_dir() / "raw"
    train_path = data_dir / entry.train_file if hasattr(entry, 'train_file') else data_dir / "train.csv"
    
    if not train_path.exists():
        print(f"[downloader] Train file not found at {train_path}")
        return False
    
    # Compute hash
    actual_hash = hashlib.md5(train_path.read_bytes()).hexdigest()
    
    if expected_hash and actual_hash != expected_hash:
        print(
            f"[downloader] Hash mismatch!\n"
            f"  Expected: {expected_hash}\n"
            f"  Actual:   {actual_hash}"
        )
        return False
    
    print(f"[downloader] Data integrity verified (hash: {actual_hash})")
    return True


def list_downloaded_competitions() -> list[str]:
    """List all competitions that have been downloaded."""
    data_dir = Path("simulator/data")
    if not data_dir.exists():
        return []
    
    downloaded = []
    for comp_dir in data_dir.iterdir():
        if comp_dir.is_dir():
            raw_dir = comp_dir / "raw"
            if raw_dir.exists() and any(raw_dir.iterdir()):
                downloaded.append(comp_dir.name)
    
    return sorted(downloaded)
