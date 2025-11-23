#!/usr/bin/env python3
"""
Verify Challenge Set Data Loading
Quick test to ensure data is accessible and valid

Usage:
    python scripts/01_verify_data.py
    
    # Or run in background with screen:
    screen -S verify_data
    python scripts/01_verify_data.py
    # Press Ctrl+A then D to detach
    
    # Check progress:
    tail -f logs/01_verify_data_*.log
    
    # Reattach:
    screen -r verify_data
"""

import json
import sys
from pathlib import Path
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from logger_config import setup_logger, log_section, log_subsection

def verify_challenge_set(logger):
    """Load and verify the challenge set"""
    
    data_path = Path(__file__).parent.parent / "data" / "raw" / "challenge_set.json"
    
    log_section(logger, "SPOTIFY CHALLENGE SET - DATA VERIFICATION")
    
    try:
        # Step 1: Load data
        log_subsection(logger, "[1/5] Loading Data")
        logger.info(f"Loading from: {data_path}")
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        logger.info("✓ File loaded successfully")
        
        # Step 2: Basic info
        playlists = data.get('playlists', [])
        log_subsection(logger, "[2/5] Dataset Overview")
        logger.info(f"Total playlists: {len(playlists):,}")
        logger.info(f"Dataset date: {data.get('date', 'N/A')}")
        logger.info(f"Version: {data.get('version', 'N/A')}")
        
        # Step 3: Sample playlist structure
        log_subsection(logger, "[3/5] Sample Playlist Structure")
        if playlists:
            sample = playlists[0]
            logger.info(f"PID: {sample.get('pid', 'N/A')}")
            logger.info(f"Name: {sample.get('name', 'N/A')}")
            logger.info(f"Num tracks: {sample.get('num_tracks', 0)}")
            logger.info(f"Num samples: {sample.get('num_samples', 0)}")
            
            if 'tracks' in sample and sample['tracks']:
                track_keys = list(sample['tracks'][0].keys())
                logger.info(f"Track attributes: {', '.join(track_keys)}")
                logger.debug(f"Sample track: {sample['tracks'][0]}")
        
        # Step 4: Metadata statistics
        log_subsection(logger, "[4/5] Metadata Analysis")
        
        has_name = sum(1 for p in playlists if 'name' in p and p['name'])
        has_tracks = sum(1 for p in playlists if 'tracks' in p and len(p['tracks']) > 0)
        
        logger.info(f"Playlists with names: {has_name:,} ({has_name/len(playlists)*100:.1f}%)")
        logger.info(f"Playlists with tracks: {has_tracks:,} ({has_tracks/len(playlists)*100:.1f}%)")
        
        # Step 5: Track distribution
        log_subsection(logger, "[5/5] Track Count Distribution")
        
        track_counts = [len(p.get('tracks', [])) for p in playlists]
        logger.info(f"Min tracks: {min(track_counts)}")
        logger.info(f"Max tracks: {max(track_counts)}")
        logger.info(f"Mean tracks: {sum(track_counts)/len(track_counts):.1f}")
        logger.info(f"Median tracks: {sorted(track_counts)[len(track_counts)//2]}")
        
        # Count unique tracks
        logger.info("Counting unique tracks...")
        all_tracks = []
        for p in playlists:
            all_tracks.extend([t.get('track_uri') for t in p.get('tracks', [])])
        
        logger.info(f"Total track instances: {len(all_tracks):,}")
        logger.info(f"Unique tracks: {len(set(all_tracks)):,}")
        
        # Success summary
        log_section(logger, "✓ DATA VERIFICATION SUCCESSFUL")
        logger.info("Challenge set is ready for analysis")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ ERROR: {type(e).__name__}: {e}")
        logger.exception("Full traceback:")
        return False

def main():
    # Set up logging
    logger = setup_logger("01_verify_data")
    
    logger.info("Starting data verification script")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Working directory: {Path.cwd()}")
    
    # Run verification
    success = verify_challenge_set(logger)
    
    if success:
        logger.info("Script completed successfully")
        sys.exit(0)
    else:
        logger.error("Script failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
