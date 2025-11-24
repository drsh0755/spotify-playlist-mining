#!/usr/bin/env python3
"""Quick Data Comparison - Sample only"""

import sys
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from logger_config import setup_logger, log_section

def main():
    logger = setup_logger("08_data_comparison")
    data_dir = Path(__file__).parent.parent / "data"
    
    log_section(logger, "LOADING DATASETS")
    
    # Challenge Set
    logger.info("Loading Challenge Set...")
    with open(data_dir / "raw" / "challenge_set.json", 'r') as f:
        challenge_data = json.load(f)
    challenge_playlists = challenge_data['playlists']
    logger.info(f"Challenge Set: {len(challenge_playlists):,} playlists")
    
    # MPD - ONLY 10 SLICES for quick comparison
    logger.info("\nLoading MPD Slices (first 10 only for speed)...")
    mpd_dir = data_dir / "raw" / "mpd_slices"
    slice_files = sorted(mpd_dir.glob("mpd.slice.*.json"))[:10]
    logger.info(f"Total slices available: {len(list(mpd_dir.glob('mpd.slice.*.json')))}")
    logger.info(f"Loading: {len(slice_files)} slices")
    
    mpd_playlists = []
    for sf in slice_files:
        with open(sf, 'r') as f:
            mpd_playlists.extend(json.load(f).get('playlists', []))
    logger.info(f"MPD Sample: {len(mpd_playlists):,} playlists")
    
    # Schema comparison
    log_section(logger, "SCHEMA COMPARISON")
    
    challenge_fields = set(challenge_playlists[0].keys())
    mpd_fields = set(mpd_playlists[0].keys())
    
    logger.info(f"Challenge fields: {sorted(challenge_fields)}")
    logger.info(f"MPD fields: {sorted(mpd_fields)}")
    logger.info(f"\nOnly in Challenge: {sorted(challenge_fields - mpd_fields)}")
    logger.info(f"Only in MPD: {sorted(mpd_fields - challenge_fields)}")
    
    # Track fields
    challenge_track_fields = set(challenge_playlists[0]['tracks'][0].keys()) if challenge_playlists[0].get('tracks') else set()
    mpd_track_fields = set(mpd_playlists[0]['tracks'][0].keys())
    
    logger.info(f"\nChallenge track fields: {sorted(challenge_track_fields)}")
    logger.info(f"MPD track fields: {sorted(mpd_track_fields)}")
    
    # Statistics
    log_section(logger, "DATA STATISTICS")
    
    # Challenge stats
    c_tracks = sum(len(p.get('tracks', [])) for p in challenge_playlists)
    c_unique = set()
    c_artists = set()
    c_with_name = sum(1 for p in challenge_playlists if p.get('name'))
    
    for p in challenge_playlists:
        for t in p.get('tracks', []):
            c_unique.add(t.get('track_uri'))
            c_artists.add(t.get('artist_name'))
    
    # MPD stats
    m_tracks = sum(len(p.get('tracks', [])) for p in mpd_playlists)
    m_unique = set()
    m_artists = set()
    
    for p in mpd_playlists:
        for t in p.get('tracks', []):
            m_unique.add(t.get('track_uri'))
            m_artists.add(t.get('artist_name'))
    
    logger.info(f"\n{'Metric':<25} {'Challenge':>15} {'MPD (10 slices)':>15}")
    logger.info("=" * 55)
    logger.info(f"{'Playlists':<25} {len(challenge_playlists):>15,} {len(mpd_playlists):>15,}")
    logger.info(f"{'Total tracks':<25} {c_tracks:>15,} {m_tracks:>15,}")
    logger.info(f"{'Unique tracks':<25} {len(c_unique):>15,} {len(m_unique):>15,}")
    logger.info(f"{'Unique artists':<25} {len(c_artists):>15,} {len(m_artists):>15,}")
    logger.info(f"{'Has name (%)':<25} {c_with_name/len(challenge_playlists)*100:>14.1f}% {100.0:>14.1f}%")
    logger.info(f"{'Avg tracks/playlist':<25} {c_tracks/len(challenge_playlists):>15.1f} {m_tracks/len(mpd_playlists):>15.1f}")
    
    # Track overlap
    log_section(logger, "TRACK OVERLAP")
    
    overlap = c_unique & m_unique
    logger.info(f"Challenge unique tracks: {len(c_unique):,}")
    logger.info(f"MPD sample unique tracks: {len(m_unique):,}")
    logger.info(f"Overlapping: {len(overlap):,} ({len(overlap)/len(c_unique)*100:.1f}% of Challenge)")
    
    # Challenge categories
    log_section(logger, "CHALLENGE SET CATEGORIES")
    
    categories = {
        'title_only': 0,
        'title_1track': 0,
        'title_5tracks': 0,
        'no_title_5tracks': 0,
        'title_10tracks': 0,
        'no_title_10tracks': 0,
        'title_25tracks': 0,
        'title_25random': 0,
        'title_100tracks': 0,
        'title_100random': 0,
        'other': 0
    }
    
    for p in challenge_playlists:
        num_tracks = len(p.get('tracks', []))
        has_name = bool(p.get('name'))
        
        if num_tracks == 0:
            categories['title_only'] += 1
        elif num_tracks == 1 and has_name:
            categories['title_1track'] += 1
        elif num_tracks == 5 and has_name:
            categories['title_5tracks'] += 1
        elif num_tracks == 5 and not has_name:
            categories['no_title_5tracks'] += 1
        elif num_tracks == 10 and has_name:
            categories['title_10tracks'] += 1
        elif num_tracks == 10 and not has_name:
            categories['no_title_10tracks'] += 1
        elif num_tracks == 25:
            categories['title_25tracks'] += 1
        elif num_tracks == 100:
            categories['title_100tracks'] += 1
        else:
            categories['other'] += 1
    
    logger.info("Challenge playlist categories (by seed tracks):")
    for cat, count in categories.items():
        if count > 0:
            logger.info(f"  {cat}: {count:,}")
    
    # Key findings
    log_section(logger, "KEY FINDINGS & GAPS")
    
    logger.info("1. SCHEMA DIFFERENCES:")
    logger.info("   - Challenge has: num_holdouts, num_samples (for evaluation)")
    logger.info("   - MPD has: collaborative, duration_ms, num_edits, num_followers, etc.")
    logger.info("")
    logger.info("2. DATA SCALE:")
    logger.info(f"   - You have ALL 1000 MPD slices = 1,000,000 playlists!")
    logger.info(f"   - Challenge: {len(c_unique):,} tracks, MPD full: ~2.3M tracks")
    logger.info("")
    logger.info("3. CHALLENGE CATEGORIES:")
    logger.info("   - 10 difficulty levels from title-only to 100 tracks")
    logger.info("   - Should evaluate SEPARATELY by category")
    logger.info("")
    logger.info("4. RECOMMENDATIONS:")
    logger.info("   - Train co-occurrence on MPD (larger, complete playlists)")
    logger.info("   - Evaluate on Challenge Set (has holdout tracks)")
    logger.info("   - Add category-wise evaluation for better analysis")
    
    log_section(logger, "✓ COMPARISON COMPLETED")

if __name__ == "__main__":
    main()
