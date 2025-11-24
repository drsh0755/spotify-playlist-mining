#!/usr/bin/env python3
"""
Data Comparison: Challenge Set vs MPD Slices

Checks for gaps and differences between the two datasets
to understand what additional processing may be needed.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from logger_config import setup_logger, log_section, log_subsection


def compare_datasets(logger):
    """Compare challenge set and MPD slices"""
    
    data_dir = Path(__file__).parent.parent / "data"
    
    log_section(logger, "LOADING DATASETS")
    
    # Load Challenge Set
    logger.info("Loading Challenge Set...")
    with open(data_dir / "raw" / "challenge_set.json", 'r') as f:
        challenge_data = json.load(f)
    challenge_playlists = challenge_data['playlists']
    logger.info(f"Challenge Set: {len(challenge_playlists):,} playlists")
    
    # Load MPD Slices
    logger.info("\nLoading MPD Slices...")
    mpd_dir = data_dir / "raw" / "mpd_slices"
    mpd_playlists = []
    
    slice_files = sorted(mpd_dir.glob("mpd.slice.*.json"))
    logger.info(f"Found {len(slice_files)} MPD slice files")
    
    for slice_file in slice_files:
        with open(slice_file, 'r') as f:
            slice_data = json.load(f)
        mpd_playlists.extend(slice_data.get('playlists', []))
    
    logger.info(f"MPD Slices: {len(mpd_playlists):,} playlists")
    
    # =====================
    # SCHEMA COMPARISON
    # =====================
    log_section(logger, "SCHEMA COMPARISON")
    
    # Check playlist fields
    challenge_fields = set()
    for p in challenge_playlists[:100]:
        challenge_fields.update(p.keys())
    
    mpd_fields = set()
    for p in mpd_playlists[:100]:
        mpd_fields.update(p.keys())
    
    logger.info("Playlist-level fields:")
    logger.info(f"  Challenge Set: {sorted(challenge_fields)}")
    logger.info(f"  MPD Slices:    {sorted(mpd_fields)}")
    
    # Find differences
    only_challenge = challenge_fields - mpd_fields
    only_mpd = mpd_fields - challenge_fields
    common = challenge_fields & mpd_fields
    
    logger.info(f"\n  Common fields: {sorted(common)}")
    logger.info(f"  Only in Challenge: {sorted(only_challenge)}")
    logger.info(f"  Only in MPD: {sorted(only_mpd)}")
    
    # Check track fields
    challenge_track_fields = set()
    for p in challenge_playlists[:100]:
        for t in p.get('tracks', [])[:10]:
            challenge_track_fields.update(t.keys())
    
    mpd_track_fields = set()
    for p in mpd_playlists[:100]:
        for t in p.get('tracks', [])[:10]:
            mpd_track_fields.update(t.keys())
    
    logger.info("\nTrack-level fields:")
    logger.info(f"  Challenge Set: {sorted(challenge_track_fields)}")
    logger.info(f"  MPD Slices:    {sorted(mpd_track_fields)}")
    
    only_challenge_tracks = challenge_track_fields - mpd_track_fields
    only_mpd_tracks = mpd_track_fields - challenge_track_fields
    
    logger.info(f"\n  Only in Challenge tracks: {sorted(only_challenge_tracks)}")
    logger.info(f"  Only in MPD tracks: {sorted(only_mpd_tracks)}")
    
    # =====================
    # DATA STATISTICS
    # =====================
    log_section(logger, "DATA STATISTICS COMPARISON")
    
    def get_playlist_stats(playlists, name):
        """Get statistics for a playlist set"""
        num_tracks = [len(p.get('tracks', [])) for p in playlists]
        has_name = sum(1 for p in playlists if p.get('name'))
        
        # Track counts
        all_tracks = []
        all_artists = []
        for p in playlists:
            for t in p.get('tracks', []):
                all_tracks.append(t.get('track_uri'))
                all_artists.append(t.get('artist_name'))
        
        return {
            'name': name,
            'num_playlists': len(playlists),
            'total_tracks': len(all_tracks),
            'unique_tracks': len(set(all_tracks)),
            'unique_artists': len(set(all_artists)),
            'has_name': has_name,
            'has_name_pct': has_name / len(playlists) * 100,
            'avg_tracks': np.mean(num_tracks),
            'min_tracks': min(num_tracks),
            'max_tracks': max(num_tracks),
            'median_tracks': np.median(num_tracks)
        }
    
    challenge_stats = get_playlist_stats(challenge_playlists, "Challenge Set")
    mpd_stats = get_playlist_stats(mpd_playlists, "MPD Slices")
    
    logger.info(f"\n{'Metric':<25} {'Challenge Set':>15} {'MPD Slices':>15}")
    logger.info("=" * 55)
    logger.info(f"{'Playlists':<25} {challenge_stats['num_playlists']:>15,} {mpd_stats['num_playlists']:>15,}")
    logger.info(f"{'Total tracks':<25} {challenge_stats['total_tracks']:>15,} {mpd_stats['total_tracks']:>15,}")
    logger.info(f"{'Unique tracks':<25} {challenge_stats['unique_tracks']:>15,} {mpd_stats['unique_tracks']:>15,}")
    logger.info(f"{'Unique artists':<25} {challenge_stats['unique_artists']:>15,} {mpd_stats['unique_artists']:>15,}")
    logger.info(f"{'Has name (%)':<25} {challenge_stats['has_name_pct']:>14.1f}% {mpd_stats['has_name_pct']:>14.1f}%")
    logger.info(f"{'Avg tracks/playlist':<25} {challenge_stats['avg_tracks']:>15.1f} {mpd_stats['avg_tracks']:>15.1f}")
    logger.info(f"{'Min tracks':<25} {challenge_stats['min_tracks']:>15} {mpd_stats['min_tracks']:>15}")
    logger.info(f"{'Max tracks':<25} {challenge_stats['max_tracks']:>15} {mpd_stats['max_tracks']:>15}")
    
    # =====================
    # TRACK OVERLAP
    # =====================
    log_section(logger, "TRACK OVERLAP ANALYSIS")
    
    challenge_tracks = set()
    for p in challenge_playlists:
        for t in p.get('tracks', []):
            challenge_tracks.add(t.get('track_uri'))
    
    mpd_tracks = set()
    for p in mpd_playlists:
        for t in p.get('tracks', []):
            mpd_tracks.add(t.get('track_uri'))
    
    overlap = challenge_tracks & mpd_tracks
    only_challenge = challenge_tracks - mpd_tracks
    only_mpd = mpd_tracks - challenge_tracks
    
    logger.info(f"Challenge Set unique tracks: {len(challenge_tracks):,}")
    logger.info(f"MPD Slices unique tracks:    {len(mpd_tracks):,}")
    logger.info(f"\nOverlapping tracks:          {len(overlap):,} ({len(overlap)/len(challenge_tracks)*100:.1f}% of Challenge)")
    logger.info(f"Only in Challenge:           {len(only_challenge):,}")
    logger.info(f"Only in MPD:                 {len(only_mpd):,}")
    
    # =====================
    # ARTIST OVERLAP
    # =====================
    log_section(logger, "ARTIST OVERLAP ANALYSIS")
    
    challenge_artists = set()
    for p in challenge_playlists:
        for t in p.get('tracks', []):
            challenge_artists.add(t.get('artist_name'))
    
    mpd_artists = set()
    for p in mpd_playlists:
        for t in p.get('tracks', []):
            mpd_artists.add(t.get('artist_name'))
    
    artist_overlap = challenge_artists & mpd_artists
    
    logger.info(f"Challenge Set unique artists: {len(challenge_artists):,}")
    logger.info(f"MPD Slices unique artists:    {len(mpd_artists):,}")
    logger.info(f"\nOverlapping artists:          {len(artist_overlap):,} ({len(artist_overlap)/len(challenge_artists)*100:.1f}% of Challenge)")
    
    # =====================
    # SPECIAL CHALLENGE SET FEATURES
    # =====================
    log_section(logger, "CHALLENGE SET SPECIAL FEATURES")
    
    # Check for 'num_samples' field (indicates hidden tracks)
    playlists_with_samples = [p for p in challenge_playlists if p.get('num_samples', 0) > 0]
    logger.info(f"Playlists with 'num_samples' (hidden tracks): {len(playlists_with_samples):,}")
    
    if playlists_with_samples:
        samples = [p.get('num_samples', 0) for p in playlists_with_samples]
        logger.info(f"  Average hidden tracks: {np.mean(samples):.1f}")
        logger.info(f"  Min hidden: {min(samples)}, Max hidden: {max(samples)}")
    
    # Check for playlists with no tracks (title-only)
    no_tracks = [p for p in challenge_playlists if len(p.get('tracks', [])) == 0]
    logger.info(f"\nPlaylists with NO tracks (title-only): {len(no_tracks):,}")
    
    # Check for playlists with only 1 track
    one_track = [p for p in challenge_playlists if len(p.get('tracks', [])) == 1]
    logger.info(f"Playlists with only 1 track: {len(one_track):,}")
    
    # =====================
    # PROCESSING GAPS
    # =====================
    log_section(logger, "PROCESSING GAPS IDENTIFIED")
    
    gaps = []
    
    if only_challenge_tracks:
        gaps.append("⚠️ Track fields differ - may need normalization")
    
    if len(playlists_with_samples) > 0:
        gaps.append("⚠️ Challenge set has 'num_samples' field indicating hidden tracks for evaluation")
    
    if len(no_tracks) > 0:
        gaps.append(f"⚠️ {len(no_tracks)} title-only playlists in challenge set (cold-start problem)")
    
    if len(overlap) / len(challenge_tracks) < 0.5:
        gaps.append("⚠️ Low track overlap - models trained on MPD may miss challenge tracks")
    
    if mpd_stats['avg_tracks'] != challenge_stats['avg_tracks']:
        gaps.append(f"⚠️ Different avg playlist lengths (Challenge: {challenge_stats['avg_tracks']:.1f}, MPD: {mpd_stats['avg_tracks']:.1f})")
    
    if gaps:
        logger.info("Identified gaps requiring attention:")
        for gap in gaps:
            logger.info(f"  {gap}")
    else:
        logger.info("✓ No significant gaps found - datasets are compatible")
    
    # =====================
    # RECOMMENDATIONS
    # =====================
    log_section(logger, "RECOMMENDATIONS")
    
    logger.info("For complete proposal implementation:")
    logger.info("")
    logger.info("1. CHALLENGE SET (current focus):")
    logger.info("   - Contains 10K incomplete playlists for testing")
    logger.info("   - Some playlists have hidden 'holdout' tracks (num_samples)")
    logger.info("   - Best for evaluation and prototyping")
    logger.info("")
    logger.info("2. MPD SLICES (for validation):")
    logger.info("   - Contains complete playlists (no hidden tracks)")
    logger.info("   - Better for training co-occurrence patterns")
    logger.info("   - Use to supplement/validate challenge set findings")
    logger.info("")
    logger.info("3. RECOMMENDED APPROACH:")
    logger.info("   - Train models on BOTH datasets combined")
    logger.info("   - Use challenge set for evaluation")
    logger.info("   - This increases track coverage from 66K to 169K+ tracks")
    
    return challenge_stats, mpd_stats


def main():
    logger = setup_logger("08_data_comparison")
    
    logger.info("Starting Data Comparison Analysis")
    
    try:
        compare_datasets(logger)
        log_section(logger, "✓ DATA COMPARISON COMPLETED")
        return True
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
