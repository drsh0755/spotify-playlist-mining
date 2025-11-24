#!/usr/bin/env python3
"""
Data Preprocessing Pipeline

Loads, processes, and caches Spotify playlist data.
Creates indexed data structures for efficient analysis.

Usage:
    python scripts/03_data_preprocessing_pipeline.py
    
    # Or with screen:
    ./scripts/run_with_screen.sh 03_data_preprocessing_pipeline.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from logger_config import setup_logger, log_section, log_subsection
from data_loader import SpotifyDataLoader


def process_challenge_set(logger):
    """Process challenge set"""
    log_section(logger, "PROCESSING CHALLENGE SET")
    
    loader = SpotifyDataLoader()
    
    # Load data (will cache automatically)
    playlists = loader.load_challenge_set(use_cache=False)
    logger.info(f"Loaded {len(playlists):,} playlists")
    
    # Get DataFrames
    log_subsection(logger, "Creating DataFrames")
    tracks_df = loader.get_tracks_dataframe()
    playlists_df = loader.get_playlists_dataframe()
    features_df = loader.get_playlist_features()
    
    logger.info(f"Tracks DataFrame: {tracks_df.shape}")
    logger.info(f"Playlists DataFrame: {playlists_df.shape}")
    logger.info(f"Features DataFrame: {features_df.shape}")
    
    # Save DataFrames
    output_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tracks_df.to_csv(output_dir / "tracks.csv", index=False)
    playlists_df.to_csv(output_dir / "playlists.csv", index=False)
    features_df.to_csv(output_dir / "playlist_features.csv", index=False)
    
    logger.info(f"✓ DataFrames saved to: {output_dir}")
    
    # Build co-occurrence matrix
    log_subsection(logger, "Building Co-occurrence Matrix")
    cooccur_matrix, metadata = loader.get_cooccurrence_matrix(min_support=2)
    
    logger.info(f"Matrix shape: {cooccur_matrix.shape}")
    logger.info(f"Matrix density: {cooccur_matrix.nnz / (cooccur_matrix.shape[0]**2) * 100:.4f}%")
    logger.info(f"Tracks in matrix: {metadata['n_tracks']:,}")
    
    # Save co-occurrence matrix
    import pickle
    cooccur_path = output_dir / "cooccurrence_matrix.pkl"
    with open(cooccur_path, 'wb') as f:
        pickle.dump({
            'matrix': cooccur_matrix,
            'metadata': metadata
        }, f)
    logger.info(f"✓ Co-occurrence matrix saved to: {cooccur_path}")
    
    return loader


def process_mpd_sample(logger, n_slices=10):
    """Process sample of MPD dataset"""
    log_section(logger, f"PROCESSING MPD SAMPLE ({n_slices} slices)")
    
    loader = SpotifyDataLoader()
    
    # Load MPD slices
    playlists = loader.load_mpd_slices(n_slices=n_slices, use_cache=False)
    logger.info(f"Loaded {len(playlists):,} playlists from {n_slices} slices")
    
    # Get DataFrames
    log_subsection(logger, "Creating DataFrames")
    tracks_df = loader.get_tracks_dataframe()
    playlists_df = loader.get_playlists_dataframe()
    
    logger.info(f"Tracks DataFrame: {tracks_df.shape}")
    logger.info(f"Playlists DataFrame: {playlists_df.shape}")
    
    # Save DataFrames
    output_dir = Path(__file__).parent.parent / "data" / "processed"
    
    tracks_df.to_csv(output_dir / f"tracks_mpd_{n_slices}.csv", index=False)
    playlists_df.to_csv(output_dir / f"playlists_mpd_{n_slices}.csv", index=False)
    
    logger.info(f"✓ MPD DataFrames saved to: {output_dir}")
    
    return loader


def generate_summary_report(logger, loader):
    """Generate summary report"""
    log_section(logger, "SUMMARY REPORT")
    
    logger.info(f"Total playlists processed: {len(loader.playlists):,}")
    logger.info(f"Unique tracks: {len(loader.track_to_id):,}")
    logger.info(f"Unique artists: {len(set(loader.track_to_artist.values())):,}")
    
    # Track statistics
    counts = list(loader.track_counter.values())
    import numpy as np
    logger.info(f"\nTrack appearance statistics:")
    logger.info(f"  Min: {min(counts)}")
    logger.info(f"  Max: {max(counts)}")
    logger.info(f"  Mean: {np.mean(counts):.2f}")
    logger.info(f"  Median: {np.median(counts):.0f}")
    
    # Top tracks
    logger.info(f"\nTop 5 most popular tracks:")
    for i, (track_uri, count) in enumerate(loader.track_counter.most_common(5), 1):
        artist = loader.track_to_artist.get(track_uri, 'Unknown')
        logger.info(f"  {i}. {artist} - {count} playlists")


def main():
    """Main execution"""
    logger = setup_logger("03_data_preprocessing_pipeline")
    
    log_section(logger, "DATA PREPROCESSING PIPELINE")
    logger.info("Starting data preprocessing")
    
    try:
        # Process challenge set
        loader = process_challenge_set(logger)
        generate_summary_report(logger, loader)
        
        # Optionally process MPD sample
        logger.info("\n" + "="*70)
        response = input("Process MPD sample (10 slices ~10K playlists)? [y/N]: ").strip().lower()
        if response == 'y':
            loader_mpd = process_mpd_sample(logger, n_slices=10)
            generate_summary_report(logger, loader_mpd)
        
        log_section(logger, "✓ PREPROCESSING COMPLETED SUCCESSFULLY")
        logger.info("All data processed and cached")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error during preprocessing: {e}")
        logger.exception("Full traceback:")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
