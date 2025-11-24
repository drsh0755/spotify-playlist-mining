"""
Build co-occurrence matrix incrementally for 1M playlists.
Memory-efficient implementation using sparse matrices and batch processing.

Author: Adarsh Singh
Date: November 2024
"""

import pandas as pd
import numpy as np
from scipy import sparse
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
import pickle
import gc
import psutil
from collections import defaultdict

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'cooccurrence_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IncrementalCooccurrence:
    """Build co-occurrence matrix incrementally for large-scale playlist data."""
    
    def __init__(self, output_dir, min_occurrences=1000):
        """
        Args:
            output_dir: Directory to save results
            min_occurrences: Minimum track occurrences to include in matrix
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_occurrences = min_occurrences
        
        # Track statistics
        self.track_counts = defaultdict(int)
        self.track_to_idx = {}
        self.idx_to_track = {}
        
        # Co-occurrence accumulator (dictionary for memory efficiency)
        self.cooccurrence_dict = defaultdict(int)
        
        self.stats = {
            'total_playlists': 0,
            'total_pairs': 0,
            'unique_tracks': 0,
            'filtered_tracks': 0
        }
    
    def get_memory_usage(self):
        """Get current memory usage in GB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024 / 1024
    
    def load_tracks_data(self, tracks_file, use_parquet=True):
        """Load tracks data efficiently."""
        logger.info(f"Loading tracks data from {tracks_file}")
        
        if use_parquet and Path(str(tracks_file).replace('.csv', '.parquet')).exists():
            tracks_file = Path(str(tracks_file).replace('.csv', '.parquet'))
            df = pd.read_parquet(tracks_file)
            logger.info("Loaded from Parquet (faster)")
        else:
            df = pd.read_csv(tracks_file)
            logger.info("Loaded from CSV")
        
        logger.info(f"Loaded {len(df):,} track entries from {df['pid'].nunique():,} playlists")
        return df
    
    def count_track_occurrences(self, tracks_df):
        """Count how many playlists each track appears in."""
        logger.info("Counting track occurrences...")
        
        track_counts = tracks_df.groupby('track_uri')['pid'].nunique()
        self.track_counts = track_counts.to_dict()
        
        # Filter tracks by minimum occurrences
        valid_tracks = [track for track, count in self.track_counts.items() 
                       if count >= self.min_occurrences]
        
        self.stats['unique_tracks'] = len(self.track_counts)
        self.stats['filtered_tracks'] = len(valid_tracks)
        
        logger.info(f"Total unique tracks: {self.stats['unique_tracks']:,}")
        logger.info(f"Tracks with >={self.min_occurrences} occurrences: {self.stats['filtered_tracks']:,}")
        
        # Create track indexing
        self.track_to_idx = {track: idx for idx, track in enumerate(sorted(valid_tracks))}
        self.idx_to_track = {idx: track for track, idx in self.track_to_idx.items()}
        
        return valid_tracks
    
    def build_cooccurrence_incremental(self, tracks_df, batch_size=10000):
        """Build co-occurrence matrix incrementally by processing playlists in batches."""
        
        logger.info("Building co-occurrence matrix...")
        start_time = datetime.now()
        
        # Filter tracks to only include valid ones
        tracks_df = tracks_df[tracks_df['track_uri'].isin(self.track_to_idx.keys())].copy()
        logger.info(f"Processing {len(tracks_df):,} track entries")
        
        # Group by playlist
        grouped = tracks_df.groupby('pid')['track_uri'].apply(list)
        playlists = list(grouped.items())
        
        self.stats['total_playlists'] = len(playlists)
        logger.info(f"Processing {len(playlists):,} playlists")
        
        # Process playlists in batches
        for batch_start in tqdm(range(0, len(playlists), batch_size), 
                               desc="Building co-occurrence"):
            
            batch_end = min(batch_start + batch_size, len(playlists))
            batch = playlists[batch_start:batch_end]
            
            # Process each playlist in batch
            for pid, tracks in batch:
                # Convert to indices
                track_indices = [self.track_to_idx[t] for t in tracks if t in self.track_to_idx]
                
                # Generate pairs (only upper triangle to avoid duplicates)
                for i in range(len(track_indices)):
                    for j in range(i + 1, len(track_indices)):
                        idx1, idx2 = track_indices[i], track_indices[j]
                        # Store with smaller index first
                        if idx1 > idx2:
                            idx1, idx2 = idx2, idx1
                        self.cooccurrence_dict[(idx1, idx2)] += 1
                        self.stats['total_pairs'] += 1
            
            # Log progress
            if (batch_start // batch_size) % 10 == 0:
                memory_gb = self.get_memory_usage()
                logger.info(f"Processed {batch_end:,}/{len(playlists):,} playlists | "
                          f"Co-occurrences: {len(self.cooccurrence_dict):,} | "
                          f"Memory: {memory_gb:.2f}GB")
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Co-occurrence building completed in {total_time/60:.2f} minutes")
        logger.info(f"Total unique co-occurrence pairs: {len(self.cooccurrence_dict):,}")
    
    def convert_to_sparse_matrix(self):
        """Convert dictionary to sparse matrix."""
        logger.info("Converting to sparse matrix format...")
        
        n_tracks = len(self.track_to_idx)
        
        # Prepare data for sparse matrix
        rows = []
        cols = []
        data = []
        
        for (i, j), count in tqdm(self.cooccurrence_dict.items(), 
                                  desc="Building sparse matrix"):
            # Add both directions (symmetric matrix)
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([count, count])
        
        # Create sparse matrix
        cooccurrence_matrix = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n_tracks, n_tracks),
            dtype=np.int32
        )
        
        # Add diagonal (track with itself = number of playlists it appears in)
        for track, idx in self.track_to_idx.items():
            cooccurrence_matrix[idx, idx] = self.track_counts[track]
        
        logger.info(f"Sparse matrix shape: {cooccurrence_matrix.shape}")
        logger.info(f"Non-zero elements: {cooccurrence_matrix.nnz:,}")
        logger.info(f"Sparsity: {(1 - cooccurrence_matrix.nnz / (n_tracks ** 2)) * 100:.4f}%")
        
        return cooccurrence_matrix
    
    def save_results(self, cooccurrence_matrix):
        """Save co-occurrence matrix and metadata."""
        logger.info("Saving results...")
        
        # Save sparse matrix
        matrix_file = self.output_dir / "cooccurrence_matrix_full.npz"
        sparse.save_npz(matrix_file, cooccurrence_matrix)
        logger.info(f"Saved sparse matrix: {matrix_file}")
        
        # Save track mappings
        mappings_file = self.output_dir / "track_mappings.pkl"
        mappings = {
            'track_to_idx': self.track_to_idx,
            'idx_to_track': self.idx_to_track,
            'track_counts': self.track_counts
        }
        with open(mappings_file, 'wb') as f:
            pickle.dump(mappings, f)
        logger.info(f"Saved track mappings: {mappings_file}")
        
        # Save statistics
        import json
        stats_file = self.output_dir / "cooccurrence_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        logger.info(f"Saved statistics: {stats_file}")
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info(f"Co-occurrence Matrix Summary:")
        logger.info(f"Matrix shape: {cooccurrence_matrix.shape}")
        logger.info(f"Non-zero entries: {cooccurrence_matrix.nnz:,}")
        logger.info(f"Total playlists processed: {self.stats['total_playlists']:,}")
        logger.info(f"Unique tracks included: {self.stats['filtered_tracks']:,}")
        logger.info(f"Matrix size on disk: {matrix_file.stat().st_size / 1024**2:.2f} MB")
        logger.info(f"{'='*60}\n")
    
    def build_from_tracks_file(self, tracks_file):
        """Complete pipeline: load data, build co-occurrence, save."""
        
        # Load tracks
        tracks_df = self.load_tracks_data(tracks_file)
        
        # Count occurrences and filter
        valid_tracks = self.count_track_occurrences(tracks_df)
        
        # Build co-occurrence incrementally
        self.build_cooccurrence_incremental(tracks_df)
        
        # Convert to sparse matrix
        cooccurrence_matrix = self.convert_to_sparse_matrix()
        
        # Save results
        self.save_results(cooccurrence_matrix)
        
        return cooccurrence_matrix

def main():
    """Main execution function."""
    
    # Configuration - RELATIVE PATHS FOR MAC
    TRACKS_FILE = "data/processed/tracks_full_mpd.csv"
    OUTPUT_DIR = "data/processed"
    MIN_OCCURRENCES = 1000  # Only tracks in 1000+ playlists (very popular hits)
    
    # Build co-occurrence matrix
    builder = IncrementalCooccurrence(
        output_dir=OUTPUT_DIR,
        min_occurrences=MIN_OCCURRENCES
    )
    
    cooccurrence_matrix = builder.build_from_tracks_file(TRACKS_FILE)
    
    logger.info("✅ Co-occurrence matrix building complete!")

if __name__ == "__main__":
    main()