"""
Evaluation Metrics: R-precision, NDCG, and Coverage
Evaluates recommendation system performance

Author: Adarsh Singh
Date: November 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
from collections import Counter

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EvaluationMetrics:
    """Calculate evaluation metrics for recommendation systems."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {}
    
    def load_data(self):
        """Load playlist and track data."""
        logger.info("Loading data...")
        
        tracks_df = pd.read_parquet("data/processed/tracks_full_mpd.parquet")
        playlists_df = pd.read_parquet("data/processed/playlists_full_mpd.parquet")
        
        logger.info(f"Loaded {len(tracks_df):,} track entries from {len(playlists_df):,} playlists")
        
        return tracks_df, playlists_df
    
    def calculate_r_precision(self, tracks_df, sample_size=5000):
        """Calculate R-precision using popularity baseline."""
        logger.info(f"Calculating R-precision on {sample_size} playlists...")
        
        # Get track popularity using Counter (more reliable)
        track_list = tracks_df['track_uri'].tolist()
        track_counter = Counter(track_list)
        top_500 = set([track for track, count in track_counter.most_common(500)])
        
        # Group tracks by playlist
        playlist_tracks = tracks_df.groupby('pid')['track_uri'].apply(set)
        
        # Sample playlists
        np.random.seed(42)
        sample_indices = np.random.choice(len(playlist_tracks), min(sample_size, len(playlist_tracks)), replace=False)
        sample_playlists = playlist_tracks.iloc[sample_indices]
        
        # Calculate R-precision for each playlist
        r_precisions = []
        for pid, tracks in sample_playlists.items():
            if len(tracks) >= 5:
                recommended = top_500
                relevant = tracks
                intersection = recommended & relevant
                
                r_precision = len(intersection) / len(relevant) if relevant else 0
                r_precisions.append(r_precision)
        
        self.metrics['r_precision_mean'] = float(np.mean(r_precisions))
        self.metrics['r_precision_median'] = float(np.median(r_precisions))
        self.metrics['r_precision_std'] = float(np.std(r_precisions))
        self.metrics['playlists_evaluated'] = len(r_precisions)
        
        logger.info(f"R-precision (mean): {self.metrics['r_precision_mean']:.4f}")
        logger.info(f"R-precision (median): {self.metrics['r_precision_median']:.4f}")
        
        return r_precisions
    
    def calculate_ndcg(self, tracks_df, sample_size=1000):
        """Calculate NDCG@10."""
        logger.info(f"Calculating NDCG@10 on {sample_size} playlists...")
        
        # Get track popularity ranking using Counter
        track_list = tracks_df['track_uri'].tolist()
        track_counter = Counter(track_list)
        track_ranks = {track: rank for rank, (track, count) in enumerate(track_counter.most_common())}
        
        # Group tracks by playlist
        playlist_tracks = tracks_df.groupby('pid')['track_uri'].apply(list)
        
        # Sample playlists
        np.random.seed(42)
        sample_indices = np.random.choice(len(playlist_tracks), min(sample_size, len(playlist_tracks)), replace=False)
        sample_playlists = playlist_tracks.iloc[sample_indices]
        
        ndcg_scores = []
        k = 10
        
        for pid, tracks in sample_playlists.items():
            if len(tracks) >= k:
                # Get top k tracks by popularity as recommendations
                recommended = sorted(tracks[:k], key=lambda t: track_ranks.get(t, float('inf')))[:k]
                
                # Calculate DCG
                dcg = 0
                for i, track in enumerate(recommended):
                    relevance = 1 if track in tracks else 0
                    dcg += relevance / np.log2(i + 2)
                
                # Calculate ideal DCG
                idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(tracks))))
                
                # NDCG
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcg_scores.append(ndcg)
        
        self.metrics['ndcg_mean'] = float(np.mean(ndcg_scores))
        self.metrics['ndcg_median'] = float(np.median(ndcg_scores))
        
        logger.info(f"NDCG@10 (mean): {self.metrics['ndcg_mean']:.4f}")
        logger.info(f"NDCG@10 (median): {self.metrics['ndcg_median']:.4f}")
        
        return ndcg_scores
    
    def calculate_coverage(self, tracks_df):
        """Calculate recommendation coverage."""
        logger.info("Calculating coverage metrics...")
        
        # Track statistics
        total_tracks = tracks_df['track_uri'].nunique()
        total_artists = tracks_df['artist_uri'].nunique()
        total_albums = tracks_df['album_uri'].nunique()
        
        # Popularity distribution using Counter
        track_list = tracks_df['track_uri'].tolist()
        track_counter = Counter(track_list)
        
        self.metrics['total_unique_tracks'] = int(total_tracks)
        self.metrics['total_unique_artists'] = int(total_artists)
        self.metrics['total_unique_albums'] = int(total_albums)
        self.metrics['tracks_in_1_playlist'] = int(sum(1 for count in track_counter.values() if count == 1))
        self.metrics['tracks_in_10plus_playlists'] = int(sum(1 for count in track_counter.values() if count >= 10))
        self.metrics['tracks_in_100plus_playlists'] = int(sum(1 for count in track_counter.values() if count >= 100))
        self.metrics['tracks_in_1000plus_playlists'] = int(sum(1 for count in track_counter.values() if count >= 1000))
        
        logger.info(f"Total unique tracks: {total_tracks:,}")
        logger.info(f"Total unique artists: {total_artists:,}")
        logger.info(f"Tracks in 1000+ playlists: {self.metrics['tracks_in_1000plus_playlists']:,}")
        
        return self.metrics
    
    def save_results(self):
        """Save evaluation metrics."""
        logger.info("Saving results...")
        
        # Save as JSON
        metrics_file = self.output_dir / "evaluation_metrics_full.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Saved metrics: {metrics_file}")
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("Evaluation Metrics Summary:")
        logger.info(f"{'='*60}")
        logger.info(f"\nPerformance Metrics:")
        logger.info(f"  R-precision (mean): {self.metrics.get('r_precision_mean', 0):.4f}")
        logger.info(f"  R-precision (median): {self.metrics.get('r_precision_median', 0):.4f}")
        logger.info(f"  NDCG@10 (mean): {self.metrics.get('ndcg_mean', 0):.4f}")
        logger.info(f"  NDCG@10 (median): {self.metrics.get('ndcg_median', 0):.4f}")
        
        logger.info(f"\nCoverage Metrics:")
        logger.info(f"  Total unique tracks: {self.metrics.get('total_unique_tracks', 0):,}")
        logger.info(f"  Tracks in 1000+ playlists: {self.metrics.get('tracks_in_1000plus_playlists', 0):,}")
        logger.info(f"  Playlists evaluated: {self.metrics.get('playlists_evaluated', 0):,}")
        logger.info(f"{'='*60}\n")

def main():
    """Main execution."""
    
    OUTPUT_DIR = "data/processed"
    
    evaluator = EvaluationMetrics(output_dir=OUTPUT_DIR)
    
    # Load data
    tracks_df, playlists_df = evaluator.load_data()
    
    # Calculate R-precision
    evaluator.calculate_r_precision(tracks_df, sample_size=5000)
    
    # Calculate NDCG
    evaluator.calculate_ndcg(tracks_df, sample_size=1000)
    
    # Calculate coverage
    evaluator.calculate_coverage(tracks_df)
    
    # Save results
    evaluator.save_results()
    
    logger.info("✅ Evaluation metrics complete!")

if __name__ == "__main__":
    main()