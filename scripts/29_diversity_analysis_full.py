"""
Diversity Analysis: Intra-list and Inter-list Diversity
Analyzes playlist and recommendation diversity

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
        logging.FileHandler(log_dir / f'diversity_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DiversityAnalyzer:
    """Analyze diversity metrics for playlists."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.diversity_metrics = {}
    
    def load_data(self):
        """Load data."""
        logger.info("Loading data...")
        
        tracks_df = pd.read_parquet("data/processed/tracks_full_mpd.parquet")
        playlist_features = pd.read_parquet("data/processed/playlist_features_full.parquet")
        
        logger.info(f"Loaded {len(tracks_df):,} track entries")
        logger.info(f"Loaded {len(playlist_features):,} playlist features")
        
        return tracks_df, playlist_features
    
    def calculate_intra_list_diversity(self, playlist_features):
        """Calculate intra-list diversity (within playlists)."""
        logger.info("Calculating intra-list diversity...")
        
        # Artist diversity (already calculated in features)
        artist_diversity = playlist_features['artist_diversity'].describe()
        album_diversity = playlist_features['album_diversity'].describe()
        
        self.diversity_metrics['artist_diversity_mean'] = artist_diversity['mean']
        self.diversity_metrics['artist_diversity_median'] = artist_diversity['50%']
        self.diversity_metrics['album_diversity_mean'] = album_diversity['mean']
        self.diversity_metrics['album_diversity_median'] = album_diversity['50%']
        
        logger.info(f"Artist diversity (mean): {artist_diversity['mean']:.3f}")
        logger.info(f"Album diversity (mean): {album_diversity['mean']:.3f}")
        
        return artist_diversity, album_diversity
    
    def calculate_genre_diversity(self, tracks_df, sample_size=10000):
        """Calculate genre diversity using playlist genre features."""
        logger.info("Calculating genre diversity...")
        
        try:
            genre_features = pd.read_parquet("data/processed/playlist_genre_features.parquet")
            genre_cols = [c for c in genre_features.columns if c.startswith('genre_')]
            
            # Calculate genre diversity: how many genres per playlist
            genre_features['num_genres'] = genre_features[genre_cols].sum(axis=1)
            
            self.diversity_metrics['avg_genres_per_playlist'] = genre_features['num_genres'].mean()
            self.diversity_metrics['median_genres_per_playlist'] = genre_features['num_genres'].median()
            
            # Genre distribution
            genre_dist = {}
            for genre in genre_cols:
                genre_dist[genre] = int(genre_features[genre].sum())
            
            self.diversity_metrics['genre_distribution'] = genre_dist
            
            logger.info(f"Avg genres per playlist: {self.diversity_metrics['avg_genres_per_playlist']:.2f}")
            
        except Exception as e:
            logger.warning(f"Could not calculate genre diversity: {e}")
    
    def calculate_popularity_distribution(self, tracks_df):
        """Analyze popularity distribution."""
        logger.info("Calculating popularity distribution...")
        
        track_list = tracks_df['track_uri'].tolist()
        track_counter = Counter(track_list)
        track_counts = pd.Series(dict(track_counter.most_common()))
        
        # Gini coefficient for popularity inequality
        sorted_counts = np.sort(track_counts.values)
        n = len(sorted_counts)
        cumsum = np.cumsum(sorted_counts)
        gini = (2 * np.sum((np.arange(1, n+1)) * sorted_counts)) / (n * cumsum[-1]) - (n + 1) / n
        
        self.diversity_metrics['popularity_gini'] = gini
        self.diversity_metrics['unique_tracks'] = len(track_counts)
        self.diversity_metrics['most_popular_track_count'] = int(track_counts.iloc[0])
        self.diversity_metrics['median_track_count'] = int(track_counts.median())
        
        logger.info(f"Popularity Gini coefficient: {gini:.3f}")
        logger.info(f"Most popular track appears in: {track_counts.iloc[0]:,} playlists")
        
        return track_counts
    
    def analyze_playlist_size_diversity(self, playlist_features):
        """Analyze relationship between playlist size and diversity."""
        logger.info("Analyzing playlist size vs diversity...")
        
        # Correlation between size and diversity
        size_diversity_corr = playlist_features[['num_tracks', 'artist_diversity']].corr().iloc[0, 1]
        
        self.diversity_metrics['size_diversity_correlation'] = size_diversity_corr
        
        logger.info(f"Size-diversity correlation: {size_diversity_corr:.3f}")
        
        return size_diversity_corr
    
    def save_results(self):
        """Save diversity analysis results."""
        logger.info("Saving results...")
        
        # Save as JSON
        metrics_file = self.output_dir / "diversity_metrics_full.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.diversity_metrics, f, indent=2, default=str)
        logger.info(f"Saved metrics: {metrics_file}")
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("Diversity Analysis Summary:")
        logger.info(f"{'='*60}")
        
        logger.info(f"\nIntra-list Diversity:")
        logger.info(f"  Artist diversity (mean): {self.diversity_metrics.get('artist_diversity_mean', 0):.3f}")
        logger.info(f"  Album diversity (mean): {self.diversity_metrics.get('album_diversity_mean', 0):.3f}")
        logger.info(f"  Avg genres per playlist: {self.diversity_metrics.get('avg_genres_per_playlist', 0):.2f}")
        
        logger.info(f"\nPopularity Distribution:")
        logger.info(f"  Gini coefficient: {self.diversity_metrics.get('popularity_gini', 0):.3f}")
        logger.info(f"  Unique tracks: {self.diversity_metrics.get('unique_tracks', 0):,}")
        
        logger.info(f"\nSize-Diversity Relationship:")
        logger.info(f"  Correlation: {self.diversity_metrics.get('size_diversity_correlation', 0):.3f}")
        
        logger.info(f"{'='*60}\n")

def main():
    """Main execution."""
    
    OUTPUT_DIR = "data/processed"
    
    analyzer = DiversityAnalyzer(output_dir=OUTPUT_DIR)
    
    # Load data
    tracks_df, playlist_features = analyzer.load_data()
    
    # Calculate intra-list diversity
    analyzer.calculate_intra_list_diversity(playlist_features)
    
    # Calculate genre diversity
    analyzer.calculate_genre_diversity(tracks_df)
    
    # Calculate popularity distribution
    analyzer.calculate_popularity_distribution(tracks_df)
    
    # Analyze size-diversity relationship
    analyzer.analyze_playlist_size_diversity(playlist_features)
    
    # Save results
    analyzer.save_results()
    
    logger.info("✅ Diversity analysis complete!")

if __name__ == "__main__":
    main()