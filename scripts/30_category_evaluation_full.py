"""
Category-wise Evaluation: Performance by Genre and Playlist Type
Evaluates recommendation performance across different categories

Author: Adarsh Singh
Date: November 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'category_eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CategoryEvaluator:
    """Evaluate performance by category (genre, size, etc.)."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.category_results = {}
    
    def load_data(self):
        """Load data."""
        logger.info("Loading data...")
        
        tracks_df = pd.read_parquet("data/processed/tracks_full_mpd.parquet")
        playlist_features = pd.read_parquet("data/processed/playlist_features_full.parquet")
        genre_features = pd.read_parquet("data/processed/playlist_genre_features.parquet")
        
        logger.info(f"Loaded {len(tracks_df):,} track entries")
        logger.info(f"Loaded {len(playlist_features):,} playlists")
        
        return tracks_df, playlist_features, genre_features
    
    def evaluate_by_genre(self, tracks_df, genre_features):
        """Evaluate performance by genre."""
        logger.info("Evaluating by genre...")
        
        genre_cols = [c for c in genre_features.columns if c.startswith('genre_')]
        
        genre_results = {}
        
        for genre_col in genre_cols:
            genre_name = genre_col.replace('genre_', '')
            genre_pids = genre_features[genre_features[genre_col] == 1]['pid'].values
            
            if len(genre_pids) < 100:
                continue
            
            # Get tracks for this genre
            genre_tracks = tracks_df[tracks_df['pid'].isin(genre_pids)]
            
            # Calculate metrics
            avg_tracks_per_playlist = len(genre_tracks) / len(genre_pids)
            unique_tracks = genre_tracks['track_uri'].nunique()
            unique_artists = genre_tracks['artist_uri'].nunique()
            
            genre_results[genre_name] = {
                'num_playlists': int(len(genre_pids)),
                'avg_tracks_per_playlist': float(avg_tracks_per_playlist),
                'unique_tracks': int(unique_tracks),
                'unique_artists': int(unique_artists)
            }
            
            logger.info(f"  {genre_name}: {len(genre_pids):,} playlists, {unique_tracks:,} unique tracks")
        
        self.category_results['by_genre'] = genre_results
        
        return genre_results
    
    def evaluate_by_size(self, tracks_df, playlist_features):
        """Evaluate performance by playlist size."""
        logger.info("Evaluating by playlist size...")
        
        # Define size categories
        playlist_features['size_category'] = pd.cut(
            playlist_features['num_tracks'],
            bins=[0, 10, 25, 50, 100, float('inf')],
            labels=['tiny', 'small', 'medium', 'large', 'huge']
        )
        
        size_results = {}
        
        for category in ['tiny', 'small', 'medium', 'large', 'huge']:
            category_pids = playlist_features[playlist_features['size_category'] == category]['pid'].values
            
            if len(category_pids) == 0:
                continue
            
            # Get tracks for this size category
            category_tracks = tracks_df[tracks_df['pid'].isin(category_pids)]
            
            # Calculate diversity
            category_playlists = playlist_features[playlist_features['size_category'] == category]
            avg_artist_diversity = category_playlists['artist_diversity'].mean()
            
            size_results[category] = {
                'num_playlists': int(len(category_pids)),
                'avg_tracks': float(category_playlists['num_tracks'].mean()),
                'avg_artist_diversity': float(avg_artist_diversity),
                'unique_tracks': int(category_tracks['track_uri'].nunique())
            }
            
            logger.info(f"  {category}: {len(category_pids):,} playlists, diversity: {avg_artist_diversity:.3f}")
        
        self.category_results['by_size'] = size_results
        
        return size_results
    
    def evaluate_by_popularity(self, playlist_features):
        """Evaluate by playlist popularity (followers)."""
        logger.info("Evaluating by playlist popularity...")
        
        # Define popularity categories
        playlist_features['popularity_category'] = pd.cut(
            playlist_features['num_followers'],
            bins=[-1, 0, 1, 10, 100, float('inf')],
            labels=['no_followers', 'few_followers', 'some_followers', 'many_followers', 'viral']
        )
        
        popularity_results = {}
        
        for category in ['no_followers', 'few_followers', 'some_followers', 'many_followers', 'viral']:
            category_playlists = playlist_features[playlist_features['popularity_category'] == category]
            
            if len(category_playlists) == 0:
                continue
            
            popularity_results[category] = {
                'num_playlists': int(len(category_playlists)),
                'avg_tracks': float(category_playlists['num_tracks'].mean()),
                'avg_artist_diversity': float(category_playlists['artist_diversity'].mean()),
                'avg_followers': float(category_playlists['num_followers'].mean())
            }
            
            logger.info(f"  {category}: {len(category_playlists):,} playlists")
        
        self.category_results['by_popularity'] = popularity_results
        
        return popularity_results
    
    def save_results(self):
        """Save category evaluation results."""
        logger.info("Saving results...")
        
        # Save as JSON
        results_file = self.output_dir / "category_evaluation_full.json"
        with open(results_file, 'w') as f:
            json.dump(self.category_results, f, indent=2)
        logger.info(f"Saved results: {results_file}")
        
        # Convert to CSV for easy viewing
        # Genre results
        if 'by_genre' in self.category_results:
            genre_df = pd.DataFrame(self.category_results['by_genre']).T
            genre_df.to_csv(self.output_dir / "category_by_genre_full.csv")
        
        # Size results
        if 'by_size' in self.category_results:
            size_df = pd.DataFrame(self.category_results['by_size']).T
            size_df.to_csv(self.output_dir / "category_by_size_full.csv")
        
        # Popularity results
        if 'by_popularity' in self.category_results:
            pop_df = pd.DataFrame(self.category_results['by_popularity']).T
            pop_df.to_csv(self.output_dir / "category_by_popularity_full.csv")
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("Category Evaluation Summary:")
        logger.info(f"{'='*60}")
        
        if 'by_genre' in self.category_results:
            logger.info(f"\nGenre categories evaluated: {len(self.category_results['by_genre'])}")
        
        if 'by_size' in self.category_results:
            logger.info(f"Size categories evaluated: {len(self.category_results['by_size'])}")
        
        if 'by_popularity' in self.category_results:
            logger.info(f"Popularity categories evaluated: {len(self.category_results['by_popularity'])}")
        
        logger.info(f"{'='*60}\n")

def main():
    """Main execution."""
    
    OUTPUT_DIR = "data/processed"
    
    evaluator = CategoryEvaluator(output_dir=OUTPUT_DIR)
    
    # Load data
    tracks_df, playlist_features, genre_features = evaluator.load_data()
    
    # Evaluate by genre
    evaluator.evaluate_by_genre(tracks_df, genre_features)
    
    # Evaluate by size
    evaluator.evaluate_by_size(tracks_df, playlist_features)
    
    # Evaluate by popularity
    evaluator.evaluate_by_popularity(playlist_features)
    
    # Save results
    evaluator.save_results()
    
    logger.info("✅ Category evaluation complete!")

if __name__ == "__main__":
    main()