"""
Genre Cross-Pollination Analysis
Analyze how genres mix in playlists

Author: Adarsh Singh
Date: November 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
from itertools import combinations

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'genre_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GenreCrossPollination:
    """Analyze genre mixing patterns in playlists."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.genres = [
            'workout', 'party', 'chill', 'rock', 'hip_hop', 
            'country', 'pop', 'indie', 'classical', 'jazz', 
            'electronic', 'latin', 'mood'
        ]
    
    def load_data(self):
        """Load playlist genre features."""
        logger.info("Loading data...")
        
        genre_df = pd.read_parquet("data/processed/playlist_genre_features.parquet")
        logger.info(f"Loaded {len(genre_df):,} playlists with genre features")
        
        return genre_df
    
    def analyze_genre_distribution(self, genre_df):
        """Analyze single genre distribution."""
        logger.info("\n" + "="*60)
        logger.info("Single Genre Distribution")
        logger.info("="*60)
        
        genre_counts = {}
        for genre in self.genres:
            if genre in genre_df.columns:
                count = genre_df[genre].sum()
                pct = (count / len(genre_df)) * 100
                genre_counts[genre] = {
                    'count': int(count),
                    'percentage': float(pct)
                }
                logger.info(f"  {genre:15s}: {count:>8,} playlists ({pct:>5.2f}%)")
        
        return genre_counts
    
    def analyze_multi_genre_playlists(self, genre_df):
        """Analyze playlists with multiple genres."""
        logger.info("\n" + "="*60)
        logger.info("Multi-Genre Playlist Analysis")
        logger.info("="*60)
        
        # Count genres per playlist
        genre_cols = [g for g in self.genres if g in genre_df.columns]
        genre_df['num_genres'] = genre_df[genre_cols].sum(axis=1)
        
        # Distribution
        genre_count_dist = genre_df['num_genres'].value_counts().sort_index()
        
        logger.info("\nNumber of Genres per Playlist:")
        for n_genres, count in genre_count_dist.items():
            pct = (count / len(genre_df)) * 100
            logger.info(f"  {n_genres} genre(s): {count:>8,} playlists ({pct:>5.2f}%)")
        
        multi_genre_stats = {
            'mean_genres_per_playlist': float(genre_df['num_genres'].mean()),
            'median_genres_per_playlist': float(genre_df['num_genres'].median()),
            'max_genres': int(genre_df['num_genres'].max()),
            'multi_genre_playlists': int((genre_df['num_genres'] > 1).sum()),
            'multi_genre_percentage': float((genre_df['num_genres'] > 1).sum() / len(genre_df) * 100)
        }
        
        logger.info(f"\nMean Genres per Playlist: {multi_genre_stats['mean_genres_per_playlist']:.2f}")
        logger.info(f"Playlists with 2+ Genres: {multi_genre_stats['multi_genre_playlists']:,} ({multi_genre_stats['multi_genre_percentage']:.2f}%)")
        
        return multi_genre_stats
    
    def analyze_genre_pairs(self, genre_df):
        """Analyze which genres co-occur most frequently."""
        logger.info("\n" + "="*60)
        logger.info("Genre Co-occurrence Analysis")
        logger.info("="*60)
        
        genre_cols = [g for g in self.genres if g in genre_df.columns]
        
        # Calculate co-occurrence matrix
        cooccurrence = {}
        for g1, g2 in combinations(genre_cols, 2):
            both = ((genre_df[g1] == 1) & (genre_df[g2] == 1)).sum()
            if both > 0:
                cooccurrence[(g1, g2)] = int(both)
        
        # Sort by frequency
        sorted_pairs = sorted(cooccurrence.items(), key=lambda x: x[1], reverse=True)
        
        logger.info("\nTop 20 Genre Pair Co-occurrences:")
        pair_list = []
        for rank, ((g1, g2), count) in enumerate(sorted_pairs[:20], 1):
            pct = (count / len(genre_df)) * 100
            pair_list.append({
                'rank': rank,
                'genre1': g1,
                'genre2': g2,
                'count': count,
                'percentage': float(pct)
            })
            logger.info(f"  {rank:2d}. {g1:12s} + {g2:12s}: {count:>6,} ({pct:>5.2f}%)")
        
        return pair_list
    
    def analyze_genre_exclusivity(self, genre_df):
        """Analyze genre exclusivity (single genre playlists)."""
        logger.info("\n" + "="*60)
        logger.info("Genre Exclusivity Analysis")
        logger.info("="*60)
        
        genre_cols = [g for g in self.genres if g in genre_df.columns]
        genre_df['num_genres'] = genre_df[genre_cols].sum(axis=1)
        
        exclusivity_stats = {}
        logger.info("\nSingle-Genre Playlists:")
        for genre in genre_cols:
            # Playlists with ONLY this genre
            exclusive = ((genre_df[genre] == 1) & (genre_df['num_genres'] == 1)).sum()
            total_genre = (genre_df[genre] == 1).sum()
            
            if total_genre > 0:
                exclusivity_pct = (exclusive / total_genre) * 100
                exclusivity_stats[genre] = {
                    'exclusive_count': int(exclusive),
                    'total_count': int(total_genre),
                    'exclusivity_percentage': float(exclusivity_pct)
                }
                logger.info(f"  {genre:15s}: {exclusive:>6,} / {total_genre:>6,} ({exclusivity_pct:>5.2f}% exclusive)")
        
        return exclusivity_stats
    
    def analyze_genre_diversity_correlation(self, genre_df):
        """Analyze correlation between genre count and playlist size."""
        logger.info("\n" + "="*60)
        logger.info("Genre Diversity Correlation")
        logger.info("="*60)
        
        if 'num_tracks' in genre_df.columns:
            genre_cols = [g for g in self.genres if g in genre_df.columns]
            genre_df['num_genres'] = genre_df[genre_cols].sum(axis=1)
            
            correlation = genre_df[['num_genres', 'num_tracks']].corr().iloc[0, 1]
            
            logger.info(f"Correlation (Genre Count vs Playlist Size): {correlation:.4f}")
            
            return {'genre_size_correlation': float(correlation)}
        else:
            logger.warning("num_tracks column not found")
            return {}
    
    def save_results(self, genre_dist, multi_genre_stats, genre_pairs, exclusivity_stats, correlation_stats):
        """Save results."""
        logger.info("\nSaving results...")
        
        results = {
            'genre_distribution': genre_dist,
            'multi_genre_analysis': multi_genre_stats,
            'top_genre_pairs': genre_pairs,
            'genre_exclusivity': exclusivity_stats,
            'correlations': correlation_stats
        }
        
        # Save JSON
        output_file = self.output_dir / "genre_cross_pollination_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved: {output_file}")
        
        # Save genre pairs CSV
        pairs_df = pd.DataFrame(genre_pairs)
        pairs_file = self.output_dir / "genre_pair_cooccurrence.csv"
        pairs_df.to_csv(pairs_file, index=False)
        logger.info(f"Saved: {pairs_file}")

def main():
    """Main execution."""
    
    OUTPUT_DIR = "data/processed"
    
    analyzer = GenreCrossPollination(output_dir=OUTPUT_DIR)
    
    # Load data
    genre_df = analyzer.load_data()
    
    # Analyze patterns
    genre_dist = analyzer.analyze_genre_distribution(genre_df)
    multi_genre_stats = analyzer.analyze_multi_genre_playlists(genre_df)
    genre_pairs = analyzer.analyze_genre_pairs(genre_df)
    exclusivity_stats = analyzer.analyze_genre_exclusivity(genre_df)
    correlation_stats = analyzer.analyze_genre_diversity_correlation(genre_df)
    
    # Save results
    analyzer.save_results(genre_dist, multi_genre_stats, genre_pairs, exclusivity_stats, correlation_stats)
    
    logger.info("\n" + "="*60)
    logger.info("✅ Genre cross-pollination analysis complete!")
    logger.info("="*60)

if __name__ == "__main__":
    main()