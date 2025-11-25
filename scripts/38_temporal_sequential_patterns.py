"""
Temporal and Sequential Pattern Analysis
Analyze track position patterns and playlist evolution

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
        logging.FileHandler(log_dir / f'temporal_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TemporalSequentialAnalysis:
    """Analyze temporal and sequential patterns in playlists."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """Load track and playlist data."""
        logger.info("Loading data...")
        
        tracks_df = pd.read_parquet("data/processed/tracks_full_mpd.parquet")
        logger.info(f"Loaded {len(tracks_df):,} track entries")
        logger.info(f"Columns: {list(tracks_df.columns)}")
        
        return tracks_df
    
    def analyze_playlist_patterns(self, tracks_df):
        """Analyze basic playlist patterns."""
        logger.info("\n" + "="*60)
        logger.info("Playlist Pattern Analysis")
        logger.info("="*60)
        
        # Playlist size distribution
        playlist_sizes = tracks_df.groupby('pid').size()
        
        size_stats = {
            'mean_size': float(playlist_sizes.mean()),
            'median_size': float(playlist_sizes.median()),
            'std_size': float(playlist_sizes.std()),
            'min_size': int(playlist_sizes.min()),
            'max_size': int(playlist_sizes.max()),
            'total_playlists': int(len(playlist_sizes))
        }
        
        logger.info(f"Total Playlists: {size_stats['total_playlists']:,}")
        logger.info(f"Mean Size: {size_stats['mean_size']:.2f} tracks")
        logger.info(f"Median Size: {size_stats['median_size']:.2f} tracks")
        logger.info(f"Std Size: {size_stats['std_size']:.2f} tracks")
        logger.info(f"Size Range: {size_stats['min_size']} - {size_stats['max_size']} tracks")
        
        # Size distribution by bins
        logger.info("\nPlaylist Size Distribution:")
        bins = [0, 10, 25, 50, 100, 200, playlist_sizes.max() + 1]
        labels = ['Tiny (0-10)', 'Small (11-25)', 'Medium (26-50)', 'Large (51-100)', 'Huge (101-200)', 'Massive (200+)']
        
        size_bins = pd.cut(playlist_sizes, bins=bins, labels=labels, right=False)
        size_dist = size_bins.value_counts().sort_index()
        
        for label, count in size_dist.items():
            pct = (count / len(playlist_sizes)) * 100
            logger.info(f"  {label:20s}: {count:>8,} ({pct:>5.2f}%)")
        
        return size_stats
    
    def analyze_track_frequency(self, tracks_df, top_n=20):
        """Analyze most frequently occurring tracks."""
        logger.info("\n" + "="*60)
        logger.info("Most Frequent Tracks (Overall)")
        logger.info("="*60)
        
        track_counter = Counter(tracks_df['track_uri'].tolist())
        
        top_tracks = []
        for rank, (track_uri, count) in enumerate(track_counter.most_common(top_n), 1):
            pct = (count / len(tracks_df)) * 100
            n_playlists = len(tracks_df[tracks_df['track_uri'] == track_uri]['pid'].unique())
            top_tracks.append({
                'rank': rank,
                'track_uri': track_uri,
                'total_occurrences': count,
                'num_playlists': n_playlists,
                'percentage': pct
            })
            logger.info(f"  {rank:2d}. Occurrences: {count:>6,} | In {n_playlists:>6,} playlists | {track_uri}")
        
        return top_tracks
    
    def analyze_track_diversity_by_size(self, tracks_df):
        """Analyze track diversity by playlist size."""
        logger.info("\n" + "="*60)
        logger.info("Track Diversity by Playlist Size")
        logger.info("="*60)
        
        # Calculate diversity per playlist
        diversity_data = []
        for pid, group in tracks_df.groupby('pid'):
            size = len(group)
            unique_tracks = group['track_uri'].nunique()
            unique_artists = group['artist_uri'].nunique() if 'artist_uri' in group.columns else 0
            
            diversity_data.append({
                'pid': pid,
                'size': size,
                'unique_tracks': unique_tracks,
                'track_diversity': unique_tracks / size if size > 0 else 0,
                'unique_artists': unique_artists,
                'artist_diversity': unique_artists / size if size > 0 else 0
            })
        
        diversity_df = pd.DataFrame(diversity_data)
        
        # Bin by size
        bins = [0, 10, 25, 50, 100, 200, diversity_df['size'].max() + 1]
        labels = ['0-10', '11-25', '26-50', '51-100', '101-200', '200+']
        diversity_df['size_bin'] = pd.cut(diversity_df['size'], bins=bins, labels=labels, right=False)
        
        # Calculate average diversity per bin
        diversity_by_size = diversity_df.groupby('size_bin').agg({
            'track_diversity': 'mean',
            'artist_diversity': 'mean',
            'size': 'count'
        }).round(4)
        
        logger.info("\nSize Bin | Avg Track Diversity | Avg Artist Diversity | Count")
        logger.info("-" * 70)
        for label in labels:
            if label in diversity_by_size.index:
                row = diversity_by_size.loc[label]
                logger.info(f"{label:10s} | {row['track_diversity']:18.4f} | {row['artist_diversity']:19.4f} | {int(row['size']):>6,}")
        
        return diversity_by_size.to_dict()
    
    def analyze_artist_patterns(self, tracks_df, top_n=20):
        """Analyze artist patterns."""
        logger.info("\n" + "="*60)
        logger.info("Most Frequent Artists")
        logger.info("="*60)
        
        if 'artist_uri' not in tracks_df.columns:
            logger.warning("No artist_uri column found, skipping artist analysis")
            return []
        
        artist_counter = Counter(tracks_df['artist_uri'].tolist())
        
        top_artists = []
        for rank, (artist_uri, count) in enumerate(artist_counter.most_common(top_n), 1):
            n_playlists = len(tracks_df[tracks_df['artist_uri'] == artist_uri]['pid'].unique())
            top_artists.append({
                'rank': rank,
                'artist_uri': artist_uri,
                'total_tracks': count,
                'num_playlists': n_playlists
            })
            logger.info(f"  {rank:2d}. Tracks: {count:>7,} | In {n_playlists:>6,} playlists | {artist_uri}")
        
        return top_artists
    
    def analyze_sequential_patterns(self, tracks_df, sample_size=50000):
        """Analyze sequential patterns in playlists."""
        logger.info("\n" + "="*60)
        logger.info("Sequential Pattern Analysis (Sampled)")
        logger.info("="*60)
        
        # Sample playlists
        unique_pids = tracks_df['pid'].unique()
        np.random.seed(42)
        sample_pids = np.random.choice(unique_pids, min(sample_size, len(unique_pids)), replace=False)
        
        logger.info(f"Analyzing {len(sample_pids):,} sampled playlists...")
        
        # Count track repetitions in same playlist
        repetitions = []
        for pid in sample_pids:
            playlist_tracks = tracks_df[tracks_df['pid'] == pid]['track_uri'].tolist()
            if len(playlist_tracks) != len(set(playlist_tracks)):
                repetitions.append(pid)
        
        rep_stats = {
            'playlists_with_repetitions': len(repetitions),
            'percentage': (len(repetitions) / len(sample_pids)) * 100
        }
        
        logger.info(f"Playlists with repeated tracks: {rep_stats['playlists_with_repetitions']:,} ({rep_stats['percentage']:.2f}%)")
        
        return rep_stats
    
    def save_results(self, playlist_stats, top_tracks, diversity_stats, top_artists, sequential_stats):
        """Save all results."""
        logger.info("\nSaving results...")
        
        results = {
            'playlist_patterns': playlist_stats,
            'top_tracks': top_tracks,
            'diversity_by_size': diversity_stats,
            'top_artists': top_artists,
            'sequential_patterns': sequential_stats
        }
        
        # Save JSON
        output_file = self.output_dir / "temporal_sequential_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved: {output_file}")
        
        # Save CSVs
        tracks_df = pd.DataFrame(top_tracks)
        tracks_file = self.output_dir / "top_frequent_tracks.csv"
        tracks_df.to_csv(tracks_file, index=False)
        logger.info(f"Saved: {tracks_file}")
        
        if top_artists:
            artists_df = pd.DataFrame(top_artists)
            artists_file = self.output_dir / "top_frequent_artists.csv"
            artists_df.to_csv(artists_file, index=False)
            logger.info(f"Saved: {artists_file}")

def main():
    """Main execution."""
    
    OUTPUT_DIR = "data/processed"
    
    analyzer = TemporalSequentialAnalysis(output_dir=OUTPUT_DIR)
    
    # Load data
    tracks_df = analyzer.load_data()
    
    # Analyze patterns
    playlist_stats = analyzer.analyze_playlist_patterns(tracks_df)
    top_tracks = analyzer.analyze_track_frequency(tracks_df, top_n=20)
    diversity_stats = analyzer.analyze_track_diversity_by_size(tracks_df)
    top_artists = analyzer.analyze_artist_patterns(tracks_df, top_n=20)
    sequential_stats = analyzer.analyze_sequential_patterns(tracks_df, sample_size=50000)
    
    # Save results
    analyzer.save_results(playlist_stats, top_tracks, diversity_stats, top_artists, sequential_stats)
    
    logger.info("\n" + "="*60)
    logger.info("✅ Temporal & sequential analysis complete!")
    logger.info("="*60)

if __name__ == "__main__":
    main()