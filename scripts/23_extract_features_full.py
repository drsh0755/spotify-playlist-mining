"""
Extract features from full MPD for clustering and recommendation.
Generates track features and playlist features for downstream analysis.

Author: Adarsh Singh
Date: November 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
import pickle
from collections import Counter

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'feature_extraction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract features from MPD for machine learning tasks."""

    def __init__(self, data_dir, output_dir):
        """
        Args:
            data_dir: Directory containing processed MPD data
            output_dir: Directory to save feature files
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self, use_parquet=True):
        """Load processed MPD data."""
        logger.info("Loading MPD data...")

        # Load playlists
        if use_parquet and (self.data_dir / "playlists_full_mpd.parquet").exists():
            self.playlists = pd.read_parquet(self.data_dir / "playlists_full_mpd.parquet")
            logger.info("Loaded playlists from Parquet")
        else:
            self.playlists = pd.read_csv(self.data_dir / "playlists_full_mpd.csv")
            logger.info("Loaded playlists from CSV")

        # Load tracks
        if use_parquet and (self.data_dir / "tracks_full_mpd.parquet").exists():
            self.tracks = pd.read_parquet(self.data_dir / "tracks_full_mpd.parquet")
            logger.info("Loaded tracks from Parquet")
        else:
            self.tracks = pd.read_csv(self.data_dir / "tracks_full_mpd.csv")
            logger.info("Loaded tracks from CSV")

        logger.info(f"Playlists: {len(self.playlists):,}")
        logger.info(f"Track entries: {len(self.tracks):,}")
        logger.info(f"Unique tracks: {self.tracks['track_uri'].nunique():,}")

    def extract_track_features(self):
        """Extract track-level features."""
        logger.info("Extracting track features...")

        # Aggregate track statistics
        track_stats = self.tracks.groupby('track_uri').agg({
            'pid': 'nunique',  # Number of playlists
            'track_name': 'first',
            'artist_uri': 'first',
            'artist_name': 'first',
            'album_uri': 'first',
            'album_name': 'first',
            'duration_ms': 'first',
            'position': ['mean', 'std']  # Average position in playlists
        }).reset_index()

        # Flatten column names
        track_stats.columns = [
            'track_uri', 'playlist_count', 'track_name', 'artist_uri',
            'artist_name', 'album_uri', 'album_name', 'duration_ms',
            'avg_position', 'std_position'
        ]

        # Calculate popularity (log-scaled)
        track_stats['popularity'] = np.log1p(track_stats['playlist_count'])

        # Position features
        track_stats['std_position'] = track_stats['std_position'].fillna(0)
        track_stats['position_consistency'] = 1 / (1 + track_stats['std_position'])

        # Artist popularity
        artist_popularity = self.tracks.groupby('artist_uri')['pid'].nunique()
        track_stats['artist_popularity'] = track_stats['artist_uri'].map(artist_popularity)
        track_stats['artist_popularity'] = np.log1p(track_stats['artist_popularity'])

        # Album popularity
        album_popularity = self.tracks.groupby('album_uri')['pid'].nunique()
        track_stats['album_popularity'] = track_stats['album_uri'].map(album_popularity)
        track_stats['album_popularity'] = np.log1p(track_stats['album_popularity'])

        # Duration features (normalize)
        track_stats['duration_minutes'] = track_stats['duration_ms'] / 60000
        track_stats['duration_normalized'] = (
                                                     track_stats['duration_minutes'] - track_stats[
                                                 'duration_minutes'].mean()
                                             ) / track_stats['duration_minutes'].std()

        logger.info(f"Extracted features for {len(track_stats):,} tracks")
        logger.info(f"Feature columns: {list(track_stats.columns)}")

        return track_stats

    def extract_playlist_features(self):
        """Extract playlist-level features."""
        logger.info("Extracting playlist features...")

        # Start with basic playlist data
        playlist_features = self.playlists.copy()

        # Add track diversity metrics
        track_diversity = self.tracks.groupby('pid').agg({
            'artist_uri': 'nunique',
            'album_uri': 'nunique',
            'track_uri': 'count'
        }).rename(columns={
            'artist_uri': 'unique_artists',
            'album_uri': 'unique_albums',
            'track_uri': 'track_count'
        })

        playlist_features = playlist_features.merge(
            track_diversity,
            left_on='pid',
            right_index=True,
            how='left'
        )

        # Artist/album diversity ratios
        playlist_features['artist_diversity'] = (
                playlist_features['unique_artists'] / playlist_features['track_count']
        )
        playlist_features['album_diversity'] = (
                playlist_features['unique_albums'] / playlist_features['track_count']
        )

        # Duration features
        playlist_features['avg_track_duration'] = (
                playlist_features['duration_ms'] / playlist_features['num_tracks']
        )
        playlist_features['duration_hours'] = playlist_features['duration_ms'] / 3600000

        # Engagement features
        playlist_features['edits_per_track'] = (
                playlist_features['num_edits'] / playlist_features['num_tracks']
        )
        playlist_features['followers_per_track'] = (
                playlist_features['num_followers'] / playlist_features['num_tracks']
        )

        # Name features (length, has description)
        playlist_features['name_length'] = playlist_features['name'].str.len()
        playlist_features['has_name'] = (playlist_features['name'].str.len() > 0).astype(int)

        # Collaborative feature
        playlist_features['is_collaborative'] = (
                playlist_features['collaborative'] == 'true'
        ).astype(int)

        logger.info(f"Extracted features for {len(playlist_features):,} playlists")
        logger.info(f"Feature columns: {list(playlist_features.columns)}")

        return playlist_features

    def extract_genre_features(self):
        """Extract genre-like features from playlist names and track patterns."""
        logger.info("Extracting genre indicators from playlist names...")

        # Common genre/mood keywords
        genre_keywords = {
            'workout': ['workout', 'gym', 'running', 'cardio', 'fitness', 'exercise'],
            'party': ['party', 'dance', 'club', 'edm', 'house'],
            'chill': ['chill', 'relax', 'calm', 'study', 'focus', 'acoustic'],
            'rock': ['rock', 'metal', 'punk', 'alternative'],
            'hip_hop': ['hip hop', 'rap', 'trap', 'r&b', 'rnb'],
            'country': ['country', 'folk', 'americana'],
            'pop': ['pop', 'top 40', 'hits', 'chart'],
            'indie': ['indie', 'alternative', 'underground'],
            'classical': ['classical', 'orchestra', 'symphony'],
            'jazz': ['jazz', 'blues', 'soul'],
            'electronic': ['electronic', 'techno', 'trance', 'dubstep'],
            'latin': ['latin', 'reggaeton', 'salsa', 'bachata'],
            'mood': ['sad', 'happy', 'angry', 'melancholy', 'upbeat']
        }

        playlist_genres = self.playlists[['pid', 'name']].copy()
        playlist_genres['name_lower'] = playlist_genres['name'].str.lower()

        # Check for each genre keyword
        for genre, keywords in genre_keywords.items():
            playlist_genres[f'genre_{genre}'] = playlist_genres['name_lower'].apply(
                lambda x: any(kw in str(x) for kw in keywords)
            ).astype(int)

        playlist_genres = playlist_genres.drop(columns=['name', 'name_lower'])

        # Count how many genre tags each playlist has
        genre_cols = [col for col in playlist_genres.columns if col.startswith('genre_')]
        playlist_genres['genre_tag_count'] = playlist_genres[genre_cols].sum(axis=1)

        logger.info(f"Extracted {len(genre_cols)} genre indicators")
        logger.info(f"Genre distribution:")
        for col in genre_cols:
            count = playlist_genres[col].sum()
            logger.info(f"  {col}: {count:,} playlists ({count / len(playlist_genres) * 100:.2f}%)")

        return playlist_genres

    def save_features(self, track_features, playlist_features, genre_features):
        """Save extracted features."""
        logger.info("Saving feature files...")

        # Save track features
        track_file = self.output_dir / "track_features_full.csv"
        track_features.to_csv(track_file, index=False)
        logger.info(f"Saved track features: {track_file}")

        # Save playlist features
        playlist_file = self.output_dir / "playlist_features_full.csv"
        playlist_features.to_csv(playlist_file, index=False)
        logger.info(f"Saved playlist features: {playlist_file}")

        # Save genre features
        genre_file = self.output_dir / "playlist_genre_features.csv"
        genre_features.to_csv(genre_file, index=False)
        logger.info(f"Saved genre features: {genre_file}")

        # Save Parquet versions for faster loading
        track_features.to_parquet(self.output_dir / "track_features_full.parquet")
        playlist_features.to_parquet(self.output_dir / "playlist_features_full.parquet")
        genre_features.to_parquet(self.output_dir / "playlist_genre_features.parquet")
        logger.info("Saved Parquet versions")

        # Summary statistics
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Feature Extraction Summary:")
        logger.info(f"Track features: {track_features.shape}")
        logger.info(f"Playlist features: {playlist_features.shape}")
        logger.info(f"Genre features: {genre_features.shape}")
        logger.info(f"{'=' * 60}\n")

    def extract_all_features(self):
        """Complete feature extraction pipeline."""

        # Load data
        self.load_data()

        # Extract features
        track_features = self.extract_track_features()
        playlist_features = self.extract_playlist_features()
        genre_features = self.extract_genre_features()

        # Save features
        self.save_features(track_features, playlist_features, genre_features)

        return track_features, playlist_features, genre_features


def main():
    """Main execution function."""

    # Configuration
    DATA_DIR = "data/processed"
    OUTPUT_DIR = "data/processed"

    # Extract features
    extractor = FeatureExtractor(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR
    )

    track_features, playlist_features, genre_features = extractor.extract_all_features()

    logger.info("✅ Feature extraction complete!")


if __name__ == "__main__":
    main()