"""
Load and process all 1,000 MPD slices with memory-efficient batch processing.
Handles 1M playlists with checkpointing and error recovery.

Author: Adarsh Singh
Date: November 2024
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
import gc
import psutil
import pickle

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'mpd_loading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MPDLoader:
    """Memory-efficient loader for Million Playlist Dataset."""

    def __init__(self, mpd_dir, output_dir, batch_size=10, checkpoint_freq=100):
        """
        Args:
            mpd_dir: Directory containing mpd.slice.*.json files
            output_dir: Directory to save processed data
            batch_size: Number of slices to process before writing (memory management)
            checkpoint_freq: Save checkpoint every N slices
        """
        self.mpd_dir = Path(mpd_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.checkpoint_freq = checkpoint_freq

        # Get all slice files sorted by number
        self.slice_files = sorted(
            self.mpd_dir.glob("mpd.slice.*.json"),
            key=lambda x: int(x.stem.split('.')[-1].split('-')[0])
        )
        logger.info(f"Found {len(self.slice_files)} MPD slice files")

        # Statistics
        self.stats = {
            'total_playlists': 0,
            'total_tracks': 0,
            'unique_tracks': set(),
            'unique_artists': set(),
            'unique_albums': set(),
            'processing_times': []
        }

    def get_memory_usage(self):
        """Get current memory usage in GB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024 / 1024

    def load_slice(self, slice_file):
        """Load a single slice file."""
        try:
            with open(slice_file, 'r') as f:
                data = json.load(f)
            return data['playlists']
        except Exception as e:
            logger.error(f"Error loading {slice_file}: {e}")
            return []

    def extract_playlist_data(self, playlists):
        """Extract playlist-level data."""
        playlist_data = []

        for playlist in playlists:
            playlist_data.append({
                'pid': playlist['pid'],
                'name': playlist.get('name', ''),
                'num_tracks': playlist['num_tracks'],
                'num_albums': playlist['num_albums'],
                'num_artists': playlist['num_artists'],
                'num_followers': playlist['num_followers'],
                'num_edits': playlist['num_edits'],
                'duration_ms': playlist['duration_ms'],
                'modified_at': playlist['modified_at'],
                'collaborative': playlist.get('collaborative', 'false')
            })

        return pd.DataFrame(playlist_data)

    def extract_track_data(self, playlists):
        """Extract track-level data with playlist associations."""
        track_data = []

        for playlist in playlists:
            pid = playlist['pid']
            for idx, track in enumerate(playlist['tracks']):
                track_data.append({
                    'pid': pid,
                    'track_uri': track['track_uri'],
                    'track_name': track['track_name'],
                    'artist_uri': track['artist_uri'],
                    'artist_name': track['artist_name'],
                    'album_uri': track['album_uri'],
                    'album_name': track['album_name'],
                    'duration_ms': track['duration_ms'],
                    'position': idx
                })

                # Update statistics
                self.stats['unique_tracks'].add(track['track_uri'])
                self.stats['unique_artists'].add(track['artist_uri'])
                self.stats['unique_albums'].add(track['album_uri'])

        return pd.DataFrame(track_data)

    def save_checkpoint(self, playlist_df, track_df, slice_num):
        """Save checkpoint data."""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint_file = checkpoint_dir / f"checkpoint_slice_{slice_num}.pkl"
        checkpoint_data = {
            'playlist_df': playlist_df,
            'track_df': track_df,
            'stats': self.stats.copy(),
            'slice_num': slice_num,
            'timestamp': datetime.now().isoformat()
        }

        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        logger.info(f"Checkpoint saved at slice {slice_num}")

    def load_checkpoint(self):
        """Load the most recent checkpoint if exists."""
        checkpoint_dir = self.output_dir / "checkpoints"
        if not checkpoint_dir.exists():
            return None, None, 0

        checkpoints = sorted(checkpoint_dir.glob("checkpoint_slice_*.pkl"))
        if not checkpoints:
            return None, None, 0

        latest_checkpoint = checkpoints[-1]
        logger.info(f"Loading checkpoint: {latest_checkpoint}")

        with open(latest_checkpoint, 'rb') as f:
            data = pickle.load(f)

        self.stats = data['stats']
        return data['playlist_df'], data['track_df'], data['slice_num']

    def process_all_slices(self, resume=True):
        """Process all MPD slices with batch processing."""

        start_time = datetime.now()
        logger.info(f"Starting MPD processing at {start_time}")
        logger.info(f"Total slices to process: {len(self.slice_files)}")
        logger.info(f"Batch size: {self.batch_size}, Checkpoint frequency: {self.checkpoint_freq}")

        # Try to resume from checkpoint
        playlist_dfs = []
        track_dfs = []
        start_slice = 0

        if resume:
            checkpoint_playlists, checkpoint_tracks, start_slice = self.load_checkpoint()
            if checkpoint_playlists is not None:
                playlist_dfs.append(checkpoint_playlists)
                track_dfs.append(checkpoint_tracks)
                logger.info(f"Resuming from slice {start_slice + 1}")

        batch_playlists = []
        batch_tracks = []

        # Process slices with progress bar
        for i, slice_file in enumerate(tqdm(self.slice_files[start_slice:],
                                            initial=start_slice,
                                            total=len(self.slice_files),
                                            desc="Processing MPD slices")):

            slice_num = start_slice + i
            slice_start = datetime.now()

            # Load slice
            playlists = self.load_slice(slice_file)
            if not playlists:
                logger.warning(f"Empty or failed slice: {slice_file}")
                continue

            # Extract data
            playlist_df = self.extract_playlist_data(playlists)
            track_df = self.extract_track_data(playlists)

            batch_playlists.append(playlist_df)
            batch_tracks.append(track_df)

            # Update statistics
            self.stats['total_playlists'] += len(playlists)
            self.stats['total_tracks'] += len(track_df)

            slice_time = (datetime.now() - slice_start).total_seconds()
            self.stats['processing_times'].append(slice_time)

            # Log progress
            if (slice_num + 1) % 10 == 0:
                avg_time = np.mean(self.stats['processing_times'][-10:])
                remaining_slices = len(self.slice_files) - (slice_num + 1)
                eta_minutes = (avg_time * remaining_slices) / 60
                memory_gb = self.get_memory_usage()

                logger.info(f"Slice {slice_num + 1}/{len(self.slice_files)} | "
                            f"Playlists: {self.stats['total_playlists']:,} | "
                            f"Tracks: {self.stats['total_tracks']:,} | "
                            f"Unique tracks: {len(self.stats['unique_tracks']):,} | "
                            f"Memory: {memory_gb:.2f}GB | "
                            f"ETA: {eta_minutes:.1f}min")

            # Save batch
            if len(batch_playlists) >= self.batch_size:
                logger.info(f"Writing batch at slice {slice_num + 1}")
                playlist_dfs.append(pd.concat(batch_playlists, ignore_index=True))
                track_dfs.append(pd.concat(batch_tracks, ignore_index=True))
                batch_playlists = []
                batch_tracks = []
                gc.collect()

            # Save checkpoint
            if self.checkpoint_freq > 0 and (slice_num + 1) % self.checkpoint_freq == 0:
                current_playlists = pd.concat(playlist_dfs + batch_playlists, ignore_index=True)
                current_tracks = pd.concat(track_dfs + batch_tracks, ignore_index=True)
                self.save_checkpoint(current_playlists, current_tracks, slice_num)

        # Process remaining batch
        if batch_playlists:
            playlist_dfs.append(pd.concat(batch_playlists, ignore_index=True))
            track_dfs.append(pd.concat(batch_tracks, ignore_index=True))

        # Combine all data
        logger.info("Combining all batches...")
        final_playlists = pd.concat(playlist_dfs, ignore_index=True)
        final_tracks = pd.concat(track_dfs, ignore_index=True)

        # Final statistics
        self.stats['unique_tracks'] = len(self.stats['unique_tracks'])
        self.stats['unique_artists'] = len(self.stats['unique_artists'])
        self.stats['unique_albums'] = len(self.stats['unique_albums'])

        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing complete!")
        logger.info(f"Total time: {total_time / 60:.2f} minutes")
        logger.info(f"Total playlists: {self.stats['total_playlists']:,}")
        logger.info(f"Total tracks: {self.stats['total_tracks']:,}")
        logger.info(f"Unique tracks: {self.stats['unique_tracks']:,}")
        logger.info(f"Unique artists: {self.stats['unique_artists']:,}")
        logger.info(f"Unique albums: {self.stats['unique_albums']:,}")
        logger.info(f"{'=' * 60}\n")

        return final_playlists, final_tracks

    def save_processed_data(self, playlist_df, track_df):
        """Save processed data to disk."""
        logger.info("Saving processed data...")

        # Save CSVs
        playlist_file = self.output_dir / "playlists_full_mpd.csv"
        track_file = self.output_dir / "tracks_full_mpd.csv"
        stats_file = self.output_dir / "mpd_statistics.json"

        playlist_df.to_csv(playlist_file, index=False)
        logger.info(
            f"Saved playlists: {playlist_file} ({playlist_df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB)")

        track_df.to_csv(track_file, index=False)
        logger.info(f"Saved tracks: {track_file} ({track_df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB)")

        # Save statistics
        stats_to_save = self.stats.copy()
        with open(stats_file, 'w') as f:
            json.dump(stats_to_save, f, indent=2)
        logger.info(f"Saved statistics: {stats_file}")

        # Save Parquet for faster loading
        playlist_df.to_parquet(self.output_dir / "playlists_full_mpd.parquet")
        track_df.to_parquet(self.output_dir / "tracks_full_mpd.parquet")
        logger.info("Saved Parquet files for faster loading")


def main():
    """Main execution function."""

    # Configuration
    MPD_DIR = "data/raw/mpd_slices"
    OUTPUT_DIR = "data/processed"
    BATCH_SIZE = 5  # Process 5 slices before writing (reduced memory)
    CHECKPOINT_FREQ = 0  # Disable checkpoints to save disk space

    # Initialize loader
    loader = MPDLoader(
        mpd_dir=MPD_DIR,
        output_dir=OUTPUT_DIR,
        batch_size=BATCH_SIZE,
        checkpoint_freq=CHECKPOINT_FREQ
    )

    # Process all slices
    playlist_df, track_df = loader.process_all_slices(resume=True)

    # Save results
    loader.save_processed_data(playlist_df, track_df)

    logger.info("✅ MPD loading complete!")


if __name__ == "__main__":
    main()