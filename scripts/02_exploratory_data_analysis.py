#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for Spotify Challenge Set

This script performs comprehensive analysis of the challenge dataset:
- Playlist characteristics
- Track distributions
- Artist/album patterns
- Metadata completeness
- Generates visualizations

Usage:
    python scripts/02_exploratory_data_analysis.py
    
    # Or with screen:
    ./scripts/run_with_screen.sh 02_exploratory_data_analysis.py
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from logger_config import setup_logger, log_section, log_subsection

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class SpotifyEDA:
    """Exploratory Data Analysis for Spotify Challenge Set"""
    
    def __init__(self, logger):
        self.logger = logger
        self.data_path = Path(__file__).parent.parent / "data" / "raw" / "challenge_set.json"
        self.output_dir = Path(__file__).parent.parent / "outputs" / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data = None
        self.playlists = []
        
    def load_data(self):
        """Load the challenge set"""
        log_subsection(self.logger, "Loading Challenge Set")
        
        self.logger.info(f"Reading from: {self.data_path}")
        with open(self.data_path, 'r') as f:
            self.data = json.load(f)
        
        self.playlists = self.data.get('playlists', [])
        self.logger.info(f"✓ Loaded {len(self.playlists):,} playlists")
        
    def analyze_playlist_metadata(self):
        """Analyze playlist-level metadata"""
        log_section(self.logger, "PLAYLIST METADATA ANALYSIS")
        
        # Basic counts
        total = len(self.playlists)
        has_name = sum(1 for p in self.playlists if p.get('name'))
        has_description = sum(1 for p in self.playlists if p.get('description'))
        has_collaborative = sum(1 for p in self.playlists if p.get('collaborative'))
        
        self.logger.info(f"Total playlists: {total:,}")
        self.logger.info(f"With names: {has_name:,} ({has_name/total*100:.1f}%)")
        self.logger.info(f"With descriptions: {has_description:,} ({has_description/total*100:.1f}%)")
        self.logger.info(f"Collaborative: {has_collaborative:,} ({has_collaborative/total*100:.1f}%)")
        
        # Name length distribution
        self.logger.info("\nAnalyzing playlist name lengths...")
        name_lengths = [len(p.get('name', '')) for p in self.playlists if p.get('name')]
        
        if name_lengths:
            self.logger.info(f"Name length - Min: {min(name_lengths)}, Max: {max(name_lengths)}, Mean: {np.mean(name_lengths):.1f}")
        
        return {
            'total': total,
            'has_name': has_name,
            'has_description': has_description,
            'name_lengths': name_lengths
        }
    
    def analyze_track_distributions(self):
        """Analyze track counts and distributions"""
        log_section(self.logger, "TRACK DISTRIBUTION ANALYSIS")
        
        # Track counts per playlist
        track_counts = [len(p.get('tracks', [])) for p in self.playlists]
        
        self.logger.info("Track count statistics:")
        self.logger.info(f"  Min: {min(track_counts)}")
        self.logger.info(f"  Max: {max(track_counts)}")
        self.logger.info(f"  Mean: {np.mean(track_counts):.2f}")
        self.logger.info(f"  Median: {np.median(track_counts):.2f}")
        self.logger.info(f"  Std Dev: {np.std(track_counts):.2f}")
        
        # Percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        self.logger.info("\nPercentiles:")
        for p in percentiles:
            val = np.percentile(track_counts, p)
            self.logger.info(f"  {p}th: {val:.0f}")
        
        # Empty playlists
        empty = sum(1 for c in track_counts if c == 0)
        self.logger.info(f"\nEmpty playlists (0 tracks): {empty:,} ({empty/len(track_counts)*100:.1f}%)")
        
        return track_counts
    
    def analyze_tracks_and_artists(self):
        """Analyze individual tracks and artists"""
        log_section(self.logger, "TRACK & ARTIST ANALYSIS")
        
        self.logger.info("Extracting track and artist information...")
        
        track_counter = Counter()
        artist_counter = Counter()
        album_counter = Counter()
        track_to_artist = {}
        
        for playlist in tqdm(self.playlists, desc="Processing playlists"):
            for track in playlist.get('tracks', []):
                track_uri = track.get('track_uri')
                artist_name = track.get('artist_name')
                album_name = track.get('album_name')
                
                if track_uri:
                    track_counter[track_uri] += 1
                    if artist_name:
                        track_to_artist[track_uri] = artist_name
                
                if artist_name:
                    artist_counter[artist_name] += 1
                
                if album_name:
                    album_counter[album_name] += 1
        
        # Track statistics
        self.logger.info(f"\nUnique tracks: {len(track_counter):,}")
        self.logger.info(f"Unique artists: {len(artist_counter):,}")
        self.logger.info(f"Unique albums: {len(album_counter):,}")
        
        # Most popular tracks
        self.logger.info("\nTop 10 Most Popular Tracks:")
        for i, (track_uri, count) in enumerate(track_counter.most_common(10), 1):
            artist = track_to_artist.get(track_uri, 'Unknown')
            self.logger.info(f"  {i}. {track_uri[:40]}... by {artist} - {count} playlists")
        
        # Most popular artists
        self.logger.info("\nTop 10 Most Popular Artists:")
        for i, (artist, count) in enumerate(artist_counter.most_common(10), 1):
            self.logger.info(f"  {i}. {artist} - {count} tracks")
        
        # Track appearance distribution
        appearances = list(track_counter.values())
        self.logger.info("\nTrack appearance statistics:")
        self.logger.info(f"  Tracks appearing once: {sum(1 for c in appearances if c == 1):,}")
        self.logger.info(f"  Tracks appearing 2-5 times: {sum(1 for c in appearances if 2 <= c <= 5):,}")
        self.logger.info(f"  Tracks appearing 6-10 times: {sum(1 for c in appearances if 6 <= c <= 10):,}")
        self.logger.info(f"  Tracks appearing 10+ times: {sum(1 for c in appearances if c > 10):,}")
        
        return {
            'track_counter': track_counter,
            'artist_counter': artist_counter,
            'album_counter': album_counter,
            'track_to_artist': track_to_artist
        }
    
    def analyze_playlist_names(self):
        """Analyze common words in playlist names"""
        log_section(self.logger, "PLAYLIST NAME ANALYSIS")
        
        self.logger.info("Analyzing playlist names...")
        
        # Get all playlist names
        names = [p.get('name', '').lower() for p in self.playlists if p.get('name')]
        
        # Extract words
        word_counter = Counter()
        for name in names:
            # Simple word extraction (split on spaces and common punctuation)
            words = name.replace('-', ' ').replace('_', ' ').split()
            # Filter out very short words
            words = [w.strip('.,!?()[]{}"\';:') for w in words if len(w) > 2]
            word_counter.update(words)
        
        self.logger.info(f"\nTotal unique words in playlist names: {len(word_counter):,}")
        
        self.logger.info("\nTop 20 Most Common Words in Playlist Names:")
        for i, (word, count) in enumerate(word_counter.most_common(20), 1):
            self.logger.info(f"  {i}. '{word}' - {count:,} times")
        
        return word_counter
    
    def create_visualizations(self, track_counts, metadata_stats):
        """Create and save visualizations"""
        log_section(self.logger, "CREATING VISUALIZATIONS")
        
        # 1. Track count distribution
        self.logger.info("Creating track count distribution plot...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Histogram
        axes[0, 0].hist(track_counts, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Number of Tracks')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Tracks per Playlist')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Box plot
        axes[0, 1].boxplot(track_counts, vert=True)
        axes[0, 1].set_ylabel('Number of Tracks')
        axes[0, 1].set_title('Track Count Box Plot')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # CDF
        sorted_counts = np.sort(track_counts)
        cdf = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
        axes[1, 0].plot(sorted_counts, cdf, linewidth=2)
        axes[1, 0].set_xlabel('Number of Tracks')
        axes[1, 0].set_ylabel('Cumulative Probability')
        axes[1, 0].set_title('Cumulative Distribution Function')
        axes[1, 0].grid(alpha=0.3)
        
        # Binned distribution
        bins = [0, 5, 10, 25, 50, 100, max(track_counts)+1]
        labels = ['0-5', '6-10', '11-25', '26-50', '51-100', '100+']
        binned = pd.cut(track_counts, bins=bins, labels=labels, right=False)
        binned_counts = binned.value_counts().sort_index()
        
        axes[1, 1].bar(range(len(binned_counts)), binned_counts.values)
        axes[1, 1].set_xticks(range(len(binned_counts)))
        axes[1, 1].set_xticklabels(labels, rotation=45)
        axes[1, 1].set_xlabel('Track Count Range')
        axes[1, 1].set_ylabel('Number of Playlists')
        axes[1, 1].set_title('Playlists by Track Count Range')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / 'track_distribution_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"✓ Saved: {output_path}")
        plt.close()
        
        # 2. Metadata completeness
        self.logger.info("Creating metadata completeness plot...")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = ['Has Name', 'Has Tracks', 'Has Description']
        values = [
            metadata_stats['has_name'],
            sum(1 for p in self.playlists if len(p.get('tracks', [])) > 0),
            metadata_stats['has_description']
        ]
        percentages = [v / metadata_stats['total'] * 100 for v in values]
        
        bars = ax.bar(categories, percentages, color=['#1DB954', '#1ed760', '#535353'])
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Playlist Metadata Completeness')
        ax.set_ylim([0, 100])
        ax.grid(axis='y', alpha=0.3)
        
        # Add percentage labels on bars
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{pct:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        output_path = self.output_dir / 'metadata_completeness.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"✓ Saved: {output_path}")
        plt.close()
        
        # 3. Name length distribution
        if metadata_stats['name_lengths']:
            self.logger.info("Creating name length distribution plot...")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            ax.hist(metadata_stats['name_lengths'], bins=30, edgecolor='black', alpha=0.7, color='#1DB954')
            ax.set_xlabel('Playlist Name Length (characters)')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Playlist Name Lengths')
            ax.grid(axis='y', alpha=0.3)
            
            # Add mean line
            mean_length = np.mean(metadata_stats['name_lengths'])
            ax.axvline(mean_length, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_length:.1f}')
            ax.legend()
            
            output_path = self.output_dir / 'name_length_distribution.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"✓ Saved: {output_path}")
            plt.close()
        
        self.logger.info(f"\n✓ All visualizations saved to: {self.output_dir}")
    
    def save_summary_statistics(self, track_counts, track_data):
        """Save summary statistics to CSV"""
        log_section(self.logger, "SAVING SUMMARY STATISTICS")
        
        output_path = Path(__file__).parent.parent / "outputs" / "results" / "eda_summary.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary = {
            'Metric': [
                'Total Playlists',
                'Total Track Instances',
                'Unique Tracks',
                'Unique Artists',
                'Unique Albums',
                'Min Tracks per Playlist',
                'Max Tracks per Playlist',
                'Mean Tracks per Playlist',
                'Median Tracks per Playlist',
                'Empty Playlists'
            ],
            'Value': [
                len(self.playlists),
                sum(track_counts),
                len(track_data['track_counter']),
                len(track_data['artist_counter']),
                len(track_data['album_counter']),
                min(track_counts),
                max(track_counts),
                f"{np.mean(track_counts):.2f}",
                np.median(track_counts),
                sum(1 for c in track_counts if c == 0)
            ]
        }
        
        df = pd.DataFrame(summary)
        df.to_csv(output_path, index=False)
        self.logger.info(f"✓ Summary statistics saved to: {output_path}")
        
        # Save top tracks
        top_tracks_path = Path(__file__).parent.parent / "outputs" / "results" / "top_tracks.csv"
        top_tracks_data = []
        for track_uri, count in track_data['track_counter'].most_common(100):
            artist = track_data['track_to_artist'].get(track_uri, 'Unknown')
            top_tracks_data.append({
                'track_uri': track_uri,
                'artist_name': artist,
                'playlist_count': count
            })
        
        pd.DataFrame(top_tracks_data).to_csv(top_tracks_path, index=False)
        self.logger.info(f"✓ Top 100 tracks saved to: {top_tracks_path}")
        
        # Save top artists
        top_artists_path = Path(__file__).parent.parent / "outputs" / "results" / "top_artists.csv"
        top_artists_data = [
            {'artist_name': artist, 'track_count': count}
            for artist, count in track_data['artist_counter'].most_common(100)
        ]
        pd.DataFrame(top_artists_data).to_csv(top_artists_path, index=False)
        self.logger.info(f"✓ Top 100 artists saved to: {top_artists_path}")
    
    def run_full_analysis(self):
        """Run complete EDA pipeline"""
        log_section(self.logger, "STARTING EXPLORATORY DATA ANALYSIS")
        
        try:
            # Load data
            self.load_data()
            
            # Run analyses
            metadata_stats = self.analyze_playlist_metadata()
            track_counts = self.analyze_track_distributions()
            track_data = self.analyze_tracks_and_artists()
            word_counter = self.analyze_playlist_names()
            
            # Create visualizations
            self.create_visualizations(track_counts, metadata_stats)
            
            # Save results
            self.save_summary_statistics(track_counts, track_data)
            
            log_section(self.logger, "✓ EDA COMPLETED SUCCESSFULLY")
            self.logger.info("All analyses completed and results saved")
            
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Error during EDA: {e}")
            self.logger.exception("Full traceback:")
            return False


def main():
    """Main execution function"""
    logger = setup_logger("02_exploratory_data_analysis")
    
    logger.info("Starting Exploratory Data Analysis")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Working directory: {Path.cwd()}")
    
    # Run EDA
    eda = SpotifyEDA(logger)
    success = eda.run_full_analysis()
    
    if success:
        logger.info("Script completed successfully")
        sys.exit(0)
    else:
        logger.error("Script failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
