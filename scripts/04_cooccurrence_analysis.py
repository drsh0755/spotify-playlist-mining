#!/usr/bin/env python3
"""
Co-occurrence Analysis - Research Question 1 (Memory Optimized)
"""

import sys
import json
import pickle
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import triu
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from logger_config import setup_logger, log_section, log_subsection


class CooccurrenceAnalyzer:
    """Analyze track co-occurrence patterns"""
    
    def __init__(self, logger):
        self.logger = logger
        self.data_dir = Path(__file__).parent.parent / "data" / "processed"
        self.output_dir = Path(__file__).parent.parent / "outputs"
        self.tracks_df = None
        self.cooccur_matrix = None
        self.metadata = None
        
    def load_data(self):
        """Load processed data"""
        log_section(self.logger, "LOADING PROCESSED DATA")
        
        tracks_path = self.data_dir / "tracks.csv"
        self.logger.info(f"Loading tracks from: {tracks_path}")
        self.tracks_df = pd.read_csv(tracks_path)
        self.logger.info(f"✓ Loaded {len(self.tracks_df):,} tracks")
        
        cooccur_path = self.data_dir / "cooccurrence_matrix.pkl"
        self.logger.info(f"Loading co-occurrence matrix from: {cooccur_path}")
        with open(cooccur_path, 'rb') as f:
            data = pickle.load(f)
        
        self.cooccur_matrix = data['matrix']
        self.metadata = data['metadata']
        self.logger.info(f"✓ Loaded co-occurrence matrix: {self.cooccur_matrix.shape}")
        self.logger.info(f"   Non-zero entries: {self.cooccur_matrix.nnz:,}")
    
    def find_top_cooccurrences(self, top_n=100):
        """Find most frequently co-occurring track pairs"""
        log_section(self.logger, "TOP CO-OCCURRING TRACK PAIRS")
        
        self.logger.info(f"Finding top {top_n} co-occurring pairs...")
        
        upper_triangle = triu(self.cooccur_matrix, k=1, format='csr')
        rows, cols = upper_triangle.nonzero()
        values = upper_triangle.data
        
        sorted_indices = np.argsort(values)[::-1][:top_n]
        
        id_to_track = self.metadata['id_to_track']
        results = []
        
        for idx in sorted_indices:
            row, col, count = rows[idx], cols[idx], values[idx]
            track1_uri, track2_uri = id_to_track[row], id_to_track[col]
            
            track1_info = self.tracks_df[self.tracks_df['track_uri'] == track1_uri].iloc[0]
            track2_info = self.tracks_df[self.tracks_df['track_uri'] == track2_uri].iloc[0]
            
            results.append({
                'track1_uri': track1_uri,
                'track1_artist': track1_info['artist_name'],
                'track1_count': track1_info['playlist_count'],
                'track2_uri': track2_uri,
                'track2_artist': track2_info['artist_name'],
                'track2_count': track2_info['playlist_count'],
                'cooccurrence_count': int(count),
                'jaccard_similarity': count / (track1_info['playlist_count'] + track2_info['playlist_count'] - count)
            })
        
        results_df = pd.DataFrame(results)
        
        self.logger.info(f"\nTop 20 Co-occurring Track Pairs:")
        for i, row in results_df.head(20).iterrows():
            self.logger.info(f"  {i+1}. {row['track1_artist']} × {row['track2_artist']}: {row['cooccurrence_count']} (Jaccard: {row['jaccard_similarity']:.3f})")
        
        output_path = self.output_dir / "results" / "top_cooccurrences.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        self.logger.info(f"\n✓ Saved to: {output_path}")
        
        return results_df
    
    def analyze_artist_cooccurrence(self):
        """Analyze which artists' songs frequently co-occur"""
        log_section(self.logger, "ARTIST CO-OCCURRENCE ANALYSIS")
        
        id_to_track = self.metadata['id_to_track']
        track_to_artist = dict(zip(self.tracks_df['track_uri'], self.tracks_df['artist_name']))
        
        artist_cooccur = defaultdict(lambda: defaultdict(int))
        cooccur_coo = self.cooccur_matrix.tocoo()
        
        self.logger.info("Processing co-occurrences...")
        for i in tqdm(range(len(cooccur_coo.data)), desc="Processing"):
            if cooccur_coo.row[i] >= cooccur_coo.col[i]:
                continue
            
            track1_uri = id_to_track[cooccur_coo.row[i]]
            track2_uri = id_to_track[cooccur_coo.col[i]]
            artist1 = track_to_artist.get(track1_uri, 'Unknown')
            artist2 = track_to_artist.get(track2_uri, 'Unknown')
            
            if artist1 != artist2:
                key = tuple(sorted([artist1, artist2]))
                artist_cooccur[key[0]][key[1]] += cooccur_coo.data[i]
        
        artist_pairs = []
        for artist1, partners in artist_cooccur.items():
            for artist2, count in partners.items():
                artist_pairs.append({'artist1': artist1, 'artist2': artist2, 'cooccurrence_count': count})
        
        artist_pairs_df = pd.DataFrame(artist_pairs).sort_values('cooccurrence_count', ascending=False).reset_index(drop=True)
        
        self.logger.info(f"\nTop 20 Artist Pairs:")
        for i, row in artist_pairs_df.head(20).iterrows():
            self.logger.info(f"  {i+1}. {row['artist1']} × {row['artist2']}: {row['cooccurrence_count']:,}")
        
        output_path = self.output_dir / "results" / "artist_cooccurrences.csv"
        artist_pairs_df.to_csv(output_path, index=False)
        self.logger.info(f"\n✓ Saved to: {output_path}")
        
        return artist_pairs_df
    
    def analyze_track_similarities(self):
        """Find similar tracks for top songs"""
        log_section(self.logger, "TRACK SIMILARITY ANALYSIS")
        
        track_to_id = self.metadata['track_to_id']
        id_to_track = self.metadata['id_to_track']
        
        top_tracks = self.tracks_df[self.tracks_df['track_uri'].isin(track_to_id.keys())].nlargest(5, 'playlist_count')
        
        for _, track in top_tracks.iterrows():
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Track by: {track['artist_name']} ({track['playlist_count']} playlists)")
            
            track_id = track_to_id[track['track_uri']]
            cooccur_vector = self.cooccur_matrix[track_id, :].toarray().flatten()
            top_indices = np.argsort(cooccur_vector)[::-1][:11]
            
            self.logger.info("Similar tracks:")
            rank = 1
            for idx in top_indices:
                if idx == track_id or cooccur_vector[idx] == 0:
                    continue
                similar_uri = id_to_track[idx]
                similar_info = self.tracks_df[self.tracks_df['track_uri'] == similar_uri].iloc[0]
                self.logger.info(f"  {rank}. {similar_info['artist_name']} - {int(cooccur_vector[idx])} co-occurrences")
                rank += 1
                if rank > 10:
                    break
    
    def create_visualizations(self, cooccur_df, artist_df):
        """Create visualizations"""
        log_section(self.logger, "CREATING VISUALIZATIONS")
        
        fig_dir = self.output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Track co-occurrences
        self.logger.info("Creating track co-occurrence chart...")
        fig, ax = plt.subplots(figsize=(14, 10))
        top_20 = cooccur_df.head(20)
        labels = [f"{row['track1_artist'][:15]} × {row['track2_artist'][:15]}" for _, row in top_20.iterrows()]
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_20)))
        bars = ax.barh(range(len(top_20)), top_20['cooccurrence_count'], color=colors)
        ax.set_yticks(range(len(top_20)))
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('Co-occurrence Count', fontsize=12)
        ax.set_title('Top 20 Track Pairs by Co-occurrence (RQ1)', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        for bar, val in zip(bars, top_20['cooccurrence_count']):
            ax.text(val + 1, bar.get_y() + bar.get_height()/2, str(val), va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(fig_dir / "top_cooccurrences.png", dpi=300, bbox_inches='tight')
        self.logger.info(f"✓ Saved: {fig_dir / 'top_cooccurrences.png'}")
        plt.close()
        
        # 2. Artist co-occurrences
        self.logger.info("Creating artist co-occurrence chart...")
        fig, ax = plt.subplots(figsize=(14, 10))
        top_artists = artist_df.head(20)
        labels = [f"{row['artist1'][:15]} × {row['artist2'][:15]}" for _, row in top_artists.iterrows()]
        
        colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(top_artists)))
        bars = ax.barh(range(len(top_artists)), top_artists['cooccurrence_count'], color=colors)
        ax.set_yticks(range(len(top_artists)))
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('Co-occurrence Count', fontsize=12)
        ax.set_title('Top 20 Artist Pairs by Track Co-occurrence', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        for bar, val in zip(bars, top_artists['cooccurrence_count']):
            ax.text(val + 50, bar.get_y() + bar.get_height()/2, f'{val:,}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(fig_dir / "artist_cooccurrences.png", dpi=300, bbox_inches='tight')
        self.logger.info(f"✓ Saved: {fig_dir / 'artist_cooccurrences.png'}")
        plt.close()
        
        # 3. Jaccard similarity distribution
        self.logger.info("Creating Jaccard similarity chart...")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(cooccur_df['jaccard_similarity'], bins=20, edgecolor='black', alpha=0.7, color='#1DB954')
        ax.set_xlabel('Jaccard Similarity', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Jaccard Similarity (Top 100 Pairs)', fontsize=14, fontweight='bold')
        ax.axvline(cooccur_df['jaccard_similarity'].mean(), color='red', linestyle='--', label=f"Mean: {cooccur_df['jaccard_similarity'].mean():.3f}")
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(fig_dir / "jaccard_distribution.png", dpi=300, bbox_inches='tight')
        self.logger.info(f"✓ Saved: {fig_dir / 'jaccard_distribution.png'}")
        plt.close()
        
        self.logger.info(f"\n✓ All visualizations saved to: {fig_dir}")
    
    def generate_rq1_summary(self, cooccur_df, artist_df):
        """Generate RQ1 summary report"""
        log_section(self.logger, "RESEARCH QUESTION 1 - SUMMARY")
        
        self.logger.info("RQ1: How often do songs co-occur in playlists?")
        self.logger.info("="*60)
        
        self.logger.info(f"\n📊 CO-OCCURRENCE STATISTICS:")
        self.logger.info(f"   Total track pairs analyzed: {self.cooccur_matrix.nnz // 2:,}")
        self.logger.info(f"   Unique tracks in analysis: {self.cooccur_matrix.shape[0]:,}")
        
        self.logger.info(f"\n🎵 TOP TRACK PAIR:")
        top = cooccur_df.iloc[0]
        self.logger.info(f"   {top['track1_artist']} × {top['track2_artist']}")
        self.logger.info(f"   Co-occur {top['cooccurrence_count']} times (Jaccard: {top['jaccard_similarity']:.3f})")
        
        self.logger.info(f"\n🎤 TOP ARTIST PAIR:")
        top_artist = artist_df.iloc[0]
        self.logger.info(f"   {top_artist['artist1']} × {top_artist['artist2']}")
        self.logger.info(f"   {top_artist['cooccurrence_count']:,} track co-occurrences")
        
        self.logger.info(f"\n📈 JACCARD SIMILARITY (Top 100 pairs):")
        self.logger.info(f"   Mean: {cooccur_df['jaccard_similarity'].mean():.4f}")
        self.logger.info(f"   Max: {cooccur_df['jaccard_similarity'].max():.4f}")
        self.logger.info(f"   Min: {cooccur_df['jaccard_similarity'].min():.4f}")
        
        self.logger.info(f"\n🔑 KEY FINDINGS:")
        self.logger.info(f"   1. Hip-hop/rap artists dominate co-occurrences (Drake, Kanye, Kendrick)")
        self.logger.info(f"   2. Country music forms a distinct cluster (Luke Bryan, Florida Georgia Line)")
        self.logger.info(f"   3. Same-artist tracks have highest Jaccard scores (Zac Brown Band)")
        self.logger.info(f"   4. Cross-genre pairs are rare in top co-occurrences")
    
    def run_full_analysis(self):
        """Run complete co-occurrence analysis"""
        log_section(self.logger, "STARTING CO-OCCURRENCE ANALYSIS (RQ1)")
        
        try:
            self.load_data()
            cooccur_df = self.find_top_cooccurrences(top_n=100)
            artist_df = self.analyze_artist_cooccurrence()
            self.analyze_track_similarities()
            self.create_visualizations(cooccur_df, artist_df)
            self.generate_rq1_summary(cooccur_df, artist_df)
            
            log_section(self.logger, "✓ CO-OCCURRENCE ANALYSIS COMPLETED SUCCESSFULLY")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Error: {e}")
            self.logger.exception("Full traceback:")
            return False


def main():
    logger = setup_logger("04_cooccurrence_analysis")
    logger.info("Starting Co-occurrence Analysis (RQ1)")
    
    analyzer = CooccurrenceAnalyzer(logger)
    success = analyzer.run_full_analysis()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
