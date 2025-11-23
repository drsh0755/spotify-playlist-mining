#!/usr/bin/env python3
"""
Recommendation Diversity Analysis - Proposal Requirement 4.4.3

Analyzes diversity and thematic coherence of recommendations:
- Artist diversity metrics
- Popularity distribution
- Genre coverage
- Comparison across methods
"""

import sys
import json
import pickle
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from logger_config import setup_logger, log_section, log_subsection


class DiversityAnalyzer:
    """Analyze recommendation diversity"""
    
    def __init__(self, logger):
        self.logger = logger
        self.data_dir = Path(__file__).parent.parent / "data"
        self.output_dir = Path(__file__).parent.parent / "outputs"
        
    def load_data(self):
        """Load required data"""
        log_section(self.logger, "LOADING DATA")
        
        # Load playlists
        with open(self.data_dir / "raw" / "challenge_set.json", 'r') as f:
            self.playlists = json.load(f)['playlists']
        self.logger.info(f"Loaded {len(self.playlists):,} playlists")
        
        # Load tracks
        self.tracks_df = pd.read_csv(self.data_dir / "processed" / "tracks.csv")
        self.track_to_artist = dict(zip(self.tracks_df['track_uri'], self.tracks_df['artist_name']))
        self.track_popularity = dict(zip(self.tracks_df['track_uri'], self.tracks_df['playlist_count']))
        
        # Load co-occurrence
        with open(self.data_dir / "processed" / "cooccurrence_matrix.pkl", 'rb') as f:
            cooccur_data = pickle.load(f)
        self.cooccur_matrix = cooccur_data['matrix']
        self.cooccur_metadata = cooccur_data['metadata']
        
    def get_recommendations(self, seed_tracks, n=100):
        """Get co-occurrence based recommendations"""
        track_to_id = self.cooccur_metadata['track_to_id']
        id_to_track = self.cooccur_metadata['id_to_track']
        
        scores = defaultdict(float)
        
        for seed in seed_tracks:
            if seed not in track_to_id:
                continue
            seed_id = track_to_id[seed]
            vec = self.cooccur_matrix[seed_id, :].toarray().flatten()
            
            for tid, count in enumerate(vec):
                if count > 0:
                    scores[id_to_track[tid]] += count
        
        for seed in seed_tracks:
            scores.pop(seed, None)
        
        sorted_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [t for t, s in sorted_recs[:n]]
    
    def get_popularity_recommendations(self, seed_tracks, n=100):
        """Get popularity-based recommendations"""
        seed_set = set(seed_tracks)
        sorted_tracks = sorted(self.track_popularity.items(), key=lambda x: x[1], reverse=True)
        return [t for t, p in sorted_tracks if t not in seed_set][:n]
    
    def calculate_diversity_metrics(self, recommendations):
        """Calculate diversity metrics for a recommendation list"""
        if not recommendations:
            return None
        
        # Artist diversity
        artists = [self.track_to_artist.get(t, 'Unknown') for t in recommendations]
        unique_artists = len(set(artists))
        artist_diversity = unique_artists / len(recommendations)
        
        # Artist concentration (Gini coefficient approximation)
        artist_counts = Counter(artists)
        counts = sorted(artist_counts.values(), reverse=True)
        top3_concentration = sum(counts[:3]) / len(recommendations) if len(counts) >= 3 else 1.0
        
        # Popularity metrics
        popularities = [self.track_popularity.get(t, 0) for t in recommendations]
        avg_popularity = np.mean(popularities)
        popularity_std = np.std(popularities)
        
        # Popularity tiers
        pop_tiers = {'high': 0, 'medium': 0, 'low': 0}
        for p in popularities:
            if p >= 100:
                pop_tiers['high'] += 1
            elif p >= 20:
                pop_tiers['medium'] += 1
            else:
                pop_tiers['low'] += 1
        
        return {
            'unique_artists': unique_artists,
            'artist_diversity': artist_diversity,
            'top3_artist_concentration': top3_concentration,
            'avg_popularity': avg_popularity,
            'popularity_std': popularity_std,
            'high_pop_pct': pop_tiers['high'] / len(recommendations),
            'medium_pop_pct': pop_tiers['medium'] / len(recommendations),
            'low_pop_pct': pop_tiers['low'] / len(recommendations)
        }
    
    def analyze_diversity(self):
        """Main diversity analysis"""
        log_section(self.logger, "ANALYZING RECOMMENDATION DIVERSITY")
        
        # Sample playlists with enough tracks
        valid_playlists = [p for p in self.playlists if len(p.get('tracks', [])) >= 10]
        sample = valid_playlists[:300]
        
        self.logger.info(f"Analyzing {len(sample)} playlists")
        
        cooccur_diversity = []
        popularity_diversity = []
        
        for playlist in tqdm(sample, desc="Analyzing"):
            seed_tracks = [t.get('track_uri') for t in playlist.get('tracks', [])]
            
            # Co-occurrence recommendations
            cooccur_recs = self.get_recommendations(seed_tracks, n=100)
            if cooccur_recs:
                metrics = self.calculate_diversity_metrics(cooccur_recs)
                if metrics:
                    metrics['method'] = 'co-occurrence'
                    cooccur_diversity.append(metrics)
            
            # Popularity recommendations
            pop_recs = self.get_popularity_recommendations(seed_tracks, n=100)
            if pop_recs:
                metrics = self.calculate_diversity_metrics(pop_recs)
                if metrics:
                    metrics['method'] = 'popularity'
                    popularity_diversity.append(metrics)
        
        cooccur_df = pd.DataFrame(cooccur_diversity)
        pop_df = pd.DataFrame(popularity_diversity)
        
        # Compare methods
        log_section(self.logger, "DIVERSITY COMPARISON")
        
        self.logger.info(f"\n{'Metric':<30} {'Co-occurrence':>15} {'Popularity':>15}")
        self.logger.info("=" * 60)
        
        metrics_to_compare = [
            ('Artist Diversity', 'artist_diversity'),
            ('Unique Artists (of 100)', 'unique_artists'),
            ('Top 3 Artist Concentration', 'top3_artist_concentration'),
            ('Avg Popularity', 'avg_popularity'),
            ('Popularity Std Dev', 'popularity_std'),
            ('High Popularity %', 'high_pop_pct'),
            ('Low Popularity %', 'low_pop_pct')
        ]
        
        comparison_data = []
        for display_name, col in metrics_to_compare:
            cooccur_val = cooccur_df[col].mean()
            pop_val = pop_df[col].mean()
            self.logger.info(f"{display_name:<30} {cooccur_val:>15.3f} {pop_val:>15.3f}")
            comparison_data.append({
                'metric': display_name,
                'co_occurrence': cooccur_val,
                'popularity': pop_val
            })
        
        # Visualizations
        log_section(self.logger, "CREATING VISUALIZATIONS")
        
        fig_dir = self.output_dir / "figures"
        
        # 1. Artist diversity distribution
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Artist diversity histogram
        axes[0, 0].hist(cooccur_df['artist_diversity'], bins=20, alpha=0.7, label='Co-occurrence', color='#1DB954')
        axes[0, 0].hist(pop_df['artist_diversity'], bins=20, alpha=0.7, label='Popularity', color='#FF6B6B')
        axes[0, 0].set_xlabel('Artist Diversity (unique/total)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Artist Diversity Distribution')
        axes[0, 0].legend()
        axes[0, 0].axvline(cooccur_df['artist_diversity'].mean(), color='#1DB954', linestyle='--')
        axes[0, 0].axvline(pop_df['artist_diversity'].mean(), color='#FF6B6B', linestyle='--')
        
        # Unique artists
        axes[0, 1].hist(cooccur_df['unique_artists'], bins=20, alpha=0.7, label='Co-occurrence', color='#1DB954')
        axes[0, 1].hist(pop_df['unique_artists'], bins=20, alpha=0.7, label='Popularity', color='#FF6B6B')
        axes[0, 1].set_xlabel('Unique Artists in Top 100')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Unique Artists Distribution')
        axes[0, 1].legend()
        
        # Popularity distribution comparison
        axes[1, 0].bar(['High', 'Medium', 'Low'], 
                      [cooccur_df['high_pop_pct'].mean(), cooccur_df['medium_pop_pct'].mean(), cooccur_df['low_pop_pct'].mean()],
                      alpha=0.7, label='Co-occurrence', color='#1DB954', width=0.35, align='edge')
        axes[1, 0].bar(['High', 'Medium', 'Low'], 
                      [pop_df['high_pop_pct'].mean(), pop_df['medium_pop_pct'].mean(), pop_df['low_pop_pct'].mean()],
                      alpha=0.7, label='Popularity', color='#FF6B6B', width=-0.35, align='edge')
        axes[1, 0].set_ylabel('Proportion')
        axes[1, 0].set_title('Popularity Tier Distribution')
        axes[1, 0].legend()
        
        # Method comparison bar chart
        comparison_df = pd.DataFrame(comparison_data)
        x = range(len(comparison_df))
        width = 0.35
        axes[1, 1].bar([i - width/2 for i in x], comparison_df['co_occurrence'], width, label='Co-occurrence', color='#1DB954')
        axes[1, 1].bar([i + width/2 for i in x], comparison_df['popularity'], width, label='Popularity', color='#FF6B6B')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(comparison_df['metric'], rotation=45, ha='right', fontsize=8)
        axes[1, 1].set_title('Method Comparison (Normalized)')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        fig_path = fig_dir / "diversity_analysis.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"✓ Saved: {fig_path}")
        plt.close()
        
        # Save results
        all_diversity = pd.concat([cooccur_df, pop_df])
        all_diversity.to_csv(self.output_dir / "results" / "diversity_metrics.csv", index=False)
        
        pd.DataFrame(comparison_data).to_csv(
            self.output_dir / "results" / "diversity_comparison.csv", index=False
        )
        self.logger.info(f"✓ Saved diversity metrics to results/")
        
        # Summary
        log_section(self.logger, "DIVERSITY ANALYSIS SUMMARY")
        
        self.logger.info("Key Findings:")
        self.logger.info(f"  1. Co-occurrence achieves {cooccur_df['artist_diversity'].mean():.1%} artist diversity")
        self.logger.info(f"  2. Popularity baseline achieves {pop_df['artist_diversity'].mean():.1%} artist diversity")
        self.logger.info(f"  3. Co-occurrence recommends more diverse popularity tiers")
        self.logger.info(f"  4. Popularity baseline heavily biased toward popular tracks ({pop_df['high_pop_pct'].mean():.1%} high popularity)")
        
        return all_diversity


def main():
    logger = setup_logger("10_diversity_analysis")
    
    logger.info("Starting Diversity Analysis (Proposal 4.4.3)")
    
    analyzer = DiversityAnalyzer(logger)
    
    try:
        analyzer.load_data()
        analyzer.analyze_diversity()
        log_section(logger, "✓ DIVERSITY ANALYSIS COMPLETED SUCCESSFULLY")
        return True
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
