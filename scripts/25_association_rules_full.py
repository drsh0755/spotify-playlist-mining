"""
Association Rule Mining on Full MPD
Discovers frequent itemsets and association rules from popular tracks

Author: Adarsh Singh
Date: November 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import pickle
from collections import defaultdict, Counter

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'association_rules_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AssociationRuleMiner:
    """Mine association rules from playlist-track co-occurrences."""
    
    def __init__(self, output_dir, min_support=100, min_confidence=0.1):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_support = min_support  # Minimum playlist count
        self.min_confidence = min_confidence
        
        self.rules = None
    
    def load_data(self):
        """Load tracks and mappings."""
        logger.info("Loading data...")
        
        # Load tracks
        tracks_df = pd.read_parquet("data/processed/tracks_full_mpd.parquet")
        logger.info(f"Loaded {len(tracks_df):,} track entries")
        
        # Load track mappings
        with open("data/processed/track_mappings.pkl", "rb") as f:
            mappings = pickle.load(f)
        
        self.track_to_idx = mappings['track_to_idx']
        self.idx_to_track = mappings['idx_to_track']
        self.track_counts = mappings['track_counts']
        
        logger.info(f"Loaded mappings for {len(self.track_to_idx):,} popular tracks")
        
        # Filter to only popular tracks
        popular_tracks = set(self.track_to_idx.keys())
        tracks_df = tracks_df[tracks_df['track_uri'].isin(popular_tracks)].copy()
        logger.info(f"Filtered to {len(tracks_df):,} entries from {tracks_df['pid'].nunique():,} playlists")
        
        return tracks_df
    
    def build_cooccurrence_stats(self, tracks_df):
        """Build co-occurrence statistics."""
        logger.info("Building co-occurrence statistics...")
        
        # Group tracks by playlist
        playlist_tracks = tracks_df.groupby('pid')['track_uri'].apply(list)
        total_playlists = len(playlist_tracks)
        
        logger.info(f"Processing {total_playlists:,} playlists")
        
        # Count track pairs
        pair_counts = defaultdict(int)
        
        for tracks in playlist_tracks:
            # Generate pairs
            for i, t1 in enumerate(tracks):
                for t2 in tracks[i+1:]:
                    # Ensure consistent ordering
                    if t1 > t2:
                        t1, t2 = t2, t1
                    pair_counts[(t1, t2)] += 1
        
        logger.info(f"Found {len(pair_counts):,} unique track pairs")
        
        return pair_counts, total_playlists
    
    def generate_rules(self, pair_counts, total_playlists):
        """Generate association rules from co-occurrence statistics."""
        logger.info(f"Generating association rules (min_support={self.min_support}, min_confidence={self.min_confidence})...")
        
        rules = []
        
        for (t1, t2), count in pair_counts.items():
            if count >= self.min_support:
                # Calculate metrics
                support = count / total_playlists
                
                # Confidence: P(t2|t1) = count(t1,t2) / count(t1)
                confidence_12 = count / self.track_counts[t1] if t1 in self.track_counts else 0
                confidence_21 = count / self.track_counts[t2] if t2 in self.track_counts else 0
                
                # Lift: support / (P(t1) * P(t2))
                p_t1 = self.track_counts.get(t1, 0) / total_playlists
                p_t2 = self.track_counts.get(t2, 0) / total_playlists
                lift = support / (p_t1 * p_t2) if p_t1 > 0 and p_t2 > 0 else 0
                
                # Add rule if confidence threshold met
                if confidence_12 >= self.min_confidence:
                    rules.append({
                        'antecedent': t1,
                        'consequent': t2,
                        'support': support,
                        'confidence': confidence_12,
                        'lift': lift,
                        'count': count
                    })
                
                if confidence_21 >= self.min_confidence and t1 != t2:
                    rules.append({
                        'antecedent': t2,
                        'consequent': t1,
                        'support': support,
                        'confidence': confidence_21,
                        'lift': lift,
                        'count': count
                    })
        
        self.rules = pd.DataFrame(rules)
        logger.info(f"Generated {len(self.rules):,} association rules")
        
        return self.rules
    
    def save_results(self):
        """Save results."""
        logger.info("Saving results...")
        
        # Save top rules by lift
        rules_file = self.output_dir / "association_rules_full.csv"
        top_rules = self.rules.nlargest(10000, 'lift')
        top_rules.to_csv(rules_file, index=False)
        logger.info(f"Saved top 10,000 rules: {rules_file}")
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("Association Rule Mining Summary:")
        logger.info(f"Total rules generated: {len(self.rules):,}")
        logger.info(f"Rules saved: {len(top_rules):,}")
        
        logger.info(f"\nSupport statistics:")
        logger.info(f"  Mean: {self.rules['support'].mean():.4f}")
        logger.info(f"  Median: {self.rules['support'].median():.4f}")
        
        logger.info(f"\nConfidence statistics:")
        logger.info(f"  Mean: {self.rules['confidence'].mean():.4f}")
        logger.info(f"  Median: {self.rules['confidence'].median():.4f}")
        
        logger.info(f"\nLift statistics:")
        logger.info(f"  Mean: {self.rules['lift'].mean():.2f}")
        logger.info(f"  Median: {self.rules['lift'].median():.2f}")
        
        logger.info(f"\nTop 5 rules by lift:")
        for idx, row in self.rules.nlargest(5, 'lift').iterrows():
            logger.info(f"  {row['antecedent'][:40]}...")
            logger.info(f"    => {row['consequent'][:40]}...")
            logger.info(f"    Lift: {row['lift']:.2f}, Confidence: {row['confidence']:.3f}, Support: {row['support']:.4f}")
        
        logger.info(f"{'='*60}\n")

def main():
    """Main execution."""
    
    OUTPUT_DIR = "data/processed"
    MIN_SUPPORT = 100  # Track pair appears in 100+ playlists
    MIN_CONFIDENCE = 0.1  # 10% confidence threshold
    
    miner = AssociationRuleMiner(
        output_dir=OUTPUT_DIR,
        min_support=MIN_SUPPORT,
        min_confidence=MIN_CONFIDENCE
    )
    
    # Load data
    tracks_df = miner.load_data()
    
    # Build co-occurrence statistics
    pair_counts, total_playlists = miner.build_cooccurrence_stats(tracks_df)
    
    # Generate association rules
    miner.generate_rules(pair_counts, total_playlists)
    
    # Save results
    miner.save_results()
    
    logger.info("✅ Association rule mining complete!")

if __name__ == "__main__":
    main()