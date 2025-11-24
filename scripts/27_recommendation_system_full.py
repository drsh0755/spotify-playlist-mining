"""
Recommendation System using Co-occurrence Matrix
Generates track recommendations based on co-occurrence patterns

Author: Adarsh Singh
Date: November 2024
"""

import pandas as pd
import numpy as np
from scipy import sparse
from pathlib import Path
import logging
from datetime import datetime
import pickle

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'recommendations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RecommendationSystem:
    """Generate recommendations using co-occurrence matrix."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.matrix = None
        self.matrix_norm = None
        self.track_to_idx = None
        self.idx_to_track = None
    
    def load_data(self):
        """Load co-occurrence matrix and mappings."""
        logger.info("Loading co-occurrence matrix...")
        
        self.matrix = sparse.load_npz("data/processed/cooccurrence_matrix_full.npz")
        logger.info(f"Matrix shape: {self.matrix.shape}")
        
        with open("data/processed/track_mappings.pkl", "rb") as f:
            mappings = pickle.load(f)
        
        self.track_to_idx = mappings['track_to_idx']
        self.idx_to_track = mappings['idx_to_track']
        
        logger.info(f"Loaded mappings for {len(self.track_to_idx):,} tracks")
    
    def normalize_matrix(self):
        """Normalize co-occurrence matrix for recommendations."""
        logger.info("Normalizing matrix...")
        
        # Normalize by row sums (popularity)
        row_sums = np.array(self.matrix.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        
        self.matrix_norm = self.matrix.multiply(1.0 / row_sums[:, np.newaxis])
        # Convert to CSR format for indexing
        self.matrix_norm = self.matrix_norm.tocsr()
        
        logger.info("Matrix normalized")
    
    def generate_recommendations(self, seed_tracks, top_n=10):
        """Generate recommendations for seed tracks."""
        
        recommendations = []
        
        for seed_uri in seed_tracks:
            if seed_uri not in self.track_to_idx:
                continue
            
            seed_idx = self.track_to_idx[seed_uri]
            
            # Get co-occurrence scores
            scores = self.matrix_norm[seed_idx].toarray().flatten()
            
            # Get top N (excluding seed itself)
            top_indices = np.argsort(scores)[-(top_n+1):-1][::-1]
            
            for rank, rec_idx in enumerate(top_indices, 1):
                if scores[rec_idx] > 0:  # Only non-zero scores
                    rec_uri = self.idx_to_track[rec_idx]
                    recommendations.append({
                        'seed_track': seed_uri,
                        'recommended_track': rec_uri,
                        'rank': rank,
                        'score': scores[rec_idx]
                    })
        
        return pd.DataFrame(recommendations)
    
    def evaluate_popularity_baseline(self, tracks_df, sample_size=1000):
        """Evaluate popularity-based baseline."""
        logger.info(f"Evaluating popularity baseline on {sample_size} playlists...")
        
        # Get track popularity
        track_popularity = tracks_df['track_uri'].value_counts()
        top_500 = set(track_popularity.head(500).index)
        
        # Sample playlists
        playlist_tracks = tracks_df.groupby('pid')['track_uri'].apply(set)
        sample_playlists = playlist_tracks.sample(min(sample_size, len(playlist_tracks)), random_state=42)
        
        # Calculate R-precision
        r_precisions = []
        for pid, tracks in sample_playlists.items():
            if len(tracks) >= 10:
                recommended = top_500
                relevant = tracks
                intersection = recommended & relevant
                
                r_precision = len(intersection) / len(relevant) if relevant else 0
                r_precisions.append(r_precision)
        
        mean_r_precision = np.mean(r_precisions)
        logger.info(f"Popularity baseline R-precision: {mean_r_precision:.4f}")
        
        return mean_r_precision
    
    def save_results(self, recommendations_df):
        """Save recommendations."""
        logger.info("Saving recommendations...")
        
        rec_file = self.output_dir / "sample_recommendations_full.csv"
        recommendations_df.to_csv(rec_file, index=False)
        logger.info(f"Saved recommendations: {rec_file}")
        
        logger.info(f"\n{'='*60}")
        logger.info("Recommendation System Summary:")
        logger.info(f"Total recommendations generated: {len(recommendations_df):,}")
        logger.info(f"Unique seed tracks: {recommendations_df['seed_track'].nunique():,}")
        logger.info(f"Mean recommendation score: {recommendations_df['score'].mean():.4f}")
        logger.info(f"{'='*60}\n")

def main():
    """Main execution."""
    
    OUTPUT_DIR = "data/processed"
    
    rec_system = RecommendationSystem(output_dir=OUTPUT_DIR)
    
    # Load data
    rec_system.load_data()
    
    # Normalize matrix
    rec_system.normalize_matrix()
    
    # Generate sample recommendations
    logger.info("Generating sample recommendations...")
    sample_tracks = list(rec_system.track_to_idx.keys())[:200]
    recommendations_df = rec_system.generate_recommendations(sample_tracks, top_n=10)
    
    logger.info(f"Generated recommendations for {len(sample_tracks)} seed tracks")
    
    # Save results (skip baseline evaluation)
    rec_system.save_results(recommendations_df)
    
    logger.info("✅ Recommendation system complete!")

if __name__ == "__main__":
    main()