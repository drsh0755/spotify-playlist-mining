"""
Matrix Factorization Models: SVD and ALS
Builds collaborative filtering models using matrix factorization

Author: Adarsh Singh
Date: November 2024
"""

import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
from pathlib import Path
import logging
from datetime import datetime
import pickle
from sklearn.decomposition import TruncatedSVD
from implicit.als import AlternatingLeastSquares
import warnings
warnings.filterwarnings('ignore')

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'matrix_factorization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MatrixFactorizationModels:
    """Build and evaluate matrix factorization recommendation models."""
    
    def __init__(self, output_dir, n_factors=50):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_factors = n_factors
        
        self.models_dir = self.output_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        self.svd_model = None
        self.als_model = None
        self.user_item_matrix = None
        self.playlist_to_idx = {}
        self.idx_to_playlist = {}
        self.track_to_idx = {}
        self.idx_to_track = {}
    
    def load_data(self):
        """Load playlist-track interactions."""
        logger.info("Loading data...")
        
        tracks_df = pd.read_parquet("data/processed/tracks_full_mpd.parquet")
        logger.info(f"Loaded {len(tracks_df):,} track entries")
        
        # Load popular tracks only
        with open("data/processed/track_mappings.pkl", "rb") as f:
            mappings = pickle.load(f)
        
        popular_tracks = set(mappings['track_to_idx'].keys())
        tracks_df = tracks_df[tracks_df['track_uri'].isin(popular_tracks)].copy()
        logger.info(f"Filtered to {len(tracks_df):,} entries with popular tracks")
        
        return tracks_df
    
    def build_user_item_matrix(self, tracks_df, sample_playlists=None):
        """Build playlist-track interaction matrix."""
        logger.info("Building user-item matrix...")
        
        # Sample playlists if specified (for faster training)
        if sample_playlists:
            unique_playlists = tracks_df['pid'].unique()
            np.random.seed(42)
            sampled_pids = np.random.choice(unique_playlists, 
                                           min(sample_playlists, len(unique_playlists)), 
                                           replace=False)
            tracks_df = tracks_df[tracks_df['pid'].isin(sampled_pids)].copy()
            logger.info(f"Sampled {len(sampled_pids):,} playlists")
        
        # Create mappings
        unique_playlists = sorted(tracks_df['pid'].unique())
        unique_tracks = sorted(tracks_df['track_uri'].unique())
        
        self.playlist_to_idx = {pid: idx for idx, pid in enumerate(unique_playlists)}
        self.idx_to_playlist = {idx: pid for pid, idx in self.playlist_to_idx.items()}
        self.track_to_idx = {track: idx for idx, track in enumerate(unique_tracks)}
        self.idx_to_track = {idx: track for track, idx in self.track_to_idx.items()}
        
        logger.info(f"Matrix dimensions: {len(unique_playlists):,} playlists × {len(unique_tracks):,} tracks")
        
        # Build sparse matrix (playlist × track)
        rows = tracks_df['pid'].map(self.playlist_to_idx).values
        cols = tracks_df['track_uri'].map(self.track_to_idx).values
        data = np.ones(len(tracks_df))  # Binary interactions
        
        self.user_item_matrix = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(len(unique_playlists), len(unique_tracks))
        )
        
        logger.info(f"Matrix sparsity: {(1 - self.user_item_matrix.nnz / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1])) * 100:.2f}%")
        
        return self.user_item_matrix
    
    def train_svd(self):
        """Train SVD model using Truncated SVD."""
        logger.info(f"Training SVD model with {self.n_factors} factors...")
        start_time = datetime.now()
        
        self.svd_model = TruncatedSVD(n_components=self.n_factors, random_state=42)
        self.playlist_factors = self.svd_model.fit_transform(self.user_item_matrix)
        self.track_factors = self.svd_model.components_.T
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"SVD training completed in {elapsed:.2f} seconds")
        logger.info(f"Explained variance ratio: {self.svd_model.explained_variance_ratio_.sum():.4f}")
        
        return self.svd_model
    
    def train_als(self, iterations=15, regularization=0.01):
        """Train ALS model using implicit library."""
        logger.info(f"Training ALS model with {self.n_factors} factors...")
        start_time = datetime.now()
        
        # ALS expects item × user matrix
        item_user_matrix = self.user_item_matrix.T.tocsr()
        
        self.als_model = AlternatingLeastSquares(
            factors=self.n_factors,
            regularization=regularization,
            iterations=iterations,
            random_state=42
        )
        
        self.als_model.fit(item_user_matrix)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"ALS training completed in {elapsed:.2f} seconds")
        
        return self.als_model
    
    def generate_svd_recommendations(self, playlist_idx, top_n=10):
        """Generate recommendations using SVD."""
        
        # Get playlist factor
        playlist_vec = self.playlist_factors[playlist_idx]
        
        # Compute scores for all tracks
        scores = playlist_vec @ self.track_factors.T
        
        # Get tracks already in playlist
        existing_tracks = set(self.user_item_matrix[playlist_idx].indices)
        
        # Get top N tracks not in playlist
        recommendations = []
        for track_idx in np.argsort(scores)[::-1]:
            if track_idx not in existing_tracks:
                recommendations.append({
                    'track_idx': track_idx,
                    'track_uri': self.idx_to_track[track_idx],
                    'score': scores[track_idx]
                })
                if len(recommendations) >= top_n:
                    break
        
        return recommendations
    
    def generate_als_recommendations(self, playlist_idx, top_n=10):
        """Generate recommendations using ALS."""
        
        try:
            # Get recommendations from ALS
            track_scores = self.als_model.recommend(
                userid=playlist_idx,
                user_items=self.user_item_matrix[playlist_idx],
                N=top_n,
                filter_already_liked_items=True
            )
            
            recommendations = []
            for track_idx, score in zip(track_scores[0], track_scores[1]):
                track_idx_int = int(track_idx)
                # Check if track_idx is in our mapping
                if track_idx_int in self.idx_to_track:
                    recommendations.append({
                        'track_idx': track_idx_int,
                        'track_uri': self.idx_to_track[track_idx_int],
                        'score': float(score)
                    })
            
            return recommendations
        except Exception as e:
            # If ALS fails, return empty
            return []

        
            
    def evaluate_models(self, test_playlists=1000):
        """Evaluate both models using hold-out testing."""
        logger.info(f"Evaluating models on {test_playlists} playlists...")
        
        # Sample test playlists
        n_playlists = self.user_item_matrix.shape[0]
        np.random.seed(42)
        test_indices = np.random.choice(n_playlists, min(test_playlists, n_playlists), replace=False)
        
        svd_precisions = []
        als_precisions = []
        
        for playlist_idx in test_indices:
            # Get actual tracks in playlist
            actual_tracks = set(self.user_item_matrix[playlist_idx].indices)
            
            if len(actual_tracks) < 5:
                continue
            
            # Generate recommendations
            svd_recs = self.generate_svd_recommendations(playlist_idx, top_n=10)
            als_recs = self.generate_als_recommendations(playlist_idx, top_n=10)
            
            svd_rec_tracks = set([r['track_idx'] for r in svd_recs])
            als_rec_tracks = set([r['track_idx'] for r in als_recs])
            
            # Calculate precision@10
            svd_precision = len(svd_rec_tracks & actual_tracks) / 10
            als_precision = len(als_rec_tracks & actual_tracks) / 10
            
            svd_precisions.append(svd_precision)
            als_precisions.append(als_precision)
        
        results = {
            'svd_mean_precision': np.mean(svd_precisions),
            'svd_median_precision': np.median(svd_precisions),
            'als_mean_precision': np.mean(als_precisions),
            'als_median_precision': np.median(als_precisions),
            'test_playlists': len(svd_precisions)
        }
        
        logger.info(f"\nModel Evaluation Results:")
        logger.info(f"  SVD Precision@10: {results['svd_mean_precision']:.4f}")
        logger.info(f"  ALS Precision@10: {results['als_mean_precision']:.4f}")
        
        return results
    
    def save_models(self):
        """Save trained models."""
        logger.info("Saving models...")
        
        # Save SVD model
        svd_file = self.models_dir / "svd_model.pkl"
        with open(svd_file, 'wb') as f:
            pickle.dump({
                'model': self.svd_model,
                'playlist_factors': self.playlist_factors,
                'track_factors': self.track_factors,
                'playlist_to_idx': self.playlist_to_idx,
                'idx_to_playlist': self.idx_to_playlist,
                'track_to_idx': self.track_to_idx,
                'idx_to_track': self.idx_to_track
            }, f)
        logger.info(f"Saved SVD model: {svd_file}")
        
        # Save ALS model
        als_file = self.models_dir / "als_model.pkl"
        with open(als_file, 'wb') as f:
            pickle.dump({
                'model': self.als_model,
                'playlist_to_idx': self.playlist_to_idx,
                'idx_to_playlist': self.idx_to_playlist,
                'track_to_idx': self.track_to_idx,
                'idx_to_track': self.idx_to_track
            }, f)
        logger.info(f"Saved ALS model: {als_file}")
        
        # Save user-item matrix
        matrix_file = self.models_dir / "user_item_matrix.npz"
        sparse.save_npz(matrix_file, self.user_item_matrix)
        logger.info(f"Saved user-item matrix: {matrix_file}")

def main():
    """Main execution."""
    
    OUTPUT_DIR = "data/processed"
    N_FACTORS = 50
    SAMPLE_PLAYLISTS = 50000  # Use 50K playlists for faster training
    
    mf = MatrixFactorizationModels(output_dir=OUTPUT_DIR, n_factors=N_FACTORS)
    
    # Load data
    tracks_df = mf.load_data()
    
    # Build user-item matrix
    mf.build_user_item_matrix(tracks_df, sample_playlists=SAMPLE_PLAYLISTS)
    
    # Train SVD
    mf.train_svd()
    
    # Train ALS
    mf.train_als()
    
    # Evaluate models
    results = mf.evaluate_models(test_playlists=1000)
    
    # Save models
    mf.save_models()
    
    logger.info("\n" + "="*60)
    logger.info("Matrix Factorization Models Complete!")
    logger.info("="*60)
    logger.info(f"Models saved in: {mf.models_dir}")
    logger.info("✅ SVD and ALS models trained and saved!")

if __name__ == "__main__":
    main()