"""
Hybrid Ensemble Recommendation System
Combines multiple models for best performance

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
import warnings
warnings.filterwarnings('ignore')

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'hybrid_ensemble_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HybridEnsemble:
    """Ensemble recommendation system combining multiple approaches."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = self.output_dir / "models"
        
        # Load all models
        self.cooccurrence_matrix = None
        self.svd_data = None
        self.neural_data = None
        self.track_mappings = None
    
    def load_models(self):
        """Load all trained models."""
        logger.info("Loading models...")
        
        # Load co-occurrence matrix
        try:
            self.cooccurrence_matrix = sparse.load_npz("data/processed/cooccurrence_matrix_full.npz")
            logger.info("✅ Loaded co-occurrence matrix")
        except:
            logger.warning("⚠️  Co-occurrence matrix not found")
        
        # Load SVD model
        try:
            with open(self.models_dir / "svd_model.pkl", 'rb') as f:
                self.svd_data = pickle.load(f)
            logger.info("✅ Loaded SVD model")
        except:
            logger.warning("⚠️  SVD model not found")
        
        # Load neural embeddings
        try:
            with open(self.models_dir / "neural_recommender.pkl", 'rb') as f:
                self.neural_data = pickle.load(f)
            logger.info("✅ Loaded neural model")
        except:
            logger.warning("⚠️  Neural model not found")
        
        # Load track mappings
        with open("data/processed/track_mappings.pkl", 'rb') as f:
            self.track_mappings = pickle.load(f)
        
        logger.info(f"Track mappings loaded: {len(self.track_mappings['track_to_idx']):,} tracks")
    
    def get_cooccurrence_recommendations(self, track_uri, top_n=20):
        """Get recommendations from co-occurrence matrix."""
        
        if self.cooccurrence_matrix is None or track_uri not in self.track_mappings['track_to_idx']:
            return {}
        
        idx = self.track_mappings['track_to_idx'][track_uri]
        
        # Normalize row
        row = self.cooccurrence_matrix[idx].toarray().flatten()
        row_sum = row.sum()
        if row_sum > 0:
            row = row / row_sum
        
        # Get top recommendations
        top_indices = np.argsort(row)[::-1][1:top_n+1]  # Exclude self
        
        recommendations = {}
        for rank, rec_idx in enumerate(top_indices):
            if row[rec_idx] > 0:
                rec_uri = self.track_mappings['idx_to_track'][rec_idx]
                recommendations[rec_uri] = {
                    'score': float(row[rec_idx]),
                    'rank': rank + 1
                }
        
        return recommendations
    
    def get_svd_recommendations(self, playlist_tracks, top_n=20):
        """Get recommendations from SVD model."""
        
        if self.svd_data is None:
            return {}
        
        # Find playlist in SVD data
        # For simplicity, average embeddings of seed tracks
        track_factors = self.svd_data['track_factors']
        track_to_idx = self.svd_data['track_to_idx']
        idx_to_track = self.svd_data['idx_to_track']
        
        # Get indices of seed tracks
        seed_indices = [track_to_idx[t] for t in playlist_tracks if t in track_to_idx]
        
        if not seed_indices:
            return {}
        
        # Average embeddings
        avg_embedding = np.mean(track_factors[seed_indices], axis=0)
        
        # Calculate similarities
        similarities = track_factors @ avg_embedding
        
        # Get top recommendations
        existing_set = set(seed_indices)
        recommendations = {}
        
        for idx in np.argsort(similarities)[::-1]:
            if idx not in existing_set:
                rec_uri = idx_to_track[idx]
                recommendations[rec_uri] = {
                    'score': float(similarities[idx]),
                    'rank': len(recommendations) + 1
                }
                if len(recommendations) >= top_n:
                    break
        
        return recommendations
    
    def get_neural_recommendations(self, playlist_tracks, top_n=20):
        """Get recommendations from neural embeddings."""
        
        if self.neural_data is None:
            return {}
        
        embeddings = self.neural_data['embeddings']
        track_to_idx = self.neural_data['track_to_idx']
        
        # Get embeddings for playlist tracks
        playlist_indices = [track_to_idx[t] for t in playlist_tracks if t in track_to_idx]
        
        if not playlist_indices:
            return {}
        
        # Average embeddings
        avg_embedding = np.mean(embeddings[playlist_indices], axis=0)
        
        # Calculate similarities
        similarities = embeddings @ avg_embedding
        similarities = similarities / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(avg_embedding))
        
        # Get top recommendations
        existing_set = set(playlist_indices)
        idx_to_track = {i: t for t, i in track_to_idx.items()}
        
        recommendations = {}
        for idx in np.argsort(similarities)[::-1]:
            if idx not in existing_set:
                rec_uri = idx_to_track[idx]
                recommendations[rec_uri] = {
                    'score': float(similarities[idx]),
                    'rank': len(recommendations) + 1
                }
                if len(recommendations) >= top_n:
                    break
        
        return recommendations
    
    def ensemble_recommendations(self, playlist_tracks, top_n=10, weights=None):
        """
        Combine recommendations from all models using weighted voting.
        
        Args:
            playlist_tracks: List of track URIs in playlist
            top_n: Number of recommendations to return
            weights: Dict with keys 'cooccurrence', 'svd', 'neural' (default: equal weights)
        """
        
        if weights is None:
            weights = {'cooccurrence': 0.4, 'svd': 0.3, 'neural': 0.3}
        
        logger.info(f"Generating ensemble recommendations with weights: {weights}")
        
        # Get recommendations from each model
        all_recommendations = {}
        
        # Co-occurrence (use first track as seed)
        if playlist_tracks and weights.get('cooccurrence', 0) > 0:
            cooc_recs = self.get_cooccurrence_recommendations(playlist_tracks[0], top_n=20)
            for track, data in cooc_recs.items():
                score = data['score'] * weights['cooccurrence'] * (1.0 / data['rank'])
                all_recommendations[track] = all_recommendations.get(track, 0) + score
        
        # SVD
        if weights.get('svd', 0) > 0:
            svd_recs = self.get_svd_recommendations(playlist_tracks, top_n=20)
            for track, data in svd_recs.items():
                score = data['score'] * weights['svd'] * (1.0 / data['rank'])
                all_recommendations[track] = all_recommendations.get(track, 0) + score
        
        # Neural
        if weights.get('neural', 0) > 0:
            neural_recs = self.get_neural_recommendations(playlist_tracks, top_n=20)
            for track, data in neural_recs.items():
                score = data['score'] * weights['neural'] * (1.0 / data['rank'])
                all_recommendations[track] = all_recommendations.get(track, 0) + score
        
        # Sort by combined score
        sorted_recs = sorted(all_recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        return [{'track_uri': track, 'ensemble_score': score} for track, score in sorted_recs]
    
    def evaluate_ensemble(self, tracks_df, sample_size=500):
        """Evaluate ensemble performance."""
        logger.info(f"Evaluating ensemble on {sample_size} playlists...")
        
        playlist_tracks = tracks_df.groupby('pid')['track_uri'].apply(list)
        np.random.seed(42)
        sample_pids = np.random.choice(len(playlist_tracks), min(sample_size, len(playlist_tracks)), replace=False)
        
        precisions = []
        
        for pid in sample_pids:
            tracks = playlist_tracks.iloc[pid]
            
            if len(tracks) < 5:
                continue
            
            # Split
            split_point = int(len(tracks) * 0.8)
            input_tracks = tracks[:split_point]
            target_tracks = set(tracks[split_point:])
            
            # Generate ensemble recommendations
            recs = self.ensemble_recommendations(input_tracks, top_n=10)
            rec_tracks = set([r['track_uri'] for r in recs])
            
            # Calculate precision
            precision = len(rec_tracks & target_tracks) / 10 if target_tracks else 0
            precisions.append(precision)
        
        mean_precision = np.mean(precisions)
        logger.info(f"Ensemble Precision@10: {mean_precision:.4f}")
        
        return {'mean_precision': mean_precision, 'test_size': len(precisions)}

def main():
    """Main execution."""
    
    OUTPUT_DIR = "data/processed"
    
    ensemble = HybridEnsemble(output_dir=OUTPUT_DIR)
    
    # Load all models
    ensemble.load_models()
    
    # Load tracks for evaluation
    tracks_df = pd.read_parquet("data/processed/tracks_full_mpd.parquet")
    
    # Evaluate ensemble
    results = ensemble.evaluate_ensemble(tracks_df, sample_size=500)
    
    logger.info("\n" + "="*60)
    logger.info("Hybrid Ensemble System Complete!")
    logger.info(f"Ensemble Precision@10: {results['mean_precision']:.4f}")
    logger.info("="*60)
    logger.info("✅ Hybrid ensemble system ready!")

if __name__ == "__main__":
    main()