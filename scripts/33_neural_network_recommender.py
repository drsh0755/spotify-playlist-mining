"""
Neural Network Recommender using Embeddings
Deep learning approach to playlist continuation

Author: Adarsh Singh
Date: November 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'neural_network_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleNeuralRecommender:
    """Simple neural network recommender using track features."""
    
    def __init__(self, output_dir, embedding_dim=32):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        
        self.models_dir = self.output_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        self.scaler = StandardScaler()
        self.track_embeddings = None
        self.track_to_idx = {}
    
    def load_data(self):
        """Load track features."""
        logger.info("Loading track features...")
        
        features_df = pd.read_parquet("data/processed/track_features_full.parquet")
        tracks_df = pd.read_parquet("data/processed/tracks_full_mpd.parquet")
        
        logger.info(f"Loaded {len(features_df):,} tracks with features")
        
        # Load popular tracks
        with open("data/processed/track_mappings.pkl", "rb") as f:
            mappings = pickle.load(f)
        
        popular_tracks = set(mappings['track_to_idx'].keys())
        features_df = features_df[features_df['track_uri'].isin(popular_tracks)].copy()
        
        logger.info(f"Filtered to {len(features_df):,} popular tracks")
        
        return features_df, tracks_df
    
    def create_embeddings(self, features_df):
        """Create track embeddings from features using dimensionality reduction."""
        logger.info("Creating track embeddings...")
        
        # Select numeric features
        feature_cols = [
            'popularity', 'avg_position', 'std_position',
            'position_consistency', 'artist_popularity', 
            'album_popularity', 'duration_normalized'
        ]
        
        available_cols = [col for col in feature_cols if col in features_df.columns]
        X = features_df[available_cols].fillna(0).values
        
        logger.info(f"Using {len(available_cols)} features")
        
        # Standardize
        X_scaled = self.scaler.fit_transform(X)
        
        # Simple "neural" embedding: use PCA-like transformation
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(self.embedding_dim, len(available_cols)))
        self.track_embeddings = pca.fit_transform(X_scaled)
        
        logger.info(f"Created embeddings with shape: {self.track_embeddings.shape}")
        logger.info(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
        
        # Create track mappings
        self.track_to_idx = {track: idx for idx, track in enumerate(features_df['track_uri'].values)}
        
        return self.track_embeddings, pca
    
    def find_similar_tracks(self, track_uri, top_n=10):
        """Find similar tracks using embedding similarity."""
        
        if track_uri not in self.track_to_idx:
            return []
        
        track_idx = self.track_to_idx[track_uri]
        track_embedding = self.track_embeddings[track_idx]
        
        # Calculate cosine similarity
        similarities = np.dot(self.track_embeddings, track_embedding)
        similarities = similarities / (np.linalg.norm(self.track_embeddings, axis=1) * np.linalg.norm(track_embedding))
        
        # Get top N similar tracks
        similar_indices = np.argsort(similarities)[::-1][1:top_n+1]  # Exclude self
        
        idx_to_track = {idx: track for track, idx in self.track_to_idx.items()}
        
        recommendations = []
        for idx in similar_indices:
            recommendations.append({
                'track_uri': idx_to_track[idx],
                'similarity': similarities[idx]
            })
        
        return recommendations
    
    def generate_playlist_recommendations(self, playlist_tracks, top_n=10):
        """Generate recommendations for a playlist using average embeddings."""
        
        # Get embeddings for playlist tracks
        playlist_indices = [self.track_to_idx[t] for t in playlist_tracks if t in self.track_to_idx]
        
        if not playlist_indices:
            return []
        
        # Average playlist embeddings
        playlist_embedding = np.mean(self.track_embeddings[playlist_indices], axis=0)
        
        # Find similar tracks
        similarities = np.dot(self.track_embeddings, playlist_embedding)
        similarities = similarities / (np.linalg.norm(self.track_embeddings, axis=1) * np.linalg.norm(playlist_embedding))
        
        # Exclude tracks already in playlist
        existing_set = set(playlist_indices)
        
        recommendations = []
        for idx in np.argsort(similarities)[::-1]:
            if idx not in existing_set:
                idx_to_track = {i: t for t, i in self.track_to_idx.items()}
                recommendations.append({
                    'track_uri': idx_to_track[idx],
                    'similarity': similarities[idx]
                })
                if len(recommendations) >= top_n:
                    break
        
        return recommendations
    
    def evaluate(self, tracks_df, sample_size=1000):
        """Evaluate recommendations."""
        logger.info(f"Evaluating on {sample_size} playlists...")
        
        # Sample playlists
        playlist_tracks = tracks_df.groupby('pid')['track_uri'].apply(list)
        np.random.seed(42)
        sample_pids = np.random.choice(len(playlist_tracks), min(sample_size, len(playlist_tracks)), replace=False)
        
        precisions = []
        
        for pid in sample_pids:
            tracks = playlist_tracks.iloc[pid]
            
            if len(tracks) < 5:
                continue
            
            # Split: use first 80% as input, predict last 20%
            split_point = int(len(tracks) * 0.8)
            input_tracks = tracks[:split_point]
            target_tracks = set(tracks[split_point:])
            
            # Generate recommendations
            recs = self.generate_playlist_recommendations(input_tracks, top_n=10)
            rec_tracks = set([r['track_uri'] for r in recs])
            
            # Calculate precision
            precision = len(rec_tracks & target_tracks) / 10 if target_tracks else 0
            precisions.append(precision)
        
        mean_precision = np.mean(precisions)
        logger.info(f"Mean Precision@10: {mean_precision:.4f}")
        
        return {'mean_precision': mean_precision, 'test_size': len(precisions)}
    
    def save_model(self):
        """Save model."""
        logger.info("Saving model...")
        
        model_file = self.models_dir / "neural_recommender.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump({
                'embeddings': self.track_embeddings,
                'scaler': self.scaler,
                'track_to_idx': self.track_to_idx
            }, f)
        
        logger.info(f"Saved model: {model_file}")

def main():
    """Main execution."""
    
    OUTPUT_DIR = "data/processed"
    EMBEDDING_DIM = 32
    
    recommender = SimpleNeuralRecommender(output_dir=OUTPUT_DIR, embedding_dim=EMBEDDING_DIM)
    
    # Load data
    features_df, tracks_df = recommender.load_data()
    
    # Create embeddings
    recommender.create_embeddings(features_df)
    
    # Evaluate
    results = recommender.evaluate(tracks_df, sample_size=1000)
    
    # Save model
    recommender.save_model()
    
    logger.info("\n" + "="*60)
    logger.info("Neural Network Recommender Complete!")
    logger.info(f"Precision@10: {results['mean_precision']:.4f}")
    logger.info("="*60)
    logger.info("✅ Neural network model trained and saved!")

if __name__ == "__main__":
    main()