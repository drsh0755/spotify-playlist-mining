"""
Model Comparison and Benchmarking
Compare all recommendation models on standardized metrics

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
import json
import warnings
warnings.filterwarnings('ignore')

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'model_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelComparison:
    """Compare and benchmark all recommendation models."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []
    
    def load_test_data(self, sample_size=1000):
        """Load test playlists."""
        logger.info("Loading test data...")
        
        tracks_df = pd.read_parquet("data/processed/tracks_full_mpd.parquet")
        playlist_tracks = tracks_df.groupby('pid')['track_uri'].apply(list)
        
        # Sample test playlists
        np.random.seed(42)
        test_indices = np.random.choice(len(playlist_tracks), 
                                       min(sample_size, len(playlist_tracks)), 
                                       replace=False)
        
        test_playlists = []
        for idx in test_indices:
            tracks = playlist_tracks.iloc[idx]
            if len(tracks) >= 10:
                # Split: 80% input, 20% target
                split = int(len(tracks) * 0.8)
                test_playlists.append({
                    'pid': playlist_tracks.index[idx],
                    'input_tracks': tracks[:split],
                    'target_tracks': set(tracks[split:])
                })
        
        logger.info(f"Prepared {len(test_playlists)} test playlists")
        return test_playlists
    
    def evaluate_popularity_baseline(self, test_playlists, tracks_df):
        """Evaluate popularity-based baseline."""
        logger.info("\n" + "="*60)
        logger.info("Evaluating Popularity Baseline")
        logger.info("="*60)
        
        # Get top 500 most popular tracks
        from collections import Counter
        track_list = tracks_df['track_uri'].tolist()
        track_counter = Counter(track_list)
        top_tracks = set([t for t, c in track_counter.most_common(500)])
        
        precisions = []
        for playlist in test_playlists:
            recommended = top_tracks
            target = playlist['target_tracks']
            hits = recommended & target
            precision = len(hits) / 10 if len(hits) <= 10 else len(hits) / len(recommended)
            precisions.append(precision)
        
        mean_precision = np.mean(precisions)
        
        self.results.append({
            'model': 'Popularity Baseline',
            'precision@10': mean_precision,
            'test_size': len(precisions)
        })
        
        logger.info(f"Precision@10: {mean_precision:.4f}")
        return mean_precision
    
    def evaluate_cooccurrence(self, test_playlists):
        """Evaluate co-occurrence model."""
        logger.info("\n" + "="*60)
        logger.info("Evaluating Co-occurrence Model")
        logger.info("="*60)
        
        try:
            matrix = sparse.load_npz("data/processed/cooccurrence_matrix_full.npz")
            with open("data/processed/track_mappings.pkl", "rb") as f:
                mappings = pickle.load(f)
            
            precisions = []
            for playlist in test_playlists:
                # Use first track as seed
                seed_track = playlist['input_tracks'][0]
                if seed_track not in mappings['track_to_idx']:
                    continue
                
                idx = mappings['track_to_idx'][seed_track]
                scores = matrix[idx].toarray().flatten()
                
                # Get top 10
                top_indices = np.argsort(scores)[::-1][1:11]
                recommended = set([mappings['idx_to_track'][i] for i in top_indices if scores[i] > 0])
                
                target = playlist['target_tracks']
                hits = recommended & target
                precision = len(hits) / 10
                precisions.append(precision)
            
            mean_precision = np.mean(precisions)
            
            self.results.append({
                'model': 'Co-occurrence',
                'precision@10': mean_precision,
                'test_size': len(precisions)
            })
            
            logger.info(f"Precision@10: {mean_precision:.4f}")
            return mean_precision
            
        except Exception as e:
            logger.warning(f"Co-occurrence model failed: {e}")
            return 0.0
    
    def evaluate_svd(self, test_playlists):
        """Evaluate SVD model."""
        logger.info("\n" + "="*60)
        logger.info("Evaluating SVD Model")
        logger.info("="*60)
        
        try:
            with open("data/processed/models/svd_model.pkl", "rb") as f:
                svd_data = pickle.load(f)
            
            track_factors = svd_data['track_factors']
            track_to_idx = svd_data['track_to_idx']
            idx_to_track = svd_data['idx_to_track']
            
            precisions = []
            for playlist in test_playlists:
                # Get indices for input tracks
                input_indices = [track_to_idx[t] for t in playlist['input_tracks'] if t in track_to_idx]
                if not input_indices:
                    continue
                
                # Average embeddings
                avg_embedding = np.mean(track_factors[input_indices], axis=0)
                
                # Calculate similarities
                similarities = track_factors @ avg_embedding
                
                # Get top 10
                existing_set = set(input_indices)
                recommended_indices = []
                for idx in np.argsort(similarities)[::-1]:
                    if idx not in existing_set:
                        recommended_indices.append(idx)
                        if len(recommended_indices) >= 10:
                            break
                
                recommended = set([idx_to_track[i] for i in recommended_indices])
                target = playlist['target_tracks']
                hits = recommended & target
                precision = len(hits) / 10
                precisions.append(precision)
            
            mean_precision = np.mean(precisions)
            
            self.results.append({
                'model': 'SVD',
                'precision@10': mean_precision,
                'test_size': len(precisions)
            })
            
            logger.info(f"Precision@10: {mean_precision:.4f}")
            return mean_precision
            
        except Exception as e:
            logger.warning(f"SVD model failed: {e}")
            return 0.0
    
    def evaluate_neural(self, test_playlists):
        """Evaluate neural model."""
        logger.info("\n" + "="*60)
        logger.info("Evaluating Neural Model")
        logger.info("="*60)
        
        try:
            with open("data/processed/models/neural_recommender.pkl", "rb") as f:
                neural_data = pickle.load(f)
            
            embeddings = neural_data['embeddings']
            track_to_idx = neural_data['track_to_idx']
            
            precisions = []
            for playlist in test_playlists:
                # Get indices
                input_indices = [track_to_idx[t] for t in playlist['input_tracks'] if t in track_to_idx]
                if not input_indices:
                    continue
                
                # Average embeddings
                avg_embedding = np.mean(embeddings[input_indices], axis=0)
                
                # Calculate similarities (cosine)
                similarities = embeddings @ avg_embedding
                norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(avg_embedding)
                similarities = similarities / norms
                
                # Get top 10
                existing_set = set(input_indices)
                idx_to_track = {i: t for t, i in track_to_idx.items()}
                
                recommended = []
                for idx in np.argsort(similarities)[::-1]:
                    if idx not in existing_set:
                        recommended.append(idx_to_track[idx])
                        if len(recommended) >= 10:
                            break
                
                recommended_set = set(recommended)
                target = playlist['target_tracks']
                hits = recommended_set & target
                precision = len(hits) / 10
                precisions.append(precision)
            
            mean_precision = np.mean(precisions)
            
            self.results.append({
                'model': 'Neural (PCA)',
                'precision@10': mean_precision,
                'test_size': len(precisions)
            })
            
            logger.info(f"Precision@10: {mean_precision:.4f}")
            return mean_precision
            
        except Exception as e:
            logger.warning(f"Neural model failed: {e}")
            return 0.0
    
    def save_results(self):
        """Save comparison results."""
        logger.info("\n" + "="*60)
        logger.info("FINAL MODEL COMPARISON")
        logger.info("="*60)
        
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('precision@10', ascending=False)
        
        # Save to CSV
        results_file = self.output_dir / "model_comparison_results.csv"
        results_df.to_csv(results_file, index=False)
        logger.info(f"\nSaved results: {results_file}")
        
        # Print comparison table
        logger.info("\nModel Rankings:")
        for idx, row in results_df.iterrows():
            logger.info(f"  {row['model']:20s} | Precision@10: {row['precision@10']:.4f} | Tests: {row['test_size']}")
        
        # Save as JSON
        results_json = self.output_dir / "model_comparison_results.json"
        with open(results_json, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info("Best Model: " + results_df.iloc[0]['model'])
        logger.info(f"Best Precision@10: {results_df.iloc[0]['precision@10']:.4f}")
        logger.info(f"{'='*60}\n")

def main():
    """Main execution."""
    
    OUTPUT_DIR = "data/processed"
    TEST_SIZE = 1000
    
    comparator = ModelComparison(output_dir=OUTPUT_DIR)
    
    # Load test data
    test_playlists = comparator.load_test_data(sample_size=TEST_SIZE)
    tracks_df = pd.read_parquet("data/processed/tracks_full_mpd.parquet")
    
    # Evaluate all models
    comparator.evaluate_popularity_baseline(test_playlists, tracks_df)
    comparator.evaluate_cooccurrence(test_playlists)
    comparator.evaluate_svd(test_playlists)
    comparator.evaluate_neural(test_playlists)
    
    # Save results
    comparator.save_results()
    
    logger.info("✅ Model comparison complete!")

if __name__ == "__main__":
    main()