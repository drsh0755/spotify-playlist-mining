"""
Track Clustering using K-means on Full MPD
Clusters tracks based on audio features and popularity

Author: Adarsh Singh
Date: November 2024
"""

import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'clustering_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrackClusterer:
    """Cluster tracks using K-means."""
    
    def __init__(self, output_dir, n_clusters=5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_clusters = n_clusters
        
        self.scaler = StandardScaler()
        self.kmeans = None
        self.features_df = None
    
    def load_features(self):
        """Load track features."""
        logger.info("Loading track features...")
        
        self.features_df = pd.read_parquet("data/processed/track_features_full.parquet")
        logger.info(f"Loaded {len(self.features_df):,} tracks")
        
        return self.features_df
    
    def prepare_features(self):
        """Prepare features for clustering."""
        logger.info("Preparing features...")
        
        # Select numeric features
        feature_cols = [
            'popularity',
            'avg_position',
            'std_position',
            'position_consistency',
            'artist_popularity',
            'album_popularity',
            'duration_normalized'
        ]
        
        # Check which columns exist
        available_cols = [col for col in feature_cols if col in self.features_df.columns]
        logger.info(f"Using {len(available_cols)} features: {', '.join(available_cols)}")
        
        # Extract and fill NaN
        X = self.features_df[available_cols].fillna(0).values
        logger.info(f"Feature matrix shape: {X.shape}")
        
        # Standardize
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, available_cols
    
    def cluster(self, X_scaled):
        """Run K-means clustering."""
        logger.info(f"Running K-means clustering with {self.n_clusters} clusters...")
        start_time = datetime.now()
        
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            batch_size=10000,
            max_iter=100
        )
        
        self.features_df['cluster'] = self.kmeans.fit_predict(X_scaled)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Clustering completed in {elapsed:.2f} seconds")
        
        return self.features_df
    
    def analyze_clusters(self, feature_cols):
        """Analyze cluster characteristics."""
        logger.info("Analyzing clusters...")
        
        # Cluster sizes
        cluster_sizes = self.features_df['cluster'].value_counts().sort_index()
        
        logger.info(f"\nCluster distribution:")
        for cluster in range(self.n_clusters):
            count = cluster_sizes.get(cluster, 0)
            pct = (count / len(self.features_df)) * 100
            logger.info(f"  Cluster {cluster}: {count:,} tracks ({pct:.1f}%)")
        
        # Cluster profiles
        cluster_profiles = self.features_df.groupby('cluster')[feature_cols].mean()
        
        logger.info(f"\nCluster profiles (mean values):")
        for cluster in range(self.n_clusters):
            logger.info(f"\n  Cluster {cluster}:")
            for col in feature_cols[:4]:  # Show first 4 features
                value = cluster_profiles.loc[cluster, col]
                logger.info(f"    {col}: {value:.3f}")
        
        return cluster_profiles
    
    def save_results(self, cluster_profiles):
        """Save clustering results."""
        logger.info("Saving results...")
        
        # Save cluster assignments
        cluster_file = self.output_dir / "track_clusters_full.csv"
        self.features_df[['track_uri', 'track_name', 'artist_name', 'cluster']].to_csv(
            cluster_file, index=False
        )
        logger.info(f"Saved cluster assignments: {cluster_file}")
        
        # Save cluster profiles
        profiles_file = self.output_dir / "cluster_profiles_full.csv"
        cluster_profiles.to_csv(profiles_file)
        logger.info(f"Saved cluster profiles: {profiles_file}")
        
        logger.info(f"\n{'='*60}")
        logger.info("Clustering Summary:")
        logger.info(f"Total tracks clustered: {len(self.features_df):,}")
        logger.info(f"Number of clusters: {self.n_clusters}")
        logger.info(f"Files saved: track_clusters_full.csv, cluster_profiles_full.csv")
        logger.info(f"{'='*60}\n")

def main():
    """Main execution."""
    
    OUTPUT_DIR = "data/processed"
    N_CLUSTERS = 5
    
    clusterer = TrackClusterer(
        output_dir=OUTPUT_DIR,
        n_clusters=N_CLUSTERS
    )
    
    # Load features
    clusterer.load_features()
    
    # Prepare features
    X_scaled, feature_cols = clusterer.prepare_features()
    
    # Run clustering
    clusterer.cluster(X_scaled)
    
    # Analyze clusters
    cluster_profiles = clusterer.analyze_clusters(feature_cols)
    
    # Save results
    clusterer.save_results(cluster_profiles)
    
    logger.info("✅ Clustering complete!")

if __name__ == "__main__":
    main()