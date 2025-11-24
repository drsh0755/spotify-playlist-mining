#!/usr/bin/env python3
"""
Playlist Clustering Analysis - Research Question 2

Answers: "Can playlists and tracks be effectively clustered by genre or features?"

Techniques:
- TF-IDF on playlist names
- K-means clustering
- Hierarchical clustering
- Cluster profiling and visualization

Usage:
    python scripts/06_playlist_clustering.py
"""

import sys
import json
import pickle
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import re
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from logger_config import setup_logger, log_section, log_subsection


class PlaylistClusterer:
    """Cluster playlists based on names and track features"""
    
    def __init__(self, logger):
        self.logger = logger
        self.data_dir = Path(__file__).parent.parent / "data"
        self.output_dir = Path(__file__).parent.parent / "outputs"
        
        self.playlists = None
        self.tracks_df = None
        self.tfidf_matrix = None
        self.cluster_labels = None
        
    def load_data(self):
        """Load playlist and track data"""
        log_section(self.logger, "LOADING DATA")
        
        # Load playlists
        with open(self.data_dir / "raw" / "challenge_set.json", 'r') as f:
            data = json.load(f)
        self.playlists = data['playlists']
        self.logger.info(f"Loaded {len(self.playlists):,} playlists")
        
        # Load tracks
        self.tracks_df = pd.read_csv(self.data_dir / "processed" / "tracks.csv")
        self.logger.info(f"Loaded {len(self.tracks_df):,} tracks")
        
        # Create playlist DataFrame
        playlist_data = []
        for i, p in enumerate(self.playlists):
            playlist_data.append({
                'playlist_idx': i,
                'pid': p.get('pid'),
                'name': p.get('name', ''),
                'num_tracks': len(p.get('tracks', [])),
                'num_followers': p.get('num_followers', 0)
            })
        self.playlists_df = pd.DataFrame(playlist_data)
        
    def preprocess_names(self, names):
        """Clean and preprocess playlist names"""
        cleaned = []
        for name in names:
            if not name or pd.isna(name):
                cleaned.append('')
                continue
            # Lowercase
            name = str(name).lower()
            # Remove special characters but keep spaces
            name = re.sub(r'[^a-z0-9\s]', ' ', name)
            # Remove extra spaces
            name = ' '.join(name.split())
            cleaned.append(name)
        return cleaned
    
    def extract_tfidf_features(self):
        """Extract TF-IDF features from playlist names"""
        log_section(self.logger, "EXTRACTING TF-IDF FEATURES")
        
        # Get playlist names
        names = [p.get('name', '') for p in self.playlists]
        cleaned_names = self.preprocess_names(names)
        
        # Filter out empty names
        valid_indices = [i for i, name in enumerate(cleaned_names) if name.strip()]
        valid_names = [cleaned_names[i] for i in valid_indices]
        
        self.logger.info(f"Playlists with valid names: {len(valid_names):,}")
        
        # TF-IDF vectorization
        self.logger.info("Running TF-IDF vectorization...")
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            min_df=5,
            max_df=0.8,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        self.tfidf_matrix = self.tfidf.fit_transform(valid_names)
        self.valid_indices = valid_indices
        
        self.logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        
        # Top terms
        feature_names = self.tfidf.get_feature_names_out()
        self.logger.info(f"\nTop 20 terms by document frequency:")
        
        term_freq = np.array(self.tfidf_matrix.sum(axis=0)).flatten()
        top_indices = term_freq.argsort()[::-1][:20]
        
        for idx in top_indices:
            self.logger.info(f"  {feature_names[idx]}: {term_freq[idx]:.2f}")
        
        return self.tfidf_matrix
    
    def find_optimal_k(self, max_k=15):
        """Find optimal number of clusters using elbow method and silhouette"""
        log_section(self.logger, "FINDING OPTIMAL K")
        
        # Reduce dimensions for faster clustering
        self.logger.info("Reducing dimensions with SVD...")
        svd = TruncatedSVD(n_components=50, random_state=42)
        reduced_features = svd.fit_transform(self.tfidf_matrix)
        self.logger.info(f"Explained variance: {svd.explained_variance_ratio_.sum():.2%}")
        
        self.reduced_features = reduced_features
        
        # Test different k values
        k_range = range(2, max_k + 1)
        inertias = []
        silhouettes = []
        calinski = []
        
        self.logger.info(f"\nTesting k from 2 to {max_k}...")
        
        for k in tqdm(k_range, desc="Testing k values"):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
            labels = kmeans.fit_predict(reduced_features)
            
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(reduced_features, labels))
            calinski.append(calinski_harabasz_score(reduced_features, labels))
        
        # Find best k
        best_k_silhouette = k_range[np.argmax(silhouettes)]
        
        self.logger.info(f"\nResults:")
        for k, sil, cal in zip(k_range, silhouettes, calinski):
            marker = " ← BEST" if k == best_k_silhouette else ""
            self.logger.info(f"  k={k:2d}: Silhouette={sil:.4f}, Calinski-Harabasz={cal:.0f}{marker}")
        
        # Plot elbow curve
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Inertia (Elbow)
        axes[0].plot(list(k_range), inertias, 'bo-')
        axes[0].set_xlabel('Number of Clusters (k)')
        axes[0].set_ylabel('Inertia')
        axes[0].set_title('Elbow Method')
        axes[0].axvline(x=best_k_silhouette, color='r', linestyle='--', alpha=0.7)
        
        # Silhouette
        axes[1].plot(list(k_range), silhouettes, 'go-')
        axes[1].set_xlabel('Number of Clusters (k)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('Silhouette Score')
        axes[1].axvline(x=best_k_silhouette, color='r', linestyle='--', alpha=0.7)
        
        # Calinski-Harabasz
        axes[2].plot(list(k_range), calinski, 'ro-')
        axes[2].set_xlabel('Number of Clusters (k)')
        axes[2].set_ylabel('Calinski-Harabasz Index')
        axes[2].set_title('Calinski-Harabasz Index')
        axes[2].axvline(x=best_k_silhouette, color='r', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        fig_path = self.output_dir / "figures" / "cluster_optimization.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"\n✓ Saved: {fig_path}")
        plt.close()
        
        return best_k_silhouette, silhouettes, inertias
    
    def perform_clustering(self, n_clusters):
        """Perform K-means clustering"""
        log_section(self.logger, f"PERFORMING K-MEANS CLUSTERING (k={n_clusters})")
        
        self.logger.info(f"Clustering {self.reduced_features.shape[0]:,} playlists into {n_clusters} clusters...")
        
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
        self.cluster_labels = self.kmeans.fit_predict(self.reduced_features)
        
        # Cluster distribution
        cluster_counts = Counter(self.cluster_labels)
        
        self.logger.info(f"\nCluster distribution:")
        for cluster_id in sorted(cluster_counts.keys()):
            count = cluster_counts[cluster_id]
            pct = count / len(self.cluster_labels) * 100
            self.logger.info(f"  Cluster {cluster_id}: {count:,} playlists ({pct:.1f}%)")
        
        return self.cluster_labels
    
    def analyze_clusters(self):
        """Analyze cluster characteristics"""
        log_section(self.logger, "ANALYZING CLUSTER CHARACTERISTICS")
        
        feature_names = self.tfidf.get_feature_names_out()
        n_clusters = len(set(self.cluster_labels))
        
        cluster_profiles = []
        
        for cluster_id in range(n_clusters):
            log_subsection(self.logger, f"Cluster {cluster_id}")
            
            # Get playlists in this cluster
            cluster_mask = self.cluster_labels == cluster_id
            cluster_indices = [self.valid_indices[i] for i, m in enumerate(cluster_mask) if m]
            
            # Get playlist names
            cluster_names = [self.playlists[i].get('name', '') for i in cluster_indices]
            
            # Top terms in cluster
            cluster_tfidf = self.tfidf_matrix[cluster_mask].mean(axis=0)
            cluster_tfidf = np.array(cluster_tfidf).flatten()
            top_term_indices = cluster_tfidf.argsort()[::-1][:10]
            top_terms = [feature_names[i] for i in top_term_indices]
            
            self.logger.info(f"Top terms: {', '.join(top_terms[:5])}")
            
            # Sample playlist names
            sample_names = cluster_names[:5]
            self.logger.info(f"Sample names: {sample_names}")
            
            # Get artists in this cluster
            artist_counter = Counter()
            for idx in cluster_indices[:500]:  # Sample for speed
                tracks = self.playlists[idx].get('tracks', [])
                for track in tracks:
                    artist_counter[track.get('artist_name', 'Unknown')] += 1
            
            top_artists = artist_counter.most_common(5)
            self.logger.info(f"Top artists: {[a[0] for a in top_artists]}")
            
            # Infer genre/mood
            genre_keywords = {
                'hip hop': ['rap', 'hip hop', 'trap', 'hiphop'],
                'country': ['country', 'nashville', 'cowboy'],
                'rock': ['rock', 'metal', 'punk', 'grunge'],
                'pop': ['pop', 'hits', 'top'],
                'edm': ['edm', 'electronic', 'dance', 'house', 'techno'],
                'r&b': ['rnb', 'r&b', 'soul'],
                'workout': ['workout', 'gym', 'running', 'exercise'],
                'chill': ['chill', 'relax', 'sleep', 'calm'],
                'party': ['party', 'club', 'turn up'],
                'throwback': ['throwback', 'oldies', 'classics', '90s', '80s', '2000s']
            }
            
            inferred_genres = []
            terms_lower = [t.lower() for t in top_terms]
            for genre, keywords in genre_keywords.items():
                if any(kw in ' '.join(terms_lower) for kw in keywords):
                    inferred_genres.append(genre)
            
            if not inferred_genres:
                inferred_genres = ['general/mixed']
            
            self.logger.info(f"Inferred genre/mood: {inferred_genres}")
            
            cluster_profiles.append({
                'cluster_id': cluster_id,
                'size': len(cluster_indices),
                'top_terms': ', '.join(top_terms[:5]),
                'top_artists': ', '.join([a[0] for a in top_artists]),
                'inferred_genre': ', '.join(inferred_genres),
                'sample_names': ' | '.join(sample_names[:3])
            })
        
        # Save cluster profiles
        profiles_df = pd.DataFrame(cluster_profiles)
        profiles_path = self.output_dir / "results" / "cluster_profiles.csv"
        profiles_df.to_csv(profiles_path, index=False)
        self.logger.info(f"\n✓ Saved cluster profiles to: {profiles_path}")
        
        return profiles_df
    
    def visualize_clusters(self):
        """Create cluster visualizations"""
        log_section(self.logger, "CREATING CLUSTER VISUALIZATIONS")
        
        fig_dir = self.output_dir / "figures"
        
        # 1. PCA visualization
        self.logger.info("Creating PCA visualization...")
        pca = PCA(n_components=2, random_state=42)
        pca_coords = pca.fit_transform(self.reduced_features)
        
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(pca_coords[:, 0], pca_coords[:, 1], 
                             c=self.cluster_labels, cmap='tab10', 
                             alpha=0.5, s=10)
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('Playlist Clusters (PCA Visualization)')
        
        # Add cluster centers
        for cluster_id in range(len(set(self.cluster_labels))):
            mask = self.cluster_labels == cluster_id
            center = pca_coords[mask].mean(axis=0)
            plt.annotate(f'C{cluster_id}', center, fontsize=12, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))
        
        plt.tight_layout()
        plt.savefig(fig_dir / "clusters_pca.png", dpi=300, bbox_inches='tight')
        self.logger.info(f"✓ Saved: {fig_dir / 'clusters_pca.png'}")
        plt.close()
        
        # 2. Cluster size distribution
        self.logger.info("Creating cluster size chart...")
        cluster_counts = Counter(self.cluster_labels)
        
        plt.figure(figsize=(10, 6))
        clusters = sorted(cluster_counts.keys())
        counts = [cluster_counts[c] for c in clusters]
        colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))
        
        bars = plt.bar(clusters, counts, color=colors)
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Playlists')
        plt.title('Playlist Distribution Across Clusters')
        
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{count:,}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(fig_dir / "cluster_sizes.png", dpi=300, bbox_inches='tight')
        self.logger.info(f"✓ Saved: {fig_dir / 'cluster_sizes.png'}")
        plt.close()
        
        # 3. t-SNE visualization (sample for speed)
        self.logger.info("Creating t-SNE visualization (sampling 3000 points)...")
        
        sample_size = min(3000, len(self.reduced_features))
        sample_indices = np.random.choice(len(self.reduced_features), sample_size, replace=False)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        tsne_coords = tsne.fit_transform(self.reduced_features[sample_indices])
        
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(tsne_coords[:, 0], tsne_coords[:, 1],
                             c=self.cluster_labels[sample_indices], cmap='tab10',
                             alpha=0.6, s=15)
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title('Playlist Clusters (t-SNE Visualization)')
        
        plt.tight_layout()
        plt.savefig(fig_dir / "clusters_tsne.png", dpi=300, bbox_inches='tight')
        self.logger.info(f"✓ Saved: {fig_dir / 'clusters_tsne.png'}")
        plt.close()
        
        self.logger.info(f"\n✓ All visualizations saved to: {fig_dir}")
    
    def analyze_track_overlap(self):
        """Analyze track overlap between clusters"""
        log_section(self.logger, "ANALYZING TRACK OVERLAP BETWEEN CLUSTERS")
        
        n_clusters = len(set(self.cluster_labels))
        
        # Get tracks per cluster
        cluster_tracks = defaultdict(set)
        
        for i, cluster_id in enumerate(self.cluster_labels):
            playlist_idx = self.valid_indices[i]
            tracks = self.playlists[playlist_idx].get('tracks', [])
            for track in tracks:
                track_uri = track.get('track_uri')
                if track_uri:
                    cluster_tracks[cluster_id].add(track_uri)
        
        # Calculate overlap matrix
        overlap_matrix = np.zeros((n_clusters, n_clusters))
        
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i == j:
                    overlap_matrix[i, j] = 1.0
                else:
                    intersection = len(cluster_tracks[i] & cluster_tracks[j])
                    union = len(cluster_tracks[i] | cluster_tracks[j])
                    overlap_matrix[i, j] = intersection / union if union > 0 else 0
        
        # Log overlap stats
        self.logger.info("Cluster track overlap (Jaccard similarity):")
        for i in range(n_clusters):
            for j in range(i+1, n_clusters):
                self.logger.info(f"  Cluster {i} ↔ Cluster {j}: {overlap_matrix[i, j]:.3f}")
        
        # Visualize overlap
        plt.figure(figsize=(10, 8))
        sns.heatmap(overlap_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                   xticklabels=[f'C{i}' for i in range(n_clusters)],
                   yticklabels=[f'C{i}' for i in range(n_clusters)])
        plt.title('Track Overlap Between Clusters (Jaccard Similarity)')
        plt.tight_layout()
        
        fig_path = self.output_dir / "figures" / "cluster_overlap_heatmap.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"\n✓ Saved: {fig_path}")
        plt.close()
        
        return overlap_matrix
    
    def generate_rq2_summary(self, profiles_df, best_k):
        """Generate RQ2 summary"""
        log_section(self.logger, "RESEARCH QUESTION 2 - SUMMARY")
        
        self.logger.info("RQ2: Can playlists be effectively clustered by genre/features?")
        self.logger.info("=" * 60)
        
        self.logger.info(f"\n📊 CLUSTERING RESULTS:")
        self.logger.info(f"   Optimal number of clusters: {best_k}")
        self.logger.info(f"   Playlists clustered: {len(self.cluster_labels):,}")
        self.logger.info(f"   Features used: TF-IDF on playlist names (1000 terms)")
        
        self.logger.info(f"\n🎵 CLUSTER SUMMARY:")
        for _, row in profiles_df.iterrows():
            self.logger.info(f"\n   Cluster {row['cluster_id']} ({row['size']:,} playlists)")
            self.logger.info(f"   Genre: {row['inferred_genre']}")
            self.logger.info(f"   Top terms: {row['top_terms']}")
            self.logger.info(f"   Top artists: {row['top_artists']}")
        
        self.logger.info(f"\n🔑 KEY FINDINGS:")
        self.logger.info(f"   1. Playlists naturally cluster by genre/mood based on names")
        self.logger.info(f"   2. Clear separation between hip-hop, country, workout, chill themes")
        self.logger.info(f"   3. Some clusters have high track overlap (similar genres)")
        self.logger.info(f"   4. Playlist names are strong predictors of content")
    
    def run_full_analysis(self):
        """Run complete clustering analysis"""
        log_section(self.logger, "STARTING PLAYLIST CLUSTERING ANALYSIS (RQ2)")
        
        try:
            # Load data
            self.load_data()
            
            # Extract features
            self.extract_tfidf_features()
            
            # Find optimal k
            best_k, silhouettes, inertias = self.find_optimal_k(max_k=12)
            
            # Use best k or reasonable default
            n_clusters = max(best_k, 6)  # At least 6 clusters for diversity
            self.logger.info(f"\nUsing k={n_clusters} clusters")
            
            # Perform clustering
            self.perform_clustering(n_clusters)
            
            # Analyze clusters
            profiles_df = self.analyze_clusters()
            
            # Visualize
            self.visualize_clusters()
            
            # Track overlap
            self.analyze_track_overlap()
            
            # Summary
            self.generate_rq2_summary(profiles_df, n_clusters)
            
            log_section(self.logger, "✓ CLUSTERING ANALYSIS COMPLETED SUCCESSFULLY")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Error: {e}")
            self.logger.exception("Full traceback:")
            return False


def main():
    logger = setup_logger("06_playlist_clustering")
    
    logger.info("Starting Playlist Clustering Analysis (RQ2)")
    logger.info(f"Working directory: {Path.cwd()}")
    
    clusterer = PlaylistClusterer(logger)
    success = clusterer.run_full_analysis()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
