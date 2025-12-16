"""
Comprehensive Metrics Extraction for Spotify Playlist Extension Project
Extracts all results into a single markdown document for presentation and paper
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

class MetricsExtractor:
    def __init__(self, data_dir="data/processed"):
        self.data_dir = Path(data_dir)
        self.results = {}
        
    def load_json(self, filename):
        """Load JSON file safely"""
        filepath = self.data_dir / filename
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load {filename}: {e}")
            return None
    
    def extract_dataset_stats(self):
        """Extract dataset statistics"""
        print("📊 Extracting Dataset Statistics...")
        
        # Load from mpd_statistics.json
        mpd_stats = self.load_json("mpd_statistics.json")
        
        # Also get from parquet files
        try:
            tracks_df = pd.read_parquet(self.data_dir / "tracks_full_mpd.parquet")
            playlists_df = pd.read_parquet(self.data_dir / "playlists_full_mpd.parquet")
            
            self.results['dataset'] = {
                'total_playlists': len(playlists_df['pid'].unique()) if 'pid' in playlists_df.columns else mpd_stats.get('total_playlists', 'N/A'),
                'total_tracks': len(tracks_df),
                'unique_tracks': len(tracks_df['track_uri'].unique()) if 'track_uri' in tracks_df.columns else 'N/A',
                'unique_artists': len(tracks_df['artist_name'].unique()) if 'artist_name' in tracks_df.columns else 'N/A',
                'unique_albums': len(tracks_df['album_name'].unique()) if 'album_name' in tracks_df.columns else 'N/A',
                'avg_playlist_length': tracks_df.groupby('pid').size().mean() if 'pid' in tracks_df.columns else 'N/A',
                'mpd_stats_full': mpd_stats
            }
            print("  ✅ Dataset stats extracted")
        except Exception as e:
            print(f"  ⚠️ Could not load dataset: {e}")
            self.results['dataset'] = mpd_stats or {}
    
    def extract_cooccurrence_stats(self):
        """Extract co-occurrence analysis results"""
        print("🔗 Extracting Co-occurrence Statistics...")
        
        cooc_stats = self.load_json("cooccurrence_stats.json")
        
        # Load association rules
        try:
            rules_df = pd.read_csv(self.data_dir / "association_rules_full.csv")
            
            # Check what columns actually exist
            print(f"  📋 Association rules columns: {rules_df.columns.tolist()}")
            
            # Extract top rules (first 10 rows as-is)
            top_rules = rules_df.head(10).to_dict('records')
            
            self.results['cooccurrence'] = {
                'total_rules': len(rules_df),
                'columns': rules_df.columns.tolist(),
                'top_10_rules': top_rules,
                'cooc_stats_full': cooc_stats
            }
            
            # Try to extract metrics if columns exist
            if 'confidence' in rules_df.columns:
                self.results['cooccurrence']['high_confidence_rules'] = len(rules_df[rules_df['confidence'] > 0.8])
                self.results['cooccurrence']['avg_confidence'] = float(rules_df['confidence'].mean())
            
            if 'lift' in rules_df.columns:
                self.results['cooccurrence']['high_lift_rules'] = len(rules_df[rules_df['lift'] > 2.0])
                self.results['cooccurrence']['avg_lift'] = float(rules_df['lift'].mean())
            
            if 'support' in rules_df.columns:
                self.results['cooccurrence']['avg_support'] = float(rules_df['support'].mean())
            
            print(f"  ✅ Found {len(rules_df)} association rules")
        except Exception as e:
            print(f"  ⚠️ Could not load association rules: {e}")
            self.results['cooccurrence'] = cooc_stats or {}
    
    def extract_clustering_results(self):
        """Extract clustering results"""
        print("🗂️ Extracting Clustering Results...")
        
        try:
            cluster_profiles = pd.read_csv(self.data_dir / "cluster_profiles_full.csv")
            
            self.results['clustering'] = {
                'num_clusters': len(cluster_profiles),
                'cluster_sizes': cluster_profiles['size'].tolist() if 'size' in cluster_profiles.columns else [],
                'cluster_profiles': cluster_profiles.to_dict('records'),
                'largest_cluster': cluster_profiles.nlargest(1, 'size').to_dict('records')[0] if 'size' in cluster_profiles.columns else {},
                'smallest_cluster': cluster_profiles.nsmallest(1, 'size').to_dict('records')[0] if 'size' in cluster_profiles.columns else {}
            }
            print(f"  ✅ Found {len(cluster_profiles)} clusters")
        except Exception as e:
            print(f"  ⚠️ Could not load clustering results: {e}")
            self.results['clustering'] = {}
    
    def extract_evaluation_metrics(self):
        """Extract recommendation evaluation metrics"""
        print("🎯 Extracting Evaluation Metrics...")
        
        eval_metrics = self.load_json("evaluation_metrics_full.json")
        diversity_metrics = self.load_json("diversity_metrics_full.json")
        category_eval = self.load_json("category_evaluation_full.json")
        model_comparison = self.load_json("model_comparison_results.json")
        
        self.results['evaluation'] = {
            'overall_metrics': eval_metrics,
            'diversity_metrics': diversity_metrics,
            'category_evaluation': category_eval,
            'model_comparison': model_comparison
        }
        print("  ✅ Evaluation metrics extracted")
    
    def extract_advanced_analysis(self):
        """Extract advanced analysis results"""
        print("🔬 Extracting Advanced Analysis...")
        
        graph_analysis = self.load_json("graph_network_analysis.json")
        temporal_analysis = self.load_json("temporal_sequential_analysis.json")
        genre_analysis = self.load_json("genre_cross_pollination_analysis.json")
        explainability = self.load_json("recommendation_explainability.json")
        
        self.results['advanced'] = {
            'graph_network': graph_analysis,
            'temporal_sequential': temporal_analysis,
            'genre_cross_pollination': genre_analysis,
            'explainability': explainability
        }
        print("  ✅ Advanced analysis extracted")
    
    def generate_markdown_report(self):
        """Generate comprehensive markdown report"""
        print("\n📝 Generating Markdown Report...")
        
        md = []
        md.append("# Spotify Playlist Extension: Complete Results Summary")
        md.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md.append("\n---\n")
        
        # DATASET STATISTICS
        md.append("## 1. Dataset Statistics\n")
        if 'dataset' in self.results:
            ds = self.results['dataset']
            
            def format_num(val):
                """Format number or return N/A"""
                if isinstance(val, (int, float)):
                    return f"{int(val):,}"
                return str(val)
            
            md.append(f"- **Total Playlists:** {format_num(ds.get('total_playlists', 'N/A'))}")
            md.append(f"- **Total Track Entries:** {format_num(ds.get('total_tracks', 'N/A'))}")
            md.append(f"- **Unique Tracks:** {format_num(ds.get('unique_tracks', 'N/A'))}")
            md.append(f"- **Unique Artists:** {format_num(ds.get('unique_artists', 'N/A'))}")
            md.append(f"- **Unique Albums:** {format_num(ds.get('unique_albums', 'N/A'))}")
            if isinstance(ds.get('avg_playlist_length'), (int, float)):
                md.append(f"- **Average Playlist Length:** {ds['avg_playlist_length']:.1f} tracks")
        md.append("\n")
        
        # RESEARCH QUESTION 1: CO-OCCURRENCE
        md.append("## 2. Research Question 1: Song Co-occurrence Patterns\n")
        md.append("### 2.1 Association Rule Mining Results\n")
        if 'cooccurrence' in self.results:
            co = self.results['cooccurrence']
            
            total = co.get('total_rules')
            if isinstance(total, int):
                md.append(f"- **Total Association Rules Generated:** {total:,}")
            else:
                md.append(f"- **Total Association Rules Generated:** {total}")
            
            high_conf = co.get('high_confidence_rules')
            if isinstance(high_conf, int):
                md.append(f"- **High Confidence Rules (>80%):** {high_conf:,}")
            
            high_lift = co.get('high_lift_rules')
            if isinstance(high_lift, int):
                md.append(f"- **High Lift Rules (>2.0):** {high_lift:,}")
            
            if isinstance(co.get('avg_confidence'), (int, float)):
                md.append(f"- **Average Confidence:** {co['avg_confidence']:.3f}")
            if isinstance(co.get('avg_lift'), (int, float)):
                md.append(f"- **Average Lift:** {co['avg_lift']:.3f}")
            if isinstance(co.get('avg_support'), (int, float)):
                md.append(f"- **Average Support:** {co['avg_support']:.6f}")
            
            # Top rules - display whatever columns exist
            if co.get('top_10_rules'):
                md.append("\n### 2.2 Top 10 Association Rules\n")
                md.append("```")
                for i, rule in enumerate(co['top_10_rules'][:10], 1):
                    md.append(f"\nRule {i}:")
                    for key, val in rule.items():
                        md.append(f"  {key}: {val}")
                md.append("```")
        md.append("\n")
        
        # RESEARCH QUESTION 2: CLUSTERING
        md.append("## 3. Research Question 2: Clustering Results\n")
        if 'clustering' in self.results:
            cl = self.results['clustering']
            md.append(f"- **Number of Clusters:** {cl.get('num_clusters', 'N/A')}")
            
            if cl.get('cluster_profiles'):
                md.append("\n### 3.1 Cluster Profiles\n")
                md.append("| Cluster ID | Size | Description |")
                md.append("|------------|------|-------------|")
                for profile in cl['cluster_profiles']:
                    cluster_id = profile.get('cluster', 'N/A')
                    size = profile.get('size', 'N/A')
                    # Format size with comma if it's a number
                    size_str = f"{int(size):,}" if isinstance(size, (int, float)) else str(size)
                    # Add any other cluster characteristics if available
                    md.append(f"| {cluster_id} | {size_str} | - |")
        md.append("\n")
        
        # RESEARCH QUESTION 3: RECOMMENDATIONS
        md.append("## 4. Research Question 3: Recommendation Performance\n")
        if 'evaluation' in self.results:
            ev = self.results['evaluation']
            
            # Overall metrics
            if ev.get('overall_metrics'):
                md.append("### 4.1 Overall Performance Metrics\n")
                om = ev['overall_metrics']
                md.append("```json")
                md.append(json.dumps(om, indent=2))
                md.append("```\n")
            
            # Model comparison
            if ev.get('model_comparison'):
                md.append("### 4.2 Model Comparison\n")
                mc = ev['model_comparison']
                md.append("```json")
                md.append(json.dumps(mc, indent=2))
                md.append("```\n")
            
            # Diversity metrics
            if ev.get('diversity_metrics'):
                md.append("### 4.3 Diversity Metrics\n")
                dm = ev['diversity_metrics']
                md.append("```json")
                md.append(json.dumps(dm, indent=2))
                md.append("```\n")
            
            # Category evaluation
            if ev.get('category_evaluation'):
                md.append("### 4.4 Category-wise Evaluation\n")
                ce = ev['category_evaluation']
                md.append("```json")
                md.append(json.dumps(ce, indent=2))
                md.append("```\n")
        
        # ADVANCED ANALYSIS
        md.append("## 5. Advanced Analysis\n")
        if 'advanced' in self.results:
            adv = self.results['advanced']
            
            if adv.get('graph_network'):
                md.append("### 5.1 Graph Network Analysis\n")
                md.append("```json")
                md.append(json.dumps(adv['graph_network'], indent=2))
                md.append("```\n")
            
            if adv.get('temporal_sequential'):
                md.append("### 5.2 Temporal/Sequential Analysis\n")
                md.append("```json")
                md.append(json.dumps(adv['temporal_sequential'], indent=2))
                md.append("```\n")
            
            if adv.get('genre_cross_pollination'):
                md.append("### 5.3 Genre Cross-Pollination\n")
                md.append("```json")
                md.append(json.dumps(adv['genre_cross_pollination'], indent=2))
                md.append("```\n")
            
            if adv.get('explainability'):
                md.append("### 5.4 Recommendation Explainability\n")
                md.append("```json")
                md.append(json.dumps(adv['explainability'], indent=2))
                md.append("```\n")
        
        return "\n".join(md)
    
    def save_results(self, output_file="PROJECT_RESULTS_SUMMARY.md"):
        """Save all results to files"""
        print("\n💾 Saving Results...")
        
        # Save markdown report
        report = self.generate_markdown_report()
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"  ✅ Markdown report saved: {output_file}")
        
        # Save JSON for programmatic access
        json_file = output_file.replace('.md', '.json')
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"  ✅ JSON results saved: {json_file}")
        
        return output_file, json_file

def main():
    print("="*60)
    print("Spotify Playlist Extension - Metrics Extraction")
    print("="*60)
    print()
    
    extractor = MetricsExtractor()
    
    # Extract all metrics
    extractor.extract_dataset_stats()
    extractor.extract_cooccurrence_stats()
    extractor.extract_clustering_results()
    extractor.extract_evaluation_metrics()
    extractor.extract_advanced_analysis()
    
    # Save results
    md_file, json_file = extractor.save_results()
    
    print("\n" + "="*60)
    print("✅ EXTRACTION COMPLETE!")
    print("="*60)
    print(f"\n📄 Review your results:")
    print(f"   - Markdown: {md_file}")
    print(f"   - JSON: {json_file}")
    print("\nUse these files for your presentation and paper!")

if __name__ == "__main__":
    main()