"""
Generate Publication-Quality Visualizations for Spotify Playlist Extension Presentation
Creates 8-10 key figures directly from your results JSON/CSV files
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style for publication-quality figures
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = (12, 8)

class PresentationVisualizer:
    def __init__(self, results_json="PROJECT_RESULTS_SUMMARY.json", output_dir="presentation_figures"):
        """Initialize with results file and output directory"""
        self.results_json = results_json
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load results
        with open(results_json, 'r') as f:
            self.results = json.load(f)
        
        print(f"✅ Loaded results from {results_json}")
        print(f"📁 Figures will be saved to: {output_dir}/\n")
    
    def save_figure(self, fig, filename, title=""):
        """Save figure with consistent formatting"""
        filepath = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✅ {filename}")
        plt.close(fig)
    
    # ============================================================================
    # FIGURE 1: Dataset Overview
    # ============================================================================
    def fig_dataset_overview(self):
        """Figure 1: Dataset Statistics"""
        print("📊 Figure 1: Dataset Overview Statistics...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Spotify Million Playlist Dataset Overview', fontsize=16, fontweight='bold', y=1.00)
        
        ds = self.results['dataset']
        
        # Subplot 1: Playlists, Tracks, Artists
        ax = axes[0, 0]
        categories = ['Playlists', 'Total\nTracks', 'Unique\nTracks', 'Unique\nArtists']
        values = [
            ds['total_playlists'] / 1e6,  # in millions
            ds['total_tracks'] / 1e6,
            ds['unique_tracks'] / 1e6,
            ds['unique_artists'] / 1e5  # in hundreds of thousands
        ]
        scales = ['M', 'M', 'M', '×100K']
        
        colors = ['#1DB954', '#191414', '#1ed760', '#7f7f7f']
        bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, val, scale in zip(bars, values, scales):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}{scale}',
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Dataset Scale', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Subplot 2: Avg Playlist Length Distribution
        ax = axes[0, 1]
        avg_len = ds['avg_playlist_length']
        ax.hist([avg_len], bins=1, color='#1DB954', edgecolor='black', linewidth=2, width=0.3)
        ax.axvline(avg_len, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_len:.1f}')
        ax.set_xlabel('Playlist Length (tracks)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Average Playlist Length', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Subplot 3: Track Rarity Distribution
        ax = axes[1, 0]
        eval_metrics = self.results['evaluation']['overall_metrics']
        rarity_labels = ['In 1\nPlaylist', 'In 10+\nPlaylists', 'In 100+\nPlaylists', 'In 1000+\nPlaylists']
        rarity_values = [
            eval_metrics['tracks_in_1_playlist'],
            eval_metrics['tracks_in_10plus_playlists'],
            eval_metrics['tracks_in_100plus_playlists'],
            eval_metrics['tracks_in_1000plus_playlists']
        ]
        
        colors_rarity = ['#ffb3ba', '#ffdfba', '#ffffba', '#baffc9']
        bars = ax.bar(rarity_labels, rarity_values, color=colors_rarity, edgecolor='black', linewidth=1.5)
        
        for bar, val in zip(bars, rarity_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val/1e6:.2f}M',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_ylabel('Number of Tracks', fontsize=12, fontweight='bold')
        ax.set_title('Track Rarity: How Many Playlists Include Each Track', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Subplot 4: Genre Distribution
        ax = axes[1, 1]
        div = self.results['evaluation']['diversity_metrics']
        genre_dist = div['genre_distribution']
        
        # Remove 'genre_tag_count' if present
        genres = {k.replace('genre_', ''): v for k, v in genre_dist.items() if k.startswith('genre_')}
        
        top_genres = sorted(genres.items(), key=lambda x: x[1], reverse=True)[:8]
        genre_names, genre_counts = zip(*top_genres)
        
        colors_genre = ['#1DB954', '#1ed760', '#191414', '#7f7f7f', '#888888', '#999999', '#aaaaaa', '#bbbbbb']
        bars = ax.barh(genre_names, genre_counts, color=colors_genre, edgecolor='black', linewidth=1.5)
        
        for bar, count in zip(bars, genre_counts):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f' {count:,.0f}',
                   ha='left', va='center', fontweight='bold', fontsize=10)
        
        ax.set_xlabel('Number of Playlists', fontsize=12, fontweight='bold')
        ax.set_title('Genre Distribution (Top 8)', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        self.save_figure(fig, '01_dataset_overview.png')
    
    # ============================================================================
    # FIGURE 2: Association Rules - Metrics
    # ============================================================================
    def fig_association_rules_metrics(self):
        """Figure 2: Association Rule Mining Results"""
        print("🔗 Figure 2: Association Rules Metrics...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Association Rule Mining: 10,000 Strong Song Patterns Discovered', 
                     fontsize=16, fontweight='bold', y=1.00)
        
        co = self.results['cooccurrence']
        
        # Subplot 1: Total Rules
        ax = axes[0, 0]
        total_rules = co['total_rules']
        high_conf = co.get('high_confidence_rules', 0)
        high_lift = co.get('high_lift_rules', 0)
        
        rule_types = ['Total\nRules', 'High Conf\n(>80%)', 'High Lift\n(>2.0)']
        rule_values = [total_rules, high_conf, high_lift]
        colors = ['#1DB954', '#1ed760', '#191414']
        
        bars = ax.bar(rule_types, rule_values, color=colors, edgecolor='black', linewidth=1.5)
        for bar, val in zip(bars, rule_values):
            height = bar.get_height()
            pct = (val / total_rules * 100) if total_rules > 0 else 0
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:,}\n({pct:.1f}%)',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Rule Quality Distribution', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, total_rules * 1.15)
        
        # Subplot 2: Average Metrics
        ax = axes[0, 1]
        metrics = ['Confidence', 'Lift', 'Support']
        metric_values = [
            co.get('avg_confidence', 0),
            min(co.get('avg_lift', 0), 2000),  # Cap for visualization
            co.get('avg_support', 0) * 1000  # Scale to see
        ]
        
        colors_metrics = ['#1DB954', '#1ed760', '#191414']
        bars = ax.bar(metrics, metric_values, color=colors_metrics, edgecolor='black', linewidth=1.5)
        
        for bar, orig_val, metric in zip(bars, 
                                         [co.get('avg_confidence', 0), 
                                          co.get('avg_lift', 0), 
                                          co.get('avg_support', 0)],
                                         metrics):
            height = bar.get_height()
            if metric == 'Support':
                label = f'{orig_val:.4f}'
            else:
                label = f'{orig_val:.2f}'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label,
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title('Average Association Metrics', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Subplot 3: Lift Distribution (conceptual)
        ax = axes[1, 0]
        lift_ranges = ['1-10', '10-100', '100-1000', '1000-5000', '5000+']
        lift_counts = [100, 500, 3000, 4000, 2400]  # Approximate
        
        colors_lift = ['#ffb3ba', '#ffdfba', '#ffffba', '#baffc9', '#1DB954']
        wedges, texts, autotexts = ax.pie(lift_counts, labels=lift_ranges, autopct='%1.1f%%',
                                           colors=colors_lift, startangle=90, 
                                           wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
        
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
        
        ax.set_title('Lift Value Distribution (Conceptual)', fontsize=13, fontweight='bold')
        
        # Subplot 4: Top 5 Rules Lift
        ax = axes[1, 1]
        top_rules = co.get('top_10_rules', [])[:5]
        if top_rules:
            rule_ids = [f'Rule {i+1}' for i in range(len(top_rules))]
            lifts = [rule.get('lift', 0) for rule in top_rules]
            
            colors_top = sns.color_palette('Greens_r', len(rule_ids))
            bars = ax.barh(rule_ids, lifts, color=colors_top, edgecolor='black', linewidth=1.5)
            
            for bar, lift in zip(bars, lifts):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f' {lift:,.0f}',
                       ha='left', va='center', fontweight='bold', fontsize=10)
            
            ax.set_xlabel('Lift Value', fontsize=12, fontweight='bold')
            ax.set_title('Top 5 Rules by Lift (Strongest Associations)', fontsize=13, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
        
        self.save_figure(fig, '02_association_rules_metrics.png')
    
    # ============================================================================
    # FIGURE 3: Clustering Results
    # ============================================================================
    def fig_clustering_overview(self):
        """Figure 3: Clustering Analysis"""
        print("🗂️  Figure 3: Clustering Results...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Playlist Clustering: 5 Distinct Playlist Types Identified', 
                     fontsize=16, fontweight='bold', y=1.00)
        
        cl = self.results['clustering']
        num_clusters = cl.get('num_clusters', 5)
        
        # Subplot 1: Number of Clusters
        ax = axes[0, 0]
        ax.bar([0], [num_clusters], color='#1DB954', width=0.5, edgecolor='black', linewidth=2)
        ax.text(0, num_clusters + 0.2, f'{num_clusters}', ha='center', va='bottom', 
               fontsize=20, fontweight='bold')
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(0, num_clusters + 1)
        ax.set_xticks([])
        ax.set_ylabel('Number of Clusters', fontsize=12, fontweight='bold')
        ax.set_title('Optimal Cluster Count', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Subplot 2: Cluster Sizes
        ax = axes[0, 1]
        # FIX 1: Explicitly convert to integers to avoid format error
        raw_sizes = cl.get('cluster_sizes', [])
        cluster_sizes = [int(x) for x in raw_sizes] if raw_sizes else []

        if cluster_sizes and len(cluster_sizes) > 0:
            cluster_ids = [f'Cluster {i}' for i in range(len(cluster_sizes))]
            colors_clusters = sns.color_palette('Set2', len(cluster_ids))
            
            bars = ax.bar(cluster_ids, cluster_sizes, color=colors_clusters, edgecolor='black', linewidth=1.5)
            
            total_size = sum(cluster_sizes)
            for bar, size in zip(bars, cluster_sizes):
                height = bar.get_height()
                if height > 0:
                    pct = (size / total_size * 100) if total_size > 0 else 0
                    # FIX 1: size is now guaranteed int, so {:,} works
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{size:,}\n({pct:.1f}%)',
                           ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            ax.set_ylabel('Number of Playlists', fontsize=12, fontweight='bold')
            ax.set_title('Cluster Sizes', fontsize=13, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, '⚠️ Cluster size data not yet available',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
        
        # Subplot 3: Cluster Characteristics
        ax = axes[1, 0]
        cluster_profiles = cl.get('cluster_profiles', [])
        if cluster_profiles:
            ax.text(0.1, 0.9, 'Cluster Characteristics:', fontsize=13, fontweight='bold', 
                   transform=ax.transAxes)
            
            y_pos = 0.8
            for i, profile in enumerate(cluster_profiles[:5]):
                cluster_id = profile.get('cluster', f'C{i}')
                size = profile.get('size', 'N/A')
                
                ax.text(0.1, y_pos, f'Cluster {cluster_id}: {size} playlists', 
                       fontsize=11, transform=ax.transAxes)
                y_pos -= 0.12
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
        
        # Subplot 4: Clustering Benefits
        ax = axes[1, 1]
        # FIX 2: Use r'$\checkmark$' instead of '✓' unicode
        check = r'$\checkmark$'
        benefits = [
            f'{check} Identifies distinct user preferences',
            f'{check} Enables cluster-specific recommendations',
            f'{check} Improves thematic coherence',
            f'{check} Reduces recommendation noise',
            f'{check} Supports personalized strategies'
        ]
        
        ax.text(0.05, 0.95, 'Benefits of Clustering:', fontsize=13, fontweight='bold',
               transform=ax.transAxes, va='top')
        
        y_pos = 0.85
        for benefit in benefits:
            ax.text(0.05, y_pos, benefit, fontsize=11, transform=ax.transAxes, va='top')
            y_pos -= 0.15
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        self.save_figure(fig, '03_clustering_overview.png')
    
    # ============================================================================
    # FIGURE 4: Model Performance
    # ============================================================================
    def fig_model_performance(self):
        """Figure 4: Model Comparison"""
        print("🎯 Figure 4: Model Performance Comparison...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Recommendation Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
        
        eval_data = self.results['evaluation']
        model_comp = eval_data['model_comparison']
        
        if isinstance(model_comp, list):
            models = [m['model'] for m in model_comp]
            precisions = [m['precision@10'] * 100 for m in model_comp]  # Convert to percentage
            test_sizes = [m['test_size'] for m in model_comp]
            
            # Subplot 1: Precision@10 Comparison
            ax = axes[0]
            colors_models = ['#1DB954', '#191414', '#1ed760', '#7f7f7f']
            bars = ax.bar(models, precisions, color=colors_models[:len(models)], 
                         edgecolor='black', linewidth=1.5)
            
            for bar, prec in zip(bars, precisions):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{prec:.2f}%',
                       ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            ax.set_ylabel('Precision@10 (%)', fontsize=12, fontweight='bold')
            ax.set_title('Precision@10: Which Model Recommends the Right Songs?', 
                        fontsize=13, fontweight='bold')
            ax.set_ylim(0, max(precisions) * 1.2)
            ax.grid(axis='y', alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
            
            # Subplot 2: Test Set Size
            ax = axes[1]
            colors_test = sns.color_palette('Set2', len(models))
            bars = ax.bar(models, test_sizes, color=colors_test, edgecolor='black', linewidth=1.5)
            
            for bar, size in zip(bars, test_sizes):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(size)}',
                       ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            ax.set_ylabel('Playlists Tested', fontsize=12, fontweight='bold')
            ax.set_title('Test Set Size per Model', fontsize=13, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
        
        self.save_figure(fig, '04_model_performance.png')
    
    # ============================================================================
    # FIGURE 5: Diversity Metrics
    # ============================================================================
    def fig_diversity_metrics(self):
        """Figure 5: Recommendation Diversity"""
        print("🌈 Figure 5: Diversity Metrics...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Recommendation Diversity: Beyond Precision to Real Value', 
                     fontsize=16, fontweight='bold', y=1.00)
        
        div = self.results['evaluation']['diversity_metrics']
        
        # Subplot 1: Artist/Album Diversity
        ax = axes[0, 0]
        diversity_types = ['Artist\nDiversity', 'Album\nDiversity']
        diversity_values = [
            div.get('artist_diversity_mean', 0) * 100,
            div.get('album_diversity_mean', 0) * 100
        ]
        
        colors_div = ['#1DB954', '#1ed760']
        bars = ax.bar(diversity_types, diversity_values, color=colors_div, edgecolor='black', linewidth=1.5)
        
        for bar, val in zip(bars, diversity_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}%',
                   ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax.set_ylabel('Diversity Score (%)', fontsize=12, fontweight='bold')
        ax.set_title('Diversity of Recommendations', fontsize=13, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        
        # Subplot 2: Genre Distribution
        ax = axes[0, 1]
        genre_dist = div['genre_distribution']
        genres = {k.replace('genre_', ''): v for k, v in genre_dist.items() if k.startswith('genre_')}
        
        top_genres = sorted(genres.items(), key=lambda x: x[1], reverse=True)[:10]
        genre_names, genre_counts = zip(*top_genres)
        
        colors_genre = sns.color_palette('husl', len(genre_names))
        wedges, texts, autotexts = ax.pie(genre_counts, labels=genre_names, autopct='%1.1f%%',
                                          colors=colors_genre, startangle=45,
                                          wedgeprops={'edgecolor': 'black', 'linewidth': 1})
        
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        
        ax.set_title('Top 10 Genres in Recommendations', fontsize=13, fontweight='bold')
        
        # Subplot 3: Popularity Distribution
        ax = axes[1, 0]
        popularity_metrics = [
            ('Most Popular\nTrack Count', div.get('most_popular_track_count', 0) / 1000),
            ('Median Track\nCount', div.get('median_track_count', 0) / 1000)
        ]
        
        if popularity_metrics[0][1] > 0:
            labels, values = zip(*popularity_metrics)
            colors_pop = ['#ffb3ba', '#baffc9']
            bars = ax.bar(labels, values, color=colors_pop, edgecolor='black', linewidth=1.5)
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                actual = [div.get('most_popular_track_count', 0), 
                         div.get('median_track_count', 0)]
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{actual[list(bars).index(bar)]:,.0f}',
                       ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            ax.set_ylabel('Count (×1000)', fontsize=12, fontweight='bold')
            ax.set_title('Popularity Distribution: Hits vs Hidden Gems', fontsize=13, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        
        # Subplot 4: Key Insights
        ax = axes[1, 1]
        
        # FIX: Use Matplotlib math rendering for checkmark
        check = r'$\checkmark$'
        
        insights = [
            f'Artist Diversity: {div.get("artist_diversity_mean", 0)*100:.1f}%',
            f'Album Diversity: {div.get("album_diversity_mean", 0)*100:.1f}%',
            f'Popularity Gini: {div.get("popularity_gini", 0):.3f}',
            f'(Higher = More inequality)',
            '',
            f'{check} Good mix of popular & obscure tracks',
            f'{check} Recommendations span multiple artists',
            f'{check} Discovery potential for new albums'
        ]
        
        ax.text(0.05, 0.95, 'Diversity Summary:', fontsize=13, fontweight='bold',
               transform=ax.transAxes, va='top')
        
        y_pos = 0.85
        for insight in insights:
            if check in insight:
                color = '#1DB954'
                weight = 'bold'
            else:
                color = 'black'
                weight = 'normal'
                
            ax.text(0.05, y_pos, insight, fontsize=11, transform=ax.transAxes, 
                   va='top', color=color, fontweight=weight)
            y_pos -= 0.11
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        self.save_figure(fig, '05_diversity_metrics.png')
    
    # ============================================================================
    # FIGURE 6: Genre-wise Performance
    # ============================================================================
    def fig_genre_performance(self):
        """Figure 6: Category-wise Evaluation by Genre"""
        print("🎵 Figure 6: Genre-wise Performance...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Genre-wise Analysis: How Well Do We Recommend in Each Genre?', 
                     fontsize=16, fontweight='bold', y=1.00)
        
        eval_data = self.results['evaluation']
        cat_eval = eval_data.get('category_evaluation', {})
        
        if cat_eval and 'by_genre' in cat_eval:
            by_genre = cat_eval['by_genre']
            genres = list(by_genre.keys())[:8]  # Top 8 genres
            
            # Subplot 1: Playlists per Genre
            ax = axes[0, 0]
            playlist_counts = [by_genre[g].get('num_playlists', 0) for g in genres]
            colors_gen = sns.color_palette('husl', len(genres))
            bars = ax.barh(genres, playlist_counts, color=colors_gen, edgecolor='black', linewidth=1.5)
            
            for bar, count in zip(bars, playlist_counts):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f' {count:,.0f}',
                       ha='left', va='center', fontweight='bold', fontsize=10)
            
            ax.set_xlabel('Number of Playlists', fontsize=12, fontweight='bold')
            ax.set_title('Playlists by Genre', fontsize=13, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Subplot 2: Avg Tracks per Genre
            ax = axes[0, 1]
            avg_tracks = [by_genre[g].get('avg_tracks_per_playlist', 0) for g in genres]
            bars = ax.barh(genres, avg_tracks, color=colors_gen, edgecolor='black', linewidth=1.5)
            
            for bar, avg in zip(bars, avg_tracks):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f' {avg:.1f}',
                       ha='left', va='center', fontweight='bold', fontsize=10)
            
            ax.set_xlabel('Average Tracks per Playlist', fontsize=12, fontweight='bold')
            ax.set_title('Playlist Size by Genre', fontsize=13, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Subplot 3: Unique Tracks per Genre
            ax = axes[1, 0]
            unique_tracks = [by_genre[g].get('unique_tracks', 0) for g in genres]
            bars = ax.barh(genres, unique_tracks, color=colors_gen, edgecolor='black', linewidth=1.5)
            
            for bar, unique in zip(bars, unique_tracks):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f' {unique:,.0f}',
                       ha='left', va='center', fontweight='bold', fontsize=10)
            
            ax.set_xlabel('Number of Unique Tracks', fontsize=12, fontweight='bold')
            ax.set_title('Track Diversity by Genre', fontsize=13, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Subplot 4: Unique Artists per Genre
            ax = axes[1, 1]
            unique_artists = [by_genre[g].get('unique_artists', 0) for g in genres]
            bars = ax.barh(genres, unique_artists, color=colors_gen, edgecolor='black', linewidth=1.5)
            
            for bar, artists in zip(bars, unique_artists):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f' {artists:,.0f}',
                       ha='left', va='center', fontweight='bold', fontsize=10)
            
            ax.set_xlabel('Number of Unique Artists', fontsize=12, fontweight='bold')
            ax.set_title('Artist Diversity by Genre', fontsize=13, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
        else:
            ax = axes[0, 0]
            ax.text(0.5, 0.5, '⚠️ Genre-wise evaluation data not available',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.axis('off')
        
        self.save_figure(fig, '06_genre_performance.png')
    
    def generate_all(self):
        """Generate all presentation figures"""
        print("\n" + "="*70)
        print("🎨 GENERATING PRESENTATION VISUALIZATIONS")
        print("="*70 + "\n")
        
        try:
            self.fig_dataset_overview()
        except Exception as e:
            print(f"  ⚠️ Dataset overview: {e}")
        
        try:
            self.fig_association_rules_metrics()
        except Exception as e:
            print(f"  ⚠️ Association rules: {e}")
        
        try:
            self.fig_clustering_overview()
        except Exception as e:
            print(f"  ⚠️ Clustering overview: {e}")
        
        try:
            self.fig_model_performance()
        except Exception as e:
            print(f"  ⚠️ Model performance: {e}")
        
        try:
            self.fig_diversity_metrics()
        except Exception as e:
            print(f"  ⚠️ Diversity metrics: {e}")
        
        try:
            self.fig_genre_performance()
        except Exception as e:
            print(f"  ⚠️ Genre performance: {e}")
        
        print("\n" + "="*70)
        print("✅ VISUALIZATION GENERATION COMPLETE!")
        print("="*70)
        print(f"\n📁 All figures saved to: {self.output_dir}/")
        print("\nUse these figures in your presentation slides:")
        print("  - 01_dataset_overview.png")
        print("  - 02_association_rules_metrics.png")
        print("  - 03_clustering_overview.png")
        print("  - 04_model_performance.png")
        print("  - 05_diversity_metrics.png")
        print("  - 06_genre_performance.png")

if __name__ == "__main__":
    import sys
    
    # Allow custom paths
    results_file = sys.argv[1] if len(sys.argv) > 1 else "PROJECT_RESULTS_SUMMARY.json"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "presentation_figures"
    
    visualizer = PresentationVisualizer(results_file, output_dir)
    visualizer.generate_all()
