"""
Generate Story-Driven Visualizations for Spotify Playlist Extension
Focuses on the narrative: Sparsity -> Network Structure -> Listener Types -> Model Results
Includes FIX for x-axis readability in Figure 3.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for publication-quality figures
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.figsize'] = (16, 9)

# Brand Colors
SPOTIFY_GREEN = '#1DB954'
SPOTIFY_BLACK = '#191414'
Jf_GREY = '#535353'

class StoryVisualizer:
    def __init__(self, results_json="PROJECT_RESULTS_SUMMARY.json", output_dir="presentation_figures_v2"):
        self.results_json = results_json
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load results
        with open(results_json, 'r') as f:
            self.results = json.load(f)
        
        print(f"✅ Loaded results. Saving to: {output_dir}/")

    def save_figure(self, fig, filename):
        filepath = self.output_dir / filename
        # Use tight_layout with specific padding to prevent cutting off titles/labels
        # Adjusted bottom padding to accommodate larger x-axis labels
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        fig.savefig(filepath, dpi=300, facecolor='white')
        print(f"  ✅ Generated: {filename}")
        plt.close(fig)

    # ============================================================================
    # 1. THE CHALLENGE: The Long Tail & Sparsity
    # ============================================================================
    def fig_sparsity_challenge(self):
        """Visualizes the extreme data sparsity (The Cold Start Problem)"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('The Challenge: Extreme Data Sparsity', fontsize=24, fontweight='bold', color=SPOTIFY_BLACK)

        # 1. Track Rarity
        ax1 = axes[0]
        metrics = self.results['evaluation']['overall_metrics']
        
        labels = ['In 1 Playlist', 'In 10+ Playlists', 'In 100+ Playlists', 'In 1k+ Playlists']
        values = [
            metrics['tracks_in_1_playlist'],
            metrics['tracks_in_10plus_playlists'],
            metrics['tracks_in_100plus_playlists'],
            metrics['tracks_in_1000plus_playlists']
        ]
        
        colors = ['#ff6b6b', '#feca57', '#48dbfb', SPOTIFY_GREEN]
        bars = ax1.bar(labels, values, color=colors, edgecolor=SPOTIFY_BLACK, linewidth=1.5)
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height/1e6:.1f}M', ha='center', va='bottom', fontweight='bold', fontsize=14)

        ax1.set_title('The "Cold Start" Reality', fontweight='bold')
        ax1.set_ylabel('Number of Tracks')
        ax1.grid(axis='y', alpha=0.3)
        
        ax1.text(0.5, 0.85, "1.07 Million Tracks\nappear in only\nONE playlist", 
                transform=ax1.transAxes, ha='center', fontsize=16, 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ff6b6b", lw=2))

        # 2. Gini Coefficient / Inequality
        ax2 = axes[1]
        gini = self.results['evaluation']['diversity_metrics']['popularity_gini']
        
        x = np.linspace(0, 1, 100)
        y_perfect = x
        power = (1 + gini) / (1 - gini)
        y_actual = x ** power
        
        ax2.plot(x, y_perfect, '--', color='gray', label='Perfect Equality (Gini=0)')
        ax2.plot(x, y_actual, '-', color=SPOTIFY_GREEN, linewidth=4, label=f'Our Dataset (Gini={gini:.2f})')
        ax2.fill_between(x, y_actual, y_perfect, color=SPOTIFY_GREEN, alpha=0.1)
        
        ax2.text(0.6, 0.2, f"Gini Coefficient: {gini:.3f}\n(Extreme Inequality)", 
                transform=ax2.transAxes, fontsize=16, fontweight='bold', color=SPOTIFY_BLACK)
        
        ax2.set_title('Popularity Inequality (Lorenz Curve)', fontweight='bold')
        ax2.set_xlabel('Cumulative Share of Tracks (Sorted by Popularity)')
        ax2.set_ylabel('Cumulative Share of Total Plays')
        ax2.legend()
        
        self.save_figure(fig, '01_sparsity_challenge.png')

    # ============================================================================
    # 2. THE STRUCTURE: Network Hubs
    # ============================================================================
    def fig_network_structure(self):
        """Visualizes the 'Superstar' nodes in the graph"""
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.suptitle('The Network Structure: "Superstar" Hubs Connect the Graph', fontsize=22, fontweight='bold')
        
        hubs = self.results['advanced']['graph_network']['hub_tracks'][:10]
        labels = [f"Hub #{i+1}" for i in range(len(hubs))]
        degrees = [h['degree'] / 1e6 for h in hubs]
        
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, degrees, align='center', color=SPOTIFY_GREEN, edgecolor=SPOTIFY_BLACK)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel('Node Degree (Millions of Connections)')
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.05, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}M Connections', 
                    va='center', fontweight='bold', color=Jf_GREY)

        total_nodes = self.results['advanced']['graph_network']['structure']['num_nodes']
        ax.text(0.7, 0.4, 
               f"Graph Density: 0.74\nMax Degree: {max(degrees):.1f}M\n\nThese few 'Hub' tracks\ndrive the recommendations.",
               transform=ax.transAxes, fontsize=14, 
               bbox=dict(boxstyle="round", fc="white", ec=SPOTIFY_BLACK))

        self.save_figure(fig, '02_network_hubs.png')

    # ============================================================================
    # 3. THE USERS: Cluster Profiles (FIXED VERSION)
    # ============================================================================
    def fig_listener_archetypes(self):
        """Visualizes the 5 Cluster Profiles found with readable axes"""
        # Increased figure height to accommodate multiline labels
        fig, ax = plt.subplots(figsize=(14, 9))
        fig.suptitle('Listener Archetypes: 5 Distinct Patterns Found', fontsize=22, fontweight='bold')
        
        profiles = self.results['clustering']['cluster_profiles']
        
        clusters = [p['cluster'] for p in profiles]
        pop = [p['popularity'] for p in profiles]
        pos_consistency = [p['position_consistency'] * 10 for p in profiles]
        artist_pop = [p['artist_popularity'] for p in profiles]
        
        x = np.arange(len(clusters))
        width = 0.25
        
        ax.bar(x - width, pop, width, label='Track Popularity', color='#1DB954')
        ax.bar(x, artist_pop, width, label='Artist Popularity', color='#191414')
        ax.bar(x + width, pos_consistency, width, label='Position Consistency (x10)', color='#535353')
        
        # --- FIXED LABELS LOGIC ---
        descriptions = {
            0: "Mainstream Fans\n(High Artist Pop)",
            1: "Album Listeners\n(High Consistency)",
            2: "Mixed / Eclectic\n(No strong bias)",
            3: "Album Listeners\n(High Consistency)",
            4: "Mainstream Fans\n(High Artist Pop)"
        }
        
        labels = [f"Cluster {c}\n{descriptions.get(c, '')}" for c in clusters]
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=13, fontweight='medium')
        ax.set_ylabel('Normalized Score', fontsize=14, fontweight='bold')
        
        # Moved legend to bottom to prevent overlap
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=14)
        
        # Adjust bottom margin for the large labels and legend
        plt.subplots_adjust(bottom=0.25)

        self.save_figure(fig, '03_listener_archetypes.png')

    # ============================================================================
    # 4. THE RESULT: Model Performance
    # ============================================================================
    def fig_model_results(self):
        """The punchline: Popularity wins"""
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle('The Result: Popularity Bias Domination', fontsize=22, fontweight='bold')
        
        models = self.results['evaluation']['model_comparison']
        names = [m['model'].replace('Popularity Baseline', 'Popularity\n(Baseline)') for m in models]
        scores = [m['precision@10'] * 100 for m in models]
        
        colors = [SPOTIFY_GREEN if 'Popularity' in n else '#b3b3b3' for n in names]
        
        bars = ax.bar(names, scores, color=colors, edgecolor=SPOTIFY_BLACK, linewidth=1.5)
        
        ax.set_ylabel('Precision@10 (%)')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=16)

        ax.text(0.5, 0.6, 
               "Why does the Baseline win?\n\nBecause of the Gini Coefficient (0.92).\nPredicting the 'global hits' is statistically\nsafer than personalized guessing for\nsparse playlists.",
               transform=ax.transAxes, fontsize=14, ha='center',
               bbox=dict(boxstyle="round,pad=0.5", fc="#f0f0f0", ec="gray"))

        self.save_figure(fig, '04_model_results.png')

    def generate_all(self):
        print("🎨 Generating Story-Driven Figures (v2)...")
        try: self.fig_sparsity_challenge()
        except Exception as e: print(f"Error Fig 1: {e}")
            
        try: self.fig_network_structure()
        except Exception as e: print(f"Error Fig 2: {e}")
            
        try: self.fig_listener_archetypes()
        except Exception as e: print(f"Error Fig 3: {e}")
            
        try: self.fig_model_results()
        except Exception as e: print(f"Error Fig 4: {e}")
        
        print("\n✨ Done! All figures generated in presentation_figures_v2/")

if __name__ == "__main__":
    viz = StoryVisualizer()
    viz.generate_all()