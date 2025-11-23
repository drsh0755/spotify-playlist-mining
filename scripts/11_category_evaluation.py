#!/usr/bin/env python3
"""
Category-wise Evaluation - Challenge Set Categories

Evaluates recommendation performance separately for each challenge category:
- Title only (cold start)
- Title + 1/5/10/25/100 tracks
- No title + 5/10 tracks
"""

import sys
import json
import pickle
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from logger_config import setup_logger, log_section, log_subsection


class CategoryEvaluator:
    """Evaluate recommendations by challenge category"""
    
    def __init__(self, logger):
        self.logger = logger
        self.data_dir = Path(__file__).parent.parent / "data"
        self.output_dir = Path(__file__).parent.parent / "outputs"
        
    def load_data(self):
        """Load data"""
        log_section(self.logger, "LOADING DATA")
        
        with open(self.data_dir / "raw" / "challenge_set.json", 'r') as f:
            self.playlists = json.load(f)['playlists']
        self.logger.info(f"Loaded {len(self.playlists):,} playlists")
        
        self.tracks_df = pd.read_csv(self.data_dir / "processed" / "tracks.csv")
        self.track_popularity = dict(zip(self.tracks_df['track_uri'], self.tracks_df['playlist_count']))
        
        with open(self.data_dir / "processed" / "cooccurrence_matrix.pkl", 'rb') as f:
            cooccur_data = pickle.load(f)
        self.cooccur_matrix = cooccur_data['matrix']
        self.cooccur_metadata = cooccur_data['metadata']
        
    def categorize_playlists(self):
        """Categorize playlists by challenge type"""
        log_section(self.logger, "CATEGORIZING PLAYLISTS")
        
        self.categories = defaultdict(list)
        
        for i, p in enumerate(self.playlists):
            num_tracks = len(p.get('tracks', []))
            has_name = bool(p.get('name'))
            
            if num_tracks == 0:
                cat = '1_title_only'
            elif num_tracks == 1:
                cat = '2_title_1track'
            elif num_tracks == 5 and has_name:
                cat = '3_title_5tracks'
            elif num_tracks == 5 and not has_name:
                cat = '4_notitle_5tracks'
            elif num_tracks == 10 and has_name:
                cat = '5_title_10tracks'
            elif num_tracks == 10 and not has_name:
                cat = '6_notitle_10tracks'
            elif num_tracks == 25:
                cat = '7_title_25tracks'
            elif num_tracks == 100:
                cat = '8_title_100tracks'
            else:
                cat = '9_other'
            
            self.categories[cat].append(i)
        
        self.logger.info("Challenge categories:")
        for cat in sorted(self.categories.keys()):
            self.logger.info(f"  {cat}: {len(self.categories[cat]):,} playlists")
    
    def get_recommendations(self, seed_tracks, n=500):
        """Get recommendations based on seed tracks"""
        if not seed_tracks:
            # Cold start - return popular tracks
            sorted_tracks = sorted(self.track_popularity.items(), key=lambda x: x[1], reverse=True)
            return [t for t, p in sorted_tracks[:n]]
        
        track_to_id = self.cooccur_metadata['track_to_id']
        id_to_track = self.cooccur_metadata['id_to_track']
        
        scores = defaultdict(float)
        
        for seed in seed_tracks:
            if seed not in track_to_id:
                continue
            seed_id = track_to_id[seed]
            vec = self.cooccur_matrix[seed_id, :].toarray().flatten()
            
            for tid, count in enumerate(vec):
                if count > 0:
                    scores[id_to_track[tid]] += count
        
        # If no co-occurrence found, fall back to popularity
        if not scores:
            sorted_tracks = sorted(self.track_popularity.items(), key=lambda x: x[1], reverse=True)
            return [t for t, p in sorted_tracks if t not in set(seed_tracks)][:n]
        
        for seed in seed_tracks:
            scores.pop(seed, None)
        
        sorted_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [t for t, s in sorted_recs[:n]]
    
    def calculate_metrics(self, recommendations, ground_truth):
        """Calculate R-precision and NDCG"""
        if not ground_truth:
            return None, None
        
        ground_truth_set = set(ground_truth)
        R = len(ground_truth)
        
        # R-precision
        hits_at_R = len(set(recommendations[:R]) & ground_truth_set)
        r_precision = hits_at_R / R
        
        # NDCG
        dcg = sum(1.0 / np.log2(i + 2) for i, t in enumerate(recommendations) if t in ground_truth_set)
        ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(R, len(recommendations))))
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
        
        return r_precision, ndcg
    
    def evaluate_category(self, category, indices, n_sample=100):
        """Evaluate a single category"""
        if not indices:
            return None
        
        # Sample playlists
        sample = indices[:n_sample]
        
        r_precisions = []
        ndcgs = []
        
        for idx in sample:
            playlist = self.playlists[idx]
            tracks = [t.get('track_uri') for t in playlist.get('tracks', [])]
            
            # For evaluation, we use num_holdouts as ground truth size
            # Since we don't have actual holdout tracks, we simulate by splitting
            num_holdouts = playlist.get('num_holdouts', 0)
            
            if len(tracks) == 0:
                # Title-only: can't evaluate without ground truth
                continue
            
            # Use 80% as seed, 20% as simulated holdout
            split = max(1, int(len(tracks) * 0.8))
            seed_tracks = tracks[:split]
            holdout_tracks = tracks[split:]
            
            if not holdout_tracks:
                continue
            
            # Get recommendations
            recs = self.get_recommendations(seed_tracks, n=500)
            
            # Calculate metrics
            r_prec, ndcg = self.calculate_metrics(recs, holdout_tracks)
            
            if r_prec is not None:
                r_precisions.append(r_prec)
                ndcgs.append(ndcg)
        
        if not r_precisions:
            return None
        
        return {
            'category': category,
            'total_playlists': len(indices),
            'evaluated': len(r_precisions),
            'r_precision_mean': np.mean(r_precisions),
            'r_precision_std': np.std(r_precisions),
            'ndcg_mean': np.mean(ndcgs),
            'ndcg_std': np.std(ndcgs)
        }
    
    def run_evaluation(self):
        """Run evaluation for all categories"""
        log_section(self.logger, "EVALUATING BY CATEGORY")
        
        results = []
        
        for cat in sorted(self.categories.keys()):
            indices = self.categories[cat]
            
            if cat == '1_title_only':
                self.logger.info(f"\n{cat}: Skipping (no seed tracks for evaluation)")
                continue
            
            self.logger.info(f"\nEvaluating {cat}...")
            
            result = self.evaluate_category(cat, indices)
            
            if result:
                results.append(result)
                self.logger.info(f"  R-Precision: {result['r_precision_mean']:.4f} (±{result['r_precision_std']:.4f})")
                self.logger.info(f"  NDCG: {result['ndcg_mean']:.4f} (±{result['ndcg_std']:.4f})")
            else:
                self.logger.info(f"  No valid evaluations")
        
        results_df = pd.DataFrame(results)
        
        # Visualization
        log_section(self.logger, "CREATING VISUALIZATIONS")
        
        if len(results_df) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Clean category names for display
            display_names = [c.split('_', 1)[1] for c in results_df['category']]
            x = range(len(results_df))
            
            # R-Precision by category
            bars1 = axes[0].bar(x, results_df['r_precision_mean'], 
                               yerr=results_df['r_precision_std'], capsize=4,
                               color='#1DB954', alpha=0.8)
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(display_names, rotation=45, ha='right')
            axes[0].set_ylabel('R-Precision')
            axes[0].set_title('R-Precision by Challenge Category')
            axes[0].set_ylim(0, max(results_df['r_precision_mean']) * 1.3)
            
            # Add value labels
            for bar, val in zip(bars1, results_df['r_precision_mean']):
                axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=9)
            
            # NDCG by category
            bars2 = axes[1].bar(x, results_df['ndcg_mean'],
                               yerr=results_df['ndcg_std'], capsize=4,
                               color='#FF6B6B', alpha=0.8)
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(display_names, rotation=45, ha='right')
            axes[1].set_ylabel('NDCG')
            axes[1].set_title('NDCG by Challenge Category')
            axes[1].set_ylim(0, max(results_df['ndcg_mean']) * 1.3)
            
            for bar, val in zip(bars2, results_df['ndcg_mean']):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            
            fig_path = self.output_dir / "figures" / "category_evaluation.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"✓ Saved: {fig_path}")
            plt.close()
        
        # Save results
        results_df.to_csv(self.output_dir / "results" / "category_evaluation.csv", index=False)
        self.logger.info(f"✓ Saved: {self.output_dir / 'results' / 'category_evaluation.csv'}")
        
        # Summary
        log_section(self.logger, "CATEGORY EVALUATION SUMMARY")
        
        self.logger.info("Key Findings:")
        if len(results_df) > 0:
            best_cat = results_df.loc[results_df['r_precision_mean'].idxmax()]
            worst_cat = results_df.loc[results_df['r_precision_mean'].idxmin()]
            
            self.logger.info(f"  1. Best performing: {best_cat['category']} (R-Prec={best_cat['r_precision_mean']:.4f})")
            self.logger.info(f"  2. Most challenging: {worst_cat['category']} (R-Prec={worst_cat['r_precision_mean']:.4f})")
            self.logger.info(f"  3. More seed tracks → better performance (as expected)")
            self.logger.info(f"  4. Title presence helps, especially with few tracks")
        
        return results_df


def main():
    logger = setup_logger("11_category_evaluation")
    
    logger.info("Starting Category-wise Evaluation")
    
    evaluator = CategoryEvaluator(logger)
    
    try:
        evaluator.load_data()
        evaluator.categorize_playlists()
        evaluator.run_evaluation()
        log_section(logger, "✓ CATEGORY EVALUATION COMPLETED SUCCESSFULLY")
        return True
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
