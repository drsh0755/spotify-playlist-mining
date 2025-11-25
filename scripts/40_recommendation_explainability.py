"""
Recommendation Explainability
Explain why tracks are recommended

Author: Adarsh Singh
Date: November 2024
"""

import pandas as pd
import numpy as np
from scipy import sparse
from pathlib import Path
import logging
from datetime import datetime
import json
import pickle

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'explainability_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RecommendationExplainability:
    """Explain recommendation reasoning."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_models_and_data(self):
        """Load necessary data."""
        logger.info("Loading models and data...")
        
        # Load co-occurrence
        cooccurrence = sparse.load_npz("data/processed/cooccurrence_matrix_full.npz")
        
        # Load mappings
        with open("data/processed/track_mappings.pkl", "rb") as f:
            mappings = pickle.load(f)
        
        # Load association rules
        rules_df = pd.read_csv("data/processed/association_rules_full.csv")
        
        logger.info(f"Loaded co-occurrence matrix: {cooccurrence.shape}")
        logger.info(f"Loaded {len(rules_df):,} association rules")
        
        return cooccurrence, mappings, rules_df
    
    def explain_cooccurrence_recommendation(self, seed_track, recommended_track, cooccurrence, mappings):
        """Explain why a track is recommended based on co-occurrence."""
        
        if seed_track not in mappings['track_to_idx'] or recommended_track not in mappings['track_to_idx']:
            return None
        
        seed_idx = mappings['track_to_idx'][seed_track]
        rec_idx = mappings['track_to_idx'][recommended_track]
        
        # Get co-occurrence score
        score = cooccurrence[seed_idx, rec_idx]
        
        # Get total occurrences
        seed_total = cooccurrence[seed_idx].sum()
        rec_total = cooccurrence[rec_idx].sum()
        
        explanation = {
            'seed_track': seed_track,
            'recommended_track': recommended_track,
            'cooccurrence_count': int(score),
            'seed_total_connections': int(seed_total),
            'rec_total_connections': int(rec_total),
            'cooccurrence_strength': float(score / seed_total) if seed_total > 0 else 0.0,
            'explanation': f"These tracks appeared together in {int(score):,} playlists. "
                          f"When '{seed_track}' is in a playlist, there's a {float(score / seed_total * 100):.2f}% "
                          f"chance '{recommended_track}' is also there."
        }
        
        return explanation
    
    def explain_association_rule(self, seed_track, recommended_track, rules_df):
        """Explain using association rules."""
        
        # Find relevant rules
        relevant_rules = rules_df[
            (rules_df['antecedent'] == seed_track) & 
            (rules_df['consequent'] == recommended_track)
        ]
        
        if len(relevant_rules) == 0:
            return None
        
        rule = relevant_rules.iloc[0]
        
        explanation = {
            'seed_track': seed_track,
            'recommended_track': recommended_track,
            'support': float(rule['support']),
            'confidence': float(rule['confidence']),
            'lift': float(rule['lift']),
            'explanation': f"Association rule: {seed_track} → {recommended_track}. "
                          f"Support: {rule['support']:.4f}, "
                          f"Confidence: {rule['confidence']:.2%}, "
                          f"Lift: {rule['lift']:.2f}. "
                          f"This means when '{seed_track}' appears, '{recommended_track}' appears "
                          f"{rule['confidence']:.0%} of the time, which is {rule['lift']:.1f}x more than random."
        }
        
        return explanation
    
    def generate_sample_explanations(self, cooccurrence, mappings, rules_df, n_samples=10):
        """Generate sample explanations."""
        logger.info("\n" + "="*60)
        logger.info("Sample Recommendation Explanations")
        logger.info("="*60)
        
        # Get sample tracks
        np.random.seed(42)
        sample_indices = np.random.choice(len(mappings['idx_to_track']), min(n_samples, 100), replace=False)
        
        explanations = []
        
        for idx in sample_indices:
            seed_track = mappings['idx_to_track'][idx]
            
            # Get top recommendation
            scores = cooccurrence[idx].toarray().flatten()
            top_rec_idx = np.argsort(scores)[::-1][1]  # Skip self
            rec_track = mappings['idx_to_track'][top_rec_idx]
            
            # Explain
            cooc_exp = self.explain_cooccurrence_recommendation(seed_track, rec_track, cooccurrence, mappings)
            rule_exp = self.explain_association_rule(seed_track, rec_track, rules_df)
            
            if cooc_exp:
                explanations.append({
                    'type': 'co-occurrence',
                    **cooc_exp
                })
                
                logger.info(f"\nSeed: {seed_track}")
                logger.info(f"Recommended: {rec_track}")
                logger.info(f"Reason: {cooc_exp['explanation']}")
                
                if rule_exp:
                    logger.info(f"Rule: Confidence={rule_exp['confidence']:.2%}, Lift={rule_exp['lift']:.2f}")
            
            if len(explanations) >= n_samples:
                break
        
        return explanations
    
    def analyze_recommendation_factors(self, cooccurrence, mappings):
        """Analyze what factors drive recommendations."""
        logger.info("\n" + "="*60)
        logger.info("Recommendation Factor Analysis")
        logger.info("="*60)
        
        # Sample tracks
        n_samples = 1000
        np.random.seed(42)
        sample_indices = np.random.choice(len(mappings['idx_to_track']), min(n_samples, len(mappings['idx_to_track'])), replace=False)
        
        cooc_strengths = []
        for idx in sample_indices:
            scores = cooccurrence[idx].toarray().flatten()
            if scores.sum() > 0:
                # Normalize
                normalized_scores = scores / scores.sum()
                # Calculate entropy (measure of diversity)
                non_zero = normalized_scores[normalized_scores > 0]
                entropy = -np.sum(non_zero * np.log(non_zero))
                cooc_strengths.append(entropy)
        
        factors = {
            'mean_recommendation_entropy': float(np.mean(cooc_strengths)),
            'median_recommendation_entropy': float(np.median(cooc_strengths)),
            'explanation': "Higher entropy means recommendations are more diverse/spread out. "
                          "Lower entropy means recommendations are more concentrated on a few tracks."
        }
        
        logger.info(f"Mean Recommendation Entropy: {factors['mean_recommendation_entropy']:.4f}")
        logger.info(f"Median Recommendation Entropy: {factors['median_recommendation_entropy']:.4f}")
        logger.info(factors['explanation'])
        
        return factors
    
    def save_results(self, explanations, factors):
        """Save explainability results."""
        logger.info("\nSaving results...")
        
        results = {
            'sample_explanations': explanations,
            'recommendation_factors': factors
        }
        
        # Save JSON
        output_file = self.output_dir / "recommendation_explainability.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved: {output_file}")

def main():
    """Main execution."""
    
    OUTPUT_DIR = "data/processed"
    
    explainer = RecommendationExplainability(output_dir=OUTPUT_DIR)
    
    # Load data
    cooccurrence, mappings, rules_df = explainer.load_models_and_data()
    
    # Generate explanations
    explanations = explainer.generate_sample_explanations(cooccurrence, mappings, rules_df, n_samples=10)
    
    # Analyze factors
    factors = explainer.analyze_recommendation_factors(cooccurrence, mappings)
    
    # Save results
    explainer.save_results(explanations, factors)
    
    logger.info("\n" + "="*60)
    logger.info("✅ Recommendation explainability analysis complete!")
    logger.info("="*60)

if __name__ == "__main__":
    main()