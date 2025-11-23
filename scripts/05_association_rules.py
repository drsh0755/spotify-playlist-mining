#!/usr/bin/env python3
"""
Association Rule Mining - Memory Optimized

Uses FP-Growth algorithm with filtered tracks to fit in 16GB RAM.
Answers: "If playlist has Song A, how likely is Song B?"
"""

import sys
import json
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from logger_config import setup_logger, log_section, log_subsection


def run_association_rules(logger, min_track_support=20, min_support=0.01, min_confidence=0.3):
    """
    Run association rule mining with memory optimization
    
    Args:
        min_track_support: Only use tracks appearing in this many playlists
        min_support: Minimum support for frequent itemsets
        min_confidence: Minimum confidence for rules
    """
    
    log_section(logger, "ASSOCIATION RULE MINING (MEMORY OPTIMIZED)")
    
    # Load data
    logger.info("Loading data...")
    data_dir = Path(__file__).parent.parent / "data"
    
    with open(data_dir / "raw" / "challenge_set.json", 'r') as f:
        challenge_data = json.load(f)
    playlists = challenge_data['playlists']
    
    tracks_df = pd.read_csv(data_dir / "processed" / "tracks.csv")
    logger.info(f"Loaded {len(playlists):,} playlists, {len(tracks_df):,} tracks")
    
    # Filter to high-frequency tracks only
    log_subsection(logger, f"Filtering Tracks (min support = {min_track_support} playlists)")
    
    high_freq_tracks = set(tracks_df[tracks_df['playlist_count'] >= min_track_support]['track_uri'])
    logger.info(f"Tracks with {min_track_support}+ appearances: {len(high_freq_tracks):,}")
    
    # Build transaction list
    logger.info("Building transactions...")
    transactions = []
    
    for playlist in tqdm(playlists, desc="Processing playlists"):
        track_uris = [t.get('track_uri') for t in playlist.get('tracks', [])]
        filtered_tracks = [uri for uri in track_uris if uri in high_freq_tracks]
        
        if len(filtered_tracks) >= 2:  # Need at least 2 tracks for rules
            transactions.append(filtered_tracks)
    
    logger.info(f"Valid transactions: {len(transactions):,}")
    
    # Create track ID mapping
    all_tracks = sorted(high_freq_tracks)
    track_to_idx = {t: i for i, t in enumerate(all_tracks)}
    idx_to_track = {i: t for t, i in track_to_idx.items()}
    
    n_tracks = len(all_tracks)
    n_transactions = len(transactions)
    
    logger.info(f"Matrix size: {n_transactions:,} × {n_tracks:,}")
    estimated_memory = (n_transactions * n_tracks) / (1024**3)  # GB (for bool)
    logger.info(f"Estimated memory: {estimated_memory:.2f} GB")
    
    if estimated_memory > 10:
        logger.warning(f"Matrix too large! Try increasing min_track_support")
        return None
    
    # Build one-hot encoded matrix with boolean dtype (memory efficient!)
    log_subsection(logger, "Building Transaction Matrix")
    
    from sklearn.preprocessing import MultiLabelBinarizer
    
    # Convert URIs to indices
    transactions_idx = [[track_to_idx[t] for t in trans] for trans in transactions]
    
    mlb = MultiLabelBinarizer(classes=list(range(n_tracks)))
    transaction_matrix = mlb.fit_transform(transactions_idx)
    
    # Convert to boolean DataFrame
    transactions_df = pd.DataFrame(transaction_matrix, dtype=bool)
    transactions_df.columns = [idx_to_track[i] for i in range(n_tracks)]
    
    logger.info(f"Transaction matrix shape: {transactions_df.shape}")
    logger.info(f"Memory usage: {transactions_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Run FP-Growth (faster and more memory efficient than Apriori)
    log_subsection(logger, "Running FP-Growth Algorithm")
    
    from mlxtend.frequent_patterns import fpgrowth, association_rules
    
    logger.info(f"Finding frequent itemsets (min_support={min_support})...")
    frequent_itemsets = fpgrowth(transactions_df, min_support=min_support, use_colnames=True, max_len=2)
    
    logger.info(f"Found {len(frequent_itemsets):,} frequent itemsets")
    
    if len(frequent_itemsets) == 0:
        logger.warning("No frequent itemsets found. Try lowering min_support.")
        return None
    
    # Generate association rules
    log_subsection(logger, "Generating Association Rules")
    
    logger.info(f"Generating rules (min_confidence={min_confidence})...")
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    if len(rules) == 0:
        logger.warning("No rules found. Try lowering min_confidence.")
        return None
    
    rules = rules.sort_values('lift', ascending=False)
    logger.info(f"Generated {len(rules):,} rules")
    
    # Add artist names
    log_subsection(logger, "Processing Results")
    
    track_to_artist = dict(zip(tracks_df['track_uri'], tracks_df['artist_name']))
    
    rules_data = []
    for _, rule in rules.iterrows():
        antecedent = list(rule['antecedents'])[0] if len(rule['antecedents']) == 1 else None
        consequent = list(rule['consequents'])[0] if len(rule['consequents']) == 1 else None
        
        if antecedent and consequent:
            rules_data.append({
                'antecedent_uri': antecedent,
                'antecedent_artist': track_to_artist.get(antecedent, 'Unknown'),
                'consequent_uri': consequent,
                'consequent_artist': track_to_artist.get(consequent, 'Unknown'),
                'support': rule['support'],
                'confidence': rule['confidence'],
                'lift': rule['lift']
            })
    
    rules_df = pd.DataFrame(rules_data)
    
    # Display top rules
    log_section(logger, "TOP ASSOCIATION RULES")
    
    logger.info("\nTop 25 Rules by Lift:")
    logger.info("(If playlist has A → likely has B)")
    logger.info("-" * 80)
    
    for i, rule in rules_df.head(25).iterrows():
        logger.info(f"\n  {i+1}. {rule['antecedent_artist'][:25]} → {rule['consequent_artist'][:25]}")
        logger.info(f"     Support: {rule['support']:.4f} | Confidence: {rule['confidence']:.2%} | Lift: {rule['lift']:.2f}x")
    
    # Save results
    output_dir = Path(__file__).parent.parent / "outputs" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rules_df.to_csv(output_dir / "association_rules.csv", index=False)
    logger.info(f"\n✓ Saved {len(rules_df)} rules to: {output_dir / 'association_rules.csv'}")
    
    # Summary statistics
    log_section(logger, "ASSOCIATION RULES SUMMARY")
    
    logger.info(f"Total rules generated: {len(rules_df):,}")
    logger.info(f"Average confidence: {rules_df['confidence'].mean():.2%}")
    logger.info(f"Average lift: {rules_df['lift'].mean():.2f}x")
    logger.info(f"Max lift: {rules_df['lift'].max():.2f}x")
    
    # Find strongest bidirectional rules
    logger.info("\n🔄 Strongest Bidirectional Relationships:")
    
    seen_pairs = set()
    bidirectional = []
    
    for _, r1 in rules_df.iterrows():
        pair = tuple(sorted([r1['antecedent_uri'], r1['consequent_uri']]))
        if pair in seen_pairs:
            continue
        
        # Look for reverse rule
        reverse = rules_df[
            (rules_df['antecedent_uri'] == r1['consequent_uri']) & 
            (rules_df['consequent_uri'] == r1['antecedent_uri'])
        ]
        
        if len(reverse) > 0:
            r2 = reverse.iloc[0]
            bidirectional.append({
                'artist1': r1['antecedent_artist'],
                'artist2': r1['consequent_artist'],
                'conf_1to2': r1['confidence'],
                'conf_2to1': r2['confidence'],
                'avg_lift': (r1['lift'] + r2['lift']) / 2
            })
            seen_pairs.add(pair)
    
    bidirectional_df = pd.DataFrame(bidirectional).sort_values('avg_lift', ascending=False)
    
    for i, row in bidirectional_df.head(10).iterrows():
        logger.info(f"  {row['artist1']} ↔ {row['artist2']}")
        logger.info(f"    {row['artist1']} → {row['artist2']}: {row['conf_1to2']:.1%}")
        logger.info(f"    {row['artist2']} → {row['artist1']}: {row['conf_2to1']:.1%}")
    
    return rules_df


def main():
    logger = setup_logger("05_association_rules")
    
    logger.info("Starting Association Rule Mining")
    logger.info(f"Working directory: {Path.cwd()}")
    
    try:
        # Run with optimized parameters
        rules = run_association_rules(
            logger,
            min_track_support=15,   # Only tracks in 15+ playlists (~3,200 tracks)
            min_support=0.005,      # 0.5% of playlists
            min_confidence=0.25     # 25% confidence
        )
        
        if rules is not None:
            log_section(logger, "✓ ASSOCIATION RULE MINING COMPLETED")
            logger.info("Script completed successfully")
            sys.exit(0)
        else:
            logger.error("No rules generated")
            sys.exit(1)
            
    except MemoryError:
        logger.error("Out of memory! Try increasing min_track_support parameter.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
