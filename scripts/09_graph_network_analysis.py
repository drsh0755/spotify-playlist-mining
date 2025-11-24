#!/usr/bin/env python3
"""
Graph/Network Analysis - Proposal Requirement 4.3.1

Builds and analyzes artist co-occurrence network:
- Network construction from co-occurrence data
- Centrality metrics (degree, betweenness)
- Community detection
- Network visualization
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from logger_config import setup_logger, log_section, log_subsection


def build_artist_network(logger):
    """Build and analyze artist co-occurrence network"""
    
    output_dir = Path(__file__).parent.parent / "outputs"
    
    log_section(logger, "LOADING ARTIST CO-OCCURRENCE DATA")
    
    artist_cooccur = pd.read_csv(output_dir / "results" / "artist_cooccurrences.csv")
    logger.info(f"Loaded {len(artist_cooccur):,} artist pairs")
    
    # Use top pairs for network (too many edges makes visualization messy)
    top_pairs = artist_cooccur.head(300)
    logger.info(f"Using top {len(top_pairs)} pairs for network")
    
    log_section(logger, "BUILDING NETWORK")
    
    G = nx.Graph()
    
    for _, row in top_pairs.iterrows():
        G.add_edge(row['artist1'], row['artist2'], weight=row['cooccurrence_count'])
    
    logger.info(f"Network created:")
    logger.info(f"  Nodes (artists): {G.number_of_nodes()}")
    logger.info(f"  Edges (connections): {G.number_of_edges()}")
    logger.info(f"  Density: {nx.density(G):.4f}")
    
    log_section(logger, "CENTRALITY ANALYSIS")
    
    # Degree centrality - most connected artists
    log_subsection(logger, "Degree Centrality (Most Connected)")
    degree_cent = nx.degree_centrality(G)
    top_degree = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:15]
    
    for i, (artist, cent) in enumerate(top_degree, 1):
        logger.info(f"  {i:2d}. {artist}: {cent:.4f}")
    
    # Betweenness centrality - bridge artists connecting different groups
    log_subsection(logger, "Betweenness Centrality (Bridge Artists)")
    between_cent = nx.betweenness_centrality(G)
    top_between = sorted(between_cent.items(), key=lambda x: x[1], reverse=True)[:15]
    
    for i, (artist, cent) in enumerate(top_between, 1):
        logger.info(f"  {i:2d}. {artist}: {cent:.4f}")
    
    # Eigenvector centrality - influential artists
    log_subsection(logger, "Eigenvector Centrality (Influential Artists)")
    try:
        eigen_cent = nx.eigenvector_centrality(G, max_iter=500)
        top_eigen = sorted(eigen_cent.items(), key=lambda x: x[1], reverse=True)[:15]
        
        for i, (artist, cent) in enumerate(top_eigen, 1):
            logger.info(f"  {i:2d}. {artist}: {cent:.4f}")
    except:
        logger.info("  (Eigenvector centrality did not converge)")
        eigen_cent = {}
    
    log_section(logger, "COMMUNITY DETECTION")
    
    # Detect communities using greedy modularity
    try:
        communities = list(nx.community.greedy_modularity_communities(G))
        logger.info(f"Detected {len(communities)} communities")
        
        for i, comm in enumerate(communities[:6], 1):
            members = sorted(comm, key=lambda x: degree_cent.get(x, 0), reverse=True)
            logger.info(f"\n  Community {i} ({len(comm)} artists):")
            logger.info(f"    Top members: {', '.join(members[:5])}")
    except Exception as e:
        logger.info(f"  Community detection failed: {e}")
        communities = []
    
    log_section(logger, "CREATING VISUALIZATIONS")
    
    fig_dir = output_dir / "figures"
    
    # Main network visualization
    logger.info("Creating network visualization...")
    
    plt.figure(figsize=(18, 14))
    
    # Node sizes based on degree centrality
    node_sizes = [degree_cent.get(node, 0.01) * 4000 + 200 for node in G.nodes()]
    
    # Node colors based on community (if detected)
    if communities:
        node_colors = []
        community_map = {}
        for i, comm in enumerate(communities):
            for node in comm:
                community_map[node] = i
        node_colors = [community_map.get(node, 0) for node in G.nodes()]
        cmap = 'tab10'
    else:
        node_colors = 'lightblue'
        cmap = None
    
    # Edge weights for thickness
    edge_weights = [G[u][v]['weight'] / 2000 + 0.5 for u, v in G.edges()]
    
    # Layout
    pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)
    
    # Draw
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
                          cmap=cmap, alpha=0.7)
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.3, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=7, font_weight='bold')
    
    plt.title("Artist Co-occurrence Network\n(Node size = connectivity, Colors = communities)", 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    fig_path = fig_dir / "artist_network.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {fig_path}")
    plt.close()
    
    # Centrality comparison chart
    logger.info("Creating centrality comparison chart...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Top 10 by degree
    top10_degree = dict(top_degree[:10])
    axes[0].barh(list(top10_degree.keys())[::-1], list(top10_degree.values())[::-1], color='#1DB954')
    axes[0].set_xlabel('Degree Centrality')
    axes[0].set_title('Most Connected Artists')
    
    # Top 10 by betweenness
    top10_between = dict(top_between[:10])
    axes[1].barh(list(top10_between.keys())[::-1], list(top10_between.values())[::-1], color='#FF6B6B')
    axes[1].set_xlabel('Betweenness Centrality')
    axes[1].set_title('Bridge Artists')
    
    # Degree distribution
    degrees = [d for n, d in G.degree()]
    axes[2].hist(degrees, bins=20, edgecolor='black', alpha=0.7, color='#4ECDC4')
    axes[2].set_xlabel('Degree (Number of Connections)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Degree Distribution')
    
    plt.tight_layout()
    
    fig_path = fig_dir / "network_centrality.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {fig_path}")
    plt.close()
    
    # Save metrics
    log_section(logger, "SAVING RESULTS")
    
    metrics_df = pd.DataFrame({
        'artist': list(degree_cent.keys()),
        'degree_centrality': list(degree_cent.values()),
        'betweenness_centrality': [between_cent.get(a, 0) for a in degree_cent.keys()],
        'eigenvector_centrality': [eigen_cent.get(a, 0) for a in degree_cent.keys()]
    }).sort_values('degree_centrality', ascending=False)
    
    metrics_path = output_dir / "results" / "network_centrality_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"✓ Saved: {metrics_path}")
    
    # Summary
    log_section(logger, "NETWORK ANALYSIS SUMMARY")
    
    logger.info("Key Findings:")
    logger.info(f"  1. Most connected artist: {top_degree[0][0]} (degree={top_degree[0][1]:.4f})")
    logger.info(f"  2. Top bridge artist: {top_between[0][0]} (betweenness={top_between[0][1]:.4f})")
    logger.info(f"  3. Network has {len(communities)} distinct communities")
    logger.info(f"  4. Average clustering coefficient: {nx.average_clustering(G):.4f}")
    
    return G, metrics_df


def main():
    logger = setup_logger("09_graph_network_analysis")
    
    logger.info("Starting Graph/Network Analysis (Proposal 4.3.1)")
    
    try:
        G, metrics = build_artist_network(logger)
        log_section(logger, "✓ GRAPH ANALYSIS COMPLETED SUCCESSFULLY")
        return True
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
