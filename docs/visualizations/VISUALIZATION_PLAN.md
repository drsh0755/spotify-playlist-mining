# Visualization Plan - Spotify Playlist Extension

**Purpose:** Create publication-quality figures for paper, presentation, and portfolio

---

## 📊 Visualization Categories

### Category 1: Dataset Overview (2-3 figures)
Understanding the data we're working with

### Category 2: Research Question 1 - Co-occurrence (3-4 figures)
Song co-occurrence patterns and associations

### Category 3: Research Question 2 - Clustering (3-4 figures)
Playlist and track groupings

### Category 4: Research Question 3 - Recommendations (4-5 figures)
Recommendation quality and comparisons

### Category 5: Advanced Analysis (2-3 figures)
Phase 3 results and insights

**Total: 15-20 high-quality visualizations**

---

## 🎨 Figure 1: Dataset Overview Dashboard

**File:** `01_dataset_overview.png`

**What to show:**
- 4-panel figure showing:
  - Top-left: Playlist length distribution (histogram)
  - Top-right: Track popularity distribution (log scale)
  - Bottom-left: Artist frequency (top 20 bar chart)
  - Bottom-right: Key statistics table

**Data sources:**
- `data/processed/tracks_full_mpd.parquet`
- `data/processed/playlists_full_mpd.parquet`

**Key insights to highlight:**
- 1M playlists, 2.3M tracks
- Average playlist: 66 tracks
- Power law distribution in popularity

---

## 🎨 Figure 2: Track Co-occurrence Heatmap

**File:** `02_cooccurrence_heatmap.png`

**What to show:**
- Heatmap of top 50 most popular tracks
- Shows which tracks frequently appear together
- Hierarchical clustering to group similar tracks

**Data sources:**
- `data/processed/cooccurrence_matrix_full.npz`
- Top 50 tracks by frequency

**Key insights:**
- Visual clusters of related tracks
- Genre groupings visible
- Strong co-occurrence patterns

---

## 🎨 Figure 3: Association Rules Network

**File:** `03_association_rules_network.png`

**What to show:**
- Network graph of top 100 association rules
- Nodes: Tracks
- Edges: Association rules (thickness = confidence)
- Colors: Genre or cluster
- Layout: Force-directed

**Data sources:**
- `outputs/results/association_rules_full.csv`
- Filter: confidence > 0.7, lift > 10

**Key insights:**
- Strong track associations visible
- Genre-based communities
- Hub tracks (high centrality)

---

## 🎨 Figure 4: Association Rule Metrics

**File:** `04_association_rules_metrics.png`

**What to show:**
- 3-panel figure:
  - Support vs Confidence scatter
  - Lift distribution histogram
  - Top 10 rules by lift (bar chart)

**Data sources:**
- `outputs/results/association_rules_full.csv`

**Key insights:**
- Rule quality distribution
- Most interesting associations
- Trade-off between support and confidence

---

## 🎨 Figure 5: Cluster Visualization (PCA)

**File:** `05_clusters_pca.png`

**What to show:**
- 2D PCA projection of playlists
- Color-coded by cluster (k=12)
- Larger sample for visibility (10K playlists)
- Cluster centroids marked

**Data sources:**
- `data/processed/tfidf_features_full.npz`
- `data/processed/cluster_assignments.pkl`

**Key insights:**
- Clear cluster separation
- Some overlap (mixed playlists)
- Distinct thematic groups

---

## 🎨 Figure 6: Cluster Characteristics

**File:** `06_cluster_characteristics.png`

**What to show:**
- Radar chart for each cluster (12 subplots)
- Axes: avg_length, popularity, diversity, energy, etc.
- Shows cluster "personality"

**Data sources:**
- `outputs/results/cluster_profiles.csv`

**Key insights:**
- Workout clusters: high energy, shorter
- Chill clusters: low energy, longer
- Party clusters: high popularity

---

## 🎨 Figure 7: Silhouette Analysis

**File:** `07_silhouette_analysis.png`

**What to show:**
- Two panels:
  - Left: Elbow curve (k vs inertia)
  - Right: Silhouette scores (k=5 to k=30)
- Optimal k=12 highlighted

**Data sources:**
- Re-compute or load from clustering script

**Key insights:**
- k=12 is optimal
- Clear elbow at k=12
- High silhouette score (0.68)

---

## 🎨 Figure 8: Model Performance Comparison

**File:** `08_model_performance.png`

**What to show:**
- Grouped bar chart
- Models: Popularity, Co-occurrence, SVD, Hybrid
- Metrics: R-Precision, NDCG@500
- Error bars if available

**Data sources:**
- `outputs/results/recommendation_evaluation.csv`

**Key insights:**
- Hybrid model best (89x improvement)
- SVD strong second
- Popularity baseline poor

---

## 🎨 Figure 9: Performance by Category

**File:** `09_performance_by_category.png`

**What to show:**
- Line plot
- X-axis: Category (0-9, hardest to easiest)
- Y-axis: R-Precision
- Multiple lines for different models

**Data sources:**
- `outputs/results/category_evaluation.csv`

**Key insights:**
- Performance improves with more seeds
- Hybrid model most robust
- All models struggle with category 0

---

## 🎨 Figure 10: Diversity Analysis

**File:** `10_diversity_analysis.png`

**What to show:**
- 2-panel figure:
  - Artist diversity (%) by model
  - Popularity distribution (violin plot)

**Data sources:**
- `outputs/results/diversity_metrics.csv`

**Key insights:**
- Hybrid maintains 78% diversity
- Popularity baseline only 45%
- Balance relevance and exploration

---

## 🎨 Figure 11: Recommendation Example

**File:** `11_recommendation_example.png`

**What to show:**
- Visual example of recommendations
- Top: Seed tracks (3-5 tracks with album art placeholders)
- Bottom: Top 10 recommendations
- Annotations showing why recommended

**Data sources:**
- Pick interesting example from evaluation
- Or generate live

**Key insights:**
- Clear thematic consistency
- Genre coherence
- Explanation of reasoning

---

## 🎨 Figure 12: Matrix Factorization - Latent Factors

**File:** `12_svd_latent_factors.png`

**What to show:**
- Heatmap of top tracks × factors
- Shows what each latent factor captures
- Annotate interpretable factors

**Data sources:**
- `data/processed/models/svd_model.pkl`
- V matrix (track factors)

**Key insights:**
- Factor 1: Genre (rock vs pop)
- Factor 2: Energy level
- Factor 3: Era/decade

---

## 🎨 Figure 13: Neural Network Embeddings

**File:** `13_neural_embeddings.png`

**What to show:**
- t-SNE projection of 7-dim embeddings
- Color by genre
- Sample 1,000 tracks for clarity

**Data sources:**
- `data/processed/models/neural_recommender.pkl`
- Track embeddings

**Key insights:**
- Clear genre clusters
- Semantic similarity preserved
- Continuous space

---

## 🎨 Figure 14: Predictive Model Results

**File:** `14_predictive_models.png`

**What to show:**
- 2-panel figure:
  - Left: Classification confusion matrix
  - Right: Regression actual vs predicted scatter

**Data sources:**
- `outputs/results/predictive_model_results.csv`
- Or regenerate from script 34

**Key insights:**
- 99.6% classification accuracy
- 81.7% R² for regression
- Strong predictive power

---

## 🎨 Figure 15: Feature Importance

**File:** `15_feature_importance.png`

**What to show:**
- Horizontal bar charts
- Top: Classification features
- Bottom: Regression features
- Color-coded by importance

**Data sources:**
- Script 34 feature importance

**Key insights:**
- Album popularity most important (50%)
- Position variability key for regression
- Track-level features matter

---

## 🎨 Figure 16: Timeline & Milestones

**File:** `16_project_timeline.png`

**What to show:**
- Gantt-style timeline
- Nov 15-25 with key milestones
- AWS → Local pivot marked
- Phase completions shown

**Data sources:**
- DEVELOPMENT_JOURNEY.md

**Key insights:**
- 10-day project
- Key pivot on day 4
- All phases completed

---

## 🎨 Figure 17: Computational Performance

**File:** `17_computational_performance.png`

**What to show:**
- Bar chart comparing:
  - Runtime (minutes)
  - Memory usage (GB)
  - For each major script/phase

**Data sources:**
- Log files or DEVELOPMENT_JOURNEY.md

**Key insights:**
- Phase 1: 22 min
- Phase 2: 45 min  
- Efficient implementation

---

## 📋 Implementation Plan

### Step 1: Create visualization script
**File:** `scripts/40_create_all_visualizations.py`
- Single script to generate all figures
- Modular functions for each figure
- Consistent styling throughout

### Step 2: Define style
- Color palette (colorblind-friendly)
- Font sizes (readable in papers/slides)
- Figure sizes (publication standard)
- DPI: 300 for papers, 150 for slides

### Step 3: Generate figures
- Run script once
- Output to `outputs/figures/publication/`
- Also create `outputs/figures/presentation/` (higher contrast)

### Step 4: Quality check
- Verify all figures generated
- Check readability
- Test in grayscale
- Ensure consistency

---

## 🎨 Style Guide

### Color Palette (Colorblind-Safe)
```python
COLORS = {
    'primary': '#1f77b4',      # Blue
    'secondary': '#ff7f0e',    # Orange
    'success': '#2ca02c',      # Green
    'danger': '#d62728',       # Red
    'warning': '#9467bd',      # Purple
    'info': '#8c564b',         # Brown
    'accent': '#e377c2',       # Pink
    'neutral': '#7f7f7f',      # Gray
}

# For clusters (12 colors)
CLUSTER_COLORS = sns.color_palette("tab20", 12)
```

### Figure Sizes
```python
SIZES = {
    'single': (8, 6),         # Single plot
    'double': (12, 6),        # Side-by-side
    'quad': (12, 10),         # 2×2 grid
    'wide': (14, 5),          # Timeline
}
```

### Fonts
```python
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica']
})
```

---

## 📦 Deliverables

### Output Structure
```
outputs/figures/
├── publication/          # High-res (300 DPI)
│   ├── 01_dataset_overview.png
│   ├── 02_cooccurrence_heatmap.png
│   ├── ...
│   └── 17_computational_performance.png
├── presentation/         # High-contrast for slides
│   ├── 01_dataset_overview.png
│   └── ...
└── thumbnails/          # Small preview versions
    └── ...
```

### Figure Manifest
**File:** `outputs/figures/FIGURE_MANIFEST.md`
- Lists all figures
- Descriptions
- Data sources
- Key insights
- Where to use (paper/slides/poster)

---

## 🚀 Next Steps

1. **Create visualization script** (scripts/40_create_all_visualizations.py)
2. **Generate all figures**
3. **Review and refine**
4. **Create figure manifest**
5. **Export for different uses**

Ready to start building? Let's create the visualization script!