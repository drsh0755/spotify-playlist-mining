# Visualization Suite - Usage Guide

**Complete:** All 17 figures implemented  
**Runtime:** 5-10 minutes  
**Output:** Publication & presentation-ready figures

---

## 🎨 What's Included

### All 17 Figures:

**Category 1: Dataset Overview**
1. ✅ Dataset overview dashboard (4-panel)
2. ✅ Co-occurrence heatmap
3. ✅ Association rules network
4. ✅ Association metrics (3-panel)

**Category 2: Clustering (RQ2)**
5. ✅ Cluster visualization (PCA)
6. ✅ Cluster characteristics (radar charts)
7. ✅ Silhouette analysis

**Category 3: Recommendations (RQ3)**
8. ✅ Model performance comparison
9. ✅ Performance by category
10. ✅ Diversity analysis
11. ✅ Recommendation example

**Category 4: Advanced Modeling (Phase 3)**
12. ✅ SVD latent factors
13. ✅ Neural embeddings (t-SNE)
14. ✅ Predictive models (confusion matrix + regression)
15. ✅ Feature importance

**Category 5: Project Meta**
16. ✅ Project timeline
17. ✅ Computational performance

---

## 🚀 Quick Start

### Step 1: Place Script in Project

```bash
# Download the script from Claude
# Place it in your project:
cd ~/Desktop/CSCI\ 6443\ Data\ Mining\ -\ Project
cp ~/Downloads/40_create_all_visualizations_complete.py scripts/

# Make executable (optional)
chmod +x scripts/40_create_all_visualizations_complete.py
```

### Step 2: Run the Script

```bash
# Activate your environment
source venv/bin/activate

# Run the complete suite
python scripts/40_create_all_visualizations_complete.py
```

**That's it!** The script will:
- Create `outputs/figures/publication/` directory
- Create `outputs/figures/presentation/` directory  
- Generate all 17 figures
- Create a manifest file
- Take 5-10 minutes total

---

## 📂 Output Structure

```
outputs/figures/
├── publication/              # High-res (300 DPI) for papers
│   ├── 01_dataset_overview.png
│   ├── 02_cooccurrence_heatmap.png
│   ├── 03_association_network.png
│   ├── ... (all 17 figures)
│   └── 17_computational_performance.png
├── presentation/             # Medium-res (150 DPI) for slides
│   ├── 01_dataset_overview.png
│   └── ... (all 17 figures)
└── FIGURE_MANIFEST.md       # Index of all figures
```

---

## 📊 What Each Figure Shows

### Figure 1: Dataset Overview
**4-panel dashboard:**
- Playlist length distribution
- Track popularity (power law)
- Top 20 artists
- Summary statistics table

**Use in:** Introduction section, dataset description

---

### Figure 2: Co-occurrence Heatmap
**50×50 heatmap** of most popular tracks showing co-occurrence patterns

**Use in:** Research Question 1 results

---

### Figure 3: Association Rules Network
**Network graph** showing strong associations (confidence >0.7, lift >10)

**Use in:** Research Question 1 visualization, presentations

---

### Figure 4: Association Rule Metrics
**3-panel analysis:**
- Support vs Confidence scatter
- Lift distribution
- Top 10 rules by lift

**Use in:** Research Question 1 detailed analysis

---

### Figure 5: Cluster Visualization (PCA)
**2D PCA projection** of 10K playlists colored by cluster (k=12)

**Use in:** Research Question 2 main result, presentations

---

### Figure 6: Cluster Characteristics
**12 radar charts** showing characteristics of each cluster

**Use in:** Research Question 2 detailed analysis

---

### Figure 7: Silhouette Analysis
**2-panel:**
- Elbow curve for k selection
- Silhouette scores (k=5 to k=30)

**Use in:** Research Question 2 methodology

---

### Figure 8: Model Performance Comparison ⭐
**Grouped bar chart** comparing R-Precision and NDCG across 4 models

**Use in:** Research Question 3 main result, abstract, presentations, poster

---

### Figure 9: Performance by Category
**Line plot** showing model performance across difficulty categories

**Use in:** Research Question 3 robustness analysis

---

### Figure 10: Diversity Analysis
**2-panel:**
- Artist diversity bar chart
- Popularity distribution violin plots

**Use in:** Research Question 3 quality analysis

---

### Figure 11: Recommendation Example
**Visual example** showing seed tracks and top 10 recommendations with reasoning

**Use in:** Presentations, explaining approach to audience

---

### Figure 12: SVD Latent Factors
**Heatmap** of latent factors showing interpretable patterns

**Use in:** Phase 3 methods, advanced analysis

---

### Figure 13: Neural Embeddings (t-SNE)
**2D visualization** of 7-dimensional embeddings colored by genre

**Use in:** Phase 3 results, presentations

---

### Figure 14: Predictive Models
**2-panel:**
- Classification confusion matrix (99.6%)
- Regression actual vs predicted scatter (R²=81.7%)

**Use in:** Phase 3 results

---

### Figure 15: Feature Importance
**2-panel horizontal bar charts:**
- Classification task features
- Regression task features

**Use in:** Phase 3 analysis, feature engineering insights

---

### Figure 16: Timeline
**Project timeline** with key milestones and pivot point

**Use in:** Presentations (development journey), process documentation

---

### Figure 17: Computational Performance
**2-panel bar charts:**
- Runtime for each major component
- Memory usage comparison

**Use in:** Presentations (efficiency), technical discussions

---

## 🎨 Customization

### Change Colors

Edit the `COLORS` dictionary at the top of the script:

```python
COLORS = {
    'primary': '#1f77b4',     # Your primary color
    'secondary': '#ff7f0e',   # Your secondary color
    # ... etc
}
```

### Change Figure Sizes

Edit the `figsize` parameter in each function:

```python
fig, ax = plt.subplots(figsize=(14, 10))  # (width, height) in inches
```

### Change DPI

Edit the `save_figure` method:

```python
def save_figure(self, fig, name):
    fig.savefig(self.pub_dir / f"{name}.png", dpi=300, ...)  # Change 300
    fig.savefig(self.pres_dir / f"{name}.png", dpi=150, ...) # Change 150
```

---

## 🐛 Troubleshooting

### Issue 1: Missing Data Files

**Error:** `FileNotFoundError: data/processed/tracks_full_mpd.parquet`

**Solution:**
```bash
# Run Phase 1 pipeline first
python scripts/24_phase1_master_pipeline.py
```

### Issue 2: Missing Results Files

**Error:** `FileNotFoundError: outputs/results/association_rules_full.csv`

**Solution:**
```bash
# Run Phase 2 pipeline first
python scripts/31_phase2_master_pipeline.py
```

**OR:** The script will create sample/placeholder visualizations for missing data

### Issue 3: Import Errors

**Error:** `ModuleNotFoundError: No module named 'seaborn'`

**Solution:**
```bash
pip install matplotlib seaborn networkx scipy scikit-learn
```

### Issue 4: Memory Errors

**Error:** `MemoryError` or process killed

**Solution:**
- Close other applications
- Reduce sample sizes in the script
- Comment out memory-intensive figures (2, 3, 5)

---

## 📝 Figure Manifest

After running, check `outputs/figures/FIGURE_MANIFEST.md` for:
- List of all generated figures
- Success/failure status
- Recommended usage for each figure

---

## ⏱️ Performance

**Expected runtime:**
- Fast figures (5-8, 11, 16-17): <10 seconds each
- Medium figures (1, 4, 9-10, 14-15): 10-30 seconds each
- Slow figures (2-3, 5-6, 12-13): 30-60 seconds each

**Total:** ~5-10 minutes for all 17 figures

---

## 🎯 Recommended Figure Sets

### For Research Paper (Core):
- Figure 1 (dataset)
- Figure 4 (association rules)
- Figure 5 (clusters)
- Figure 8 (performance comparison) ⭐
- Figure 9 (category performance)
- Figure 15 (feature importance)

### For Presentation (Visual Impact):
- Figure 1 (overview)
- Figure 5 (clusters PCA)
- Figure 8 (performance) ⭐
- Figure 11 (example)
- Figure 13 (embeddings)
- Figure 16 (timeline)

### For Poster (High-Level):
- Figure 1 (overview)
- Figure 8 (performance) ⭐
- Figure 5 (clusters)
- Figure 10 (diversity)

---

## ✨ Tips for Best Results

1. **Run after all pipelines complete** - Ensures real data for all figures

2. **Check figure quality** - View at actual size (300 DPI = print quality)

3. **Test in grayscale** - Convert to see if figures work without color

4. **Use consistent style** - All figures use same color palette

5. **Annotate important insights** - Figures have built-in annotations

6. **Export formats** - PNG for most uses, can convert to PDF/EPS if needed

---

## 📚 Next Steps After Generating

1. **Review all figures** - Check for quality and accuracy

2. **Select for paper** - Choose 6-8 most important figures

3. **Select for presentation** - Choose 6-10 visual impact figures

4. **Create captions** - Write descriptive captions for each

5. **Reference in text** - "As shown in Figure 8..."

6. **Create composite figures** - Combine related figures if needed

---

**Ready to visualize your results?** Run the script and generate all 17 publication-quality figures! 📊✨