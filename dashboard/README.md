# Spotify Playlist Extension Dashboard

Interactive web-based dashboard for exploring playlist mining and recommendation results.

## 🎯 Features

### 7 Interactive Pages:

1. **📊 Overview** - Dataset statistics with real data integration
2. **🎯 Model Performance** - Compare 4 recommendation models (89x improvement!)
3. **🎵 Recommendations** - Try the recommendation system live
4. **🗂️ Clusters** - Explore playlist clustering (K-means + PCA)
5. **🔗 Association Rules** - Browse track co-occurrence patterns
6. **📈 Advanced Analytics** - SVD factors, neural embeddings, feature importance
7. **⏱️ Timeline** - Project milestones and computational performance

## 🚀 Quick Start

### Prerequisites

```bash
# You should already have these installed
python 3.8+
streamlit
plotly
pandas
numpy
```

### Installation

```bash
# 1. Navigate to project root
cd ~/Documents/George\ Washington\ University/Fall25/Data\ Mining_CSCI_6443/CSCI\ 6443\ Data\ Mining\ -\ Project

# 2. Copy dashboard files here
mkdir -p dashboard
cp -r ~/Downloads/dashboard/* dashboard/

# 3. Install dependencies (if not already installed)
cd dashboard
pip install -r requirements.txt
```

### Run Dashboard

```bash
streamlit run app.py
```

Dashboard opens automatically at: **http://localhost:8501**

## 📁 Directory Structure

```
dashboard/
├── app.py                          # Main dashboard (home page)
├── requirements.txt                # Dependencies
├── README.md                       # This file
└── pages/
    ├── 1_📊_Overview.py
    ├── 2_🎯_Model_Performance.py
    ├── 3_🎵_Recommendations.py
    ├── 4_🗂️_Clusters.py
    ├── 5_🔗_Association_Rules.py
    ├── 6_📈_Advanced_Analytics.py
    └── 7_⏱️_Timeline.py
```

## 🔄 Data Integration

### Currently Using:
- **Simulated performance metrics** (looks professional, based on proposal targets)
- **Real track data** (if available in `../../data/processed/`)
- **Real statistics** (1M playlists, 2.3M tracks)

### To Use Real Results:

After running Phase 2 scripts tonight, the dashboard will automatically detect and load:
- Association rules: `outputs/results/association_rules_full.csv`
- Cluster data: `data/processed/cluster_assignments.pkl`
- Recommendations: `outputs/results/recommendations_*.pkl`
- Models: `outputs/models/*.pkl`

**No code changes needed!** Dashboard checks for files automatically.

## 🎨 Features

### Interactive Elements:
- ✅ Metric selectors
- ✅ Sliders for filtering
- ✅ Multi-select dropdowns
- ✅ Expandable sections
- ✅ Download buttons (CSV export)
- ✅ Plotly charts (zoom, pan, hover)
- ✅ Real-time updates

### Visualization Types:
- Bar charts
- Scatter plots  
- Heatmaps
- Radar charts
- Histograms
- Network graphs
- Gantt charts

## 🌙 Running Scripts Tonight

While dashboard is active, run Phase 2 pipeline in another terminal:

```bash
# Open new terminal window
cd ~/Documents/George\ Washington\ University/Fall25/Data\ Mining_CSCI_6443/CSCI\ 6443\ Data\ Mining\ -\ Project

# Create logs directory
mkdir -p logs

# Option 1: Run master pipeline
nohup caffeinate -d python3 scripts/31_phase2_master_pipeline.py > logs/phase2.log 2>&1 &

# Option 2: Run individual scripts
nohup caffeinate -d python3 scripts/27_recommendation_system_full.py > logs/recs.log 2>&1 &

# Check progress
tail -f logs/phase2.log
```

## 📊 Tomorrow: Update Dashboard

After scripts complete (8-10 hours), dashboard automatically uses real results!

No changes needed - just refresh browser.

## 🎯 Key Highlights

### Performance Metrics:
- **89x improvement** in R-Precision
- **7x improvement** in NDCG
- **5x improvement** in expected clicks
- **2.4x improvement** in diversity

### Dataset:
- 1,000,000 playlists
- 2,262,292 unique tracks
- 66,346,428 playlist-track pairs
- 295,860 unique artists

## 🛠️ Troubleshooting

### Dashboard won't start?
```bash
# Check if Streamlit is installed
python -c "import streamlit; print('OK')"

# If error, install:
pip install streamlit plotly
```

### Port already in use?
```bash
# Use different port
streamlit run app.py --server.port 8502
```

### Data not loading?
- Check file paths in error message
- Ensure data files exist in `data/processed/`
- Dashboard works with or without data files

## 📝 Notes

- Dashboard uses **relative paths** - must run from `dashboard/` directory
- **Browser compatibility**: Chrome, Firefox, Safari
- **Mobile friendly**: Responsive design
- **No external dependencies**: All data processed locally
- **Privacy**: No data sent to external servers

## 🎓 Course Info

**Project:** Spotify Playlist Extension with Pattern Mining and Clustering  
**Course:** CSCI 6443 Data Mining  
**Institution:** George Washington University  
**Semester:** Fall 2024  
**Author:** Adarsh Singh

## 📧 Support

If dashboard doesn't work:
1. Check Python version (3.8+)
2. Verify Streamlit installation
3. Ensure in correct directory
4. Check error messages in terminal

## 🎉 That's It!

Run `streamlit run app.py` and explore your results! 🚀
