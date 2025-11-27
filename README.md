# Spotify Playlist Extension with Pattern Mining and Clustering

**Course:** CSCI 6443 Data Mining  
**Institution:** George Washington University  
**Semester:** Fall 2025  
**Author:** Adarsh Singh [G39508544]

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/drsh0755/spotify-playlist-mining)
[![Python](https://img.shields.io/badge/Python-3.13+-green)](https://www.python.org/)

---

## 🎯 Project Overview

This project addresses the **automatic playlist continuation** challenge using advanced data mining techniques on the Spotify Million Playlist Dataset. By analyzing 1 million playlists containing 2.3 million unique tracks, we developed a hybrid recommendation system that achieves **89x improvement** over a popularity baseline.

### Key Achievements

- ✅ **89x improvement** in recommendation accuracy (R-precision: 13.3% vs baseline: 0.15%)
- ✅ Analyzed **66 million playlist-track entries** from 1M playlists
- ✅ Discovered **13,000 meaningful association rules** using pattern mining
- ✅ Identified **5 distinct playlist clusters** through K-means clustering
- ✅ Built **interactive dashboard** with 100% real data integration
- ✅ Generated **17 publication-quality visualizations**
- ✅ Implemented **7 ML models** (clustering, rules, matrix factorization, neural networks)

---

## 📚 Research Questions

### RQ1: Track Co-occurrence Patterns
**"How often do songs co-occur or co-disappear in playlists, and how can this knowledge inform recommendations?"**

**Method:** Association rule mining with Apriori algorithm  
**Result:** 13,000 high-confidence rules (lift >1.2, confidence >0.10)  
**Key Finding:** Strong genre-based co-occurrence patterns enable accurate track predictions

### RQ2: Playlist Clustering
**"Can playlists and tracks be effectively clustered by genre or other features to improve recommendation relevance?"**

**Method:** K-means clustering on 2.26M tracks using popularity, position, and artist features  
**Result:** 5 optimal clusters with distinct characteristics  
**Key Finding:** Cluster-aware recommendations improve relevance by 40%

### RQ3: Metadata Influence on Quality
**"How does playlist metadata (titles, partial track seeds) influence recommendation quality?"**

**Method:** Hybrid ensemble combining co-occurrence, SVD, and neural embeddings  
**Result:** R-precision: 13.3%, NDCG: 1.0  
**Key Finding:** Metadata significantly boosts cold-start performance

---

## 🏗️ Project Structure
```
spotify-playlist-mining/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── data/                              # Data directory (gitignored)
│   ├── raw/                          # Original MPD data (35 GB)
│   └── processed/                    # Processed data (3 GB)
│       ├── tracks_full_mpd.parquet   # 66M track entries (4.8 GB)
│       ├── playlists_full_mpd.parquet # 1M playlists (20 MB)
│       ├── association_rules_full.csv # 13K rules (1.3 MB)
│       ├── track_clusters_full.csv    # 2.26M clustered tracks (160 MB)
│       └── models/                    # Trained models (62 MB)
│
├── scripts/                           # 42 processing scripts
│   ├── 01-24: Phase 1 (Data loading)
│   ├── 25-30: Phase 2 (Experiments)
│   ├── 32-35: Phase 3 (Advanced models)
│   └── 41-42: Phase 4 (Visualizations)
│
├── dashboard/                         # Interactive Streamlit dashboard
│   ├── app.py                        # Home page
│   └── pages/                        # 7 interactive pages
│       ├── 1_Overview.py             # Dataset statistics
│       ├── 2_Model_Performance.py    # 89x improvement display
│       ├── 3_Recommendations.py      # Live recommendation demo
│       ├── 4_Clusters.py             # Clustering visualizations
│       ├── 5_Association_Rules.py    # Co-occurrence patterns
│       ├── 6_Advanced_Analytics.py   # Phase 3 models
│       └── 7_Timeline.py             # Project timeline
│
├── outputs/                           # Results and figures
│   └── figures/                      # 17 publication-quality figures
│       ├── presentation/             # For slides (150 DPI)
│       └── publication/              # For papers (300 DPI)
│
└── docs/                              # Detailed documentation
    ├── README.md                      # Documentation index
    ├── DEVELOPMENT_JOURNEY.md         # Development timeline & pivots
    ├── SCRIPTS_REFERENCE.md           # Complete script documentation
    └── PROJECT_SETUP_GUIDE.md         # Installation guide
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.13+** (tested on 3.13)
- **35 GB free disk space** (for raw data)
- **16 GB RAM minimum** (32 GB recommended)
- **macOS, Linux, or Windows**

### Installation
```bash
# 1. Clone repository
git clone https://github.com/drsh0755/spotify-playlist-mining.git
cd spotify-playlist-mining

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Download Spotify Million Playlist Dataset
# Visit: https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge
# Extract to: data/raw/mpd_slices/

# 5. Verify setup
python scripts/01_verify_data.py
```

### Run the Project
```bash
# Process data (one-time, ~90 minutes)
python scripts/24_phase1_master_pipeline.py  # 22 minutes
python scripts/31_phase2_master_pipeline.py  # 65 minutes

# Launch interactive dashboard
cd dashboard
streamlit run app.py
# Opens at http://localhost:8501
```

---

## 📊 Key Results

### Model Performance

| Model | R-Precision | NDCG | Improvement |
|-------|-------------|------|-------------|
| Popularity Baseline | 0.0015 (0.15%) | 0.05 | 1x (baseline) |
| Co-occurrence | 0.05 (5%) | 0.15 | 33x |
| SVD Factorization | 0.08 (8%) | 0.22 | 53x |
| **Hybrid Ensemble** | **0.133 (13.3%)** | **1.0** | **89x** ⭐ |

### Dataset Statistics

- **Playlists:** 1,000,000
- **Unique tracks:** 2,262,292
- **Unique artists:** 295,860
- **Track entries:** 66,346,428
- **Average playlist length:** 66 tracks
- **Association rules:** 13,000 (high-quality)
- **Clusters:** 5 distinct groups

### Computational Performance

- **Total runtime:** ~90 minutes (Phases 1-2)
- **Phase 1 (Data loading):** 22 minutes
- **Phase 2 (Experiments):** 65 minutes
- **Memory usage:** Peak 12 GB RAM
- **Platform:** M4 MacBook Air (32GB RAM)

---

## 🔬 Technical Approach

### Phase 1: Data Processing (Scripts 01-24)
- Load and validate 1M playlists
- Extract 66M track entries
- Build feature matrices
- Create co-occurrence matrix (10K×10K)

### Phase 2: Core Experiments (Scripts 25-30)
- **Association Rules:** Apriori algorithm (13K rules)
- **Clustering:** K-means on 2.26M tracks (5 clusters)
- **Recommendations:** Co-occurrence, SVD, hybrid models
- **Evaluation:** R-precision, NDCG on 5K playlists

### Phase 3: Advanced Models (Scripts 32-35)
- **Matrix Factorization:** SVD (50 factors) and ALS
- **Neural Networks:** Track embeddings (32 dimensions)
- **Predictive Models:** Classification (99.6% accuracy)
- **Hybrid Ensemble:** Optimal weight tuning

### Phase 4: Visualization & Dashboard (Scripts 41-42)
- Generate 17 publication-quality figures
- Build interactive Streamlit dashboard
- Integrate all Phase 1-3 results

---

## 🎨 Interactive Dashboard

7-page interactive web application showcasing all results:

1. **Overview** - Dataset statistics with 66M track entries
2. **Model Performance** - 89x improvement visualization
3. **Recommendations** - Live recommendation demo
4. **Clusters** - K-means clustering visualization
5. **Association Rules** - Browse 13K co-occurrence patterns
6. **Advanced Analytics** - SVD factors, neural embeddings
7. **Timeline** - Project development journey

**Launch:**
```bash
cd dashboard
streamlit run app.py
```

---

## 📈 Visualizations

17 publication-ready figures in `outputs/figures/`:

**Dataset & Patterns:**
- Dataset overview dashboard
- Co-occurrence heatmap
- Association rules network

**Clustering:**
- PCA projection (5 clusters)
- Cluster characteristics
- Silhouette analysis

**Recommendations:**
- Model performance comparison ⭐
- Category-wise performance
- Diversity analysis

**Advanced:**
- SVD latent factors
- Neural embeddings (t-SNE)
- Feature importance

**Meta:**
- Project timeline
- Computational performance

---

## 🛠️ Technology Stack

### Core Libraries
- **Data Processing:** pandas, numpy, scipy
- **Machine Learning:** scikit-learn, implicit
- **Pattern Mining:** mlxtend (FP-Growth)
- **Visualization:** matplotlib, seaborn, plotly
- **Dashboard:** streamlit

### Advanced Components
- **Matrix Factorization:** Truncated SVD, ALS
- **Neural Networks:** sklearn neural networks, PCA embeddings
- **Sparse Matrices:** scipy.sparse (memory-efficient)
- **Clustering:** K-means with MiniBatch for scale

---

## 📖 Documentation

### Quick Links
- **[Development Journey](docs/DEVELOPMENT_JOURNEY.md)** - Complete development timeline with pivots and decisions
- **[Scripts Reference](docs/SCRIPTS_REFERENCE.md)** - Detailed documentation of all 42 scripts
- **[Setup Guide](docs/PROJECT_SETUP_GUIDE.md)** - Installation and configuration
- **[Figure Manifest](outputs/figures/FIGURE_MANIFEST.md)** - Catalog of all visualizations

### Key Documents
- `README.md` (this file) - Project overview
- `docs/DEVELOPMENT_JOURNEY.md` - Development story, pivots, learnings
- `docs/SCRIPTS_REFERENCE.md` - Complete script reference
- `dashboard/README.md` - Dashboard usage guide

---

## 🎓 Academic Context

### Course Information
- **Course:** CSCI 6443 Data Mining
- **Instructor:** George Washington University Faculty
- **Semester:** Fall 2025
- **Student:** Adarsh Singh [G39508544]

### Project Requirements Met
✅ Pattern mining (association rules)  
✅ Clustering (K-means)  
✅ Classification/Regression (predictive models)  
✅ Large-scale dataset (1M+ instances)  
✅ Reproducible methodology  
✅ Publication-quality results  
✅ Interactive demonstration

---

## 🏆 Key Innovations

### 1. Hybrid Ensemble Approach
Combines three complementary methods:
- Co-occurrence patterns (40% weight)
- Matrix factorization (30% weight)
- Neural embeddings (30% weight)

### 2. Cluster-Aware Recommendations
Uses playlist clustering to improve cold-start performance by 40%

### 3. Efficient Sparse Matrix Processing
Handles 10K×10K co-occurrence matrix in memory-efficient format

### 4. 100% Real Data Dashboard
All dashboard pages use actual experimental results (no simulated data)

---

## 📊 Reproducibility

### System Requirements
- **CPU:** 4+ cores recommended
- **RAM:** 16 GB minimum, 32 GB recommended
- **Storage:** 50 GB free space
- **OS:** macOS, Linux, or Windows
- **Python:** 3.13+ (tested on 3.13)

### Expected Runtime
- **Phase 1:** 20-25 minutes
- **Phase 2:** 60-70 minutes
- **Phase 3:** 120-180 minutes (optional)
- **Total:** ~90 minutes (Phases 1-2)

### Reproducibility Checklist
- [ ] Python 3.13+ installed
- [ ] 50 GB free disk space
- [ ] Spotify MPD dataset downloaded
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Scripts run in order (01-24, then 25-30)
- [ ] Results verified in `data/processed/`

---

## 🚧 Troubleshooting

### Common Issues

**Issue:** `MemoryError` during processing  
**Solution:** Close other applications, use 16+ GB RAM system

**Issue:** `FileNotFoundError` for dataset  
**Solution:** Ensure MPD data is in `data/raw/mpd_slices/`

**Issue:** Dashboard won't start  
**Solution:** `pip install streamlit plotly`, ensure in `dashboard/` directory

**Issue:** Scripts run slowly  
**Solution:** Use SSD, close background apps, check system resources

### Getting Help
- Check [Scripts Reference](docs/SCRIPTS_REFERENCE.md) for script-specific issues
- Review [Development Journey](docs/DEVELOPMENT_JOURNEY.md) for context
- Check GitHub Issues for known problems

---

## 🔮 Future Work

### Potential Extensions
- **Graph Neural Networks** for playlist-track relationships
- **Temporal Analysis** of music trends over time
- **Audio Features** integration (when available)
- **Multi-Objective Optimization** for diversity vs relevance
- **Real-Time System** for live recommendations
- **A/B Testing Framework** for model comparison

### Research Directions
- Cold-start problem for new tracks/artists
- Session-based recommendations
- Contextual recommendations (time, mood, activity)
- Cross-dataset generalization

---

## 📄 Citation

If you use this work, please cite:
```bibtex
@misc{singh2025spotify,
  title={Spotify Playlist Extension with Pattern Mining and Clustering},
  author={Singh, Adarsh},
  year={2025},
  school={George Washington University},
  note={CSCI 6443 Data Mining Final Project}
}
```

---

## 📜 License

This project is licensed under the MIT License - see LICENSE file for details.

### Dataset License
The Spotify Million Playlist Dataset is provided by Spotify for research purposes under the [RecSys Challenge 2018 terms](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge).

---

## 🙏 Acknowledgments

- **Spotify** for the Million Playlist Dataset
- **RecSys Challenge 2018** for the evaluation framework
- **George Washington University** for academic support
- **Open-source community** for the excellent Python libraries

---

## 📞 Contact

**Adarsh Singh**  
MS in Data Science  
George Washington University  

**Repository:** https://github.com/drsh0755/spotify-playlist-mining

---

## 🎯 Project Status

- [x] Phase 1: Data Processing (Complete)
- [x] Phase 2: Core Experiments (Complete)
- [x] Phase 3: Advanced Models (Complete)
- [x] Phase 4: Visualization & Dashboard (Complete)
- [x] Documentation (Complete)
- [ ] Optional: A/B Testing Framework
- [ ] Optional: Real-Time Deployment

**Status:** ✅ Complete and ready for presentation/submission

**Last Updated:** November 26, 2025

---

*For detailed technical documentation, see [docs/](docs/) directory.*