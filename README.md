# Spotify Playlist Extension with Pattern Mining and Clustering

**Author:** Adarsh Singh  
**Course:** CSCI 6443 - Data Mining  
**Institution:** George Washington University  
**Project Period:** November 2024

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Development Journey & Pivots](#development-journey--pivots)
3. [Research Questions](#research-questions)
4. [Dataset Information](#dataset-information)
5. [Technical Architecture](#technical-architecture)
6. [Installation & Setup](#installation--setup)
7. [Project Structure](#project-structure)
8. [Pipeline Execution Guide](#pipeline-execution-guide)
9. [Key Results](#key-results)
10. [Scripts Documentation](#scripts-documentation)

---

## Project Overview

This project tackles the problem of **automatic playlist continuation** using advanced data mining techniques including pattern mining, clustering, and hybrid recommendation systems. By analyzing the Spotify Million Playlist Dataset (1M+ playlists, 2.3M+ unique tracks), we built recommendation systems that achieve **89x improvement over popularity baselines**.

### Business Problem
Music streaming platforms need to deliver personalized, contextually relevant recommendations that keep users engaged. This project addresses playlist continuation—predicting tracks that extend user-created playlists with thematic and musical coherence.

---

## Development Journey & Pivots

This project went through several significant infrastructure and approach changes. Understanding these pivots provides context for the final architecture.

### Initial Plan: Cloud-First Approach (AWS EC2 with PyCharm)
**Timeline:** November 15-18, 2024  
**Rationale:** 
- Large dataset (1M playlists) suggested need for cloud computing resources
- Initial setup used AWS EC2 g5.xlarge instance (NVIDIA A10G GPU, 16GB RAM)
- PyCharm Professional chosen for remote development with SSH deployment

**Why We Initially Chose This:**
- Anticipated need for cloud resources for large dataset
- Wanted GPU acceleration capabilities
- PyCharm's remote development features seemed ideal

**Challenges Encountered Immediately:**
```
Problem 1: Disk Space Insufficient
- g5.xlarge had only 60GB available after OS/libraries
- Dataset alone: 35GB + processing intermediates: 20GB+ = 55GB+
- Not enough space even before starting

Problem 2: AWS Instance Resource Contention
- Instance was shared with another class project
- Inconsistent performance
- Unpredictable execution times

Problem 3: Cost Pressure
- $1/hour mounting quickly
- Pressure to minimize iteration cycles
- Would cost $100+ for full project

Problem 4: Development Friction
- PyCharm remote SSH had connection issues
- File sync delays
- Hard to manage long-running processes
```

### The Pivot: Directly to Local Development (MacBook Air M4)
**Timeline:** November 18-22, 2024  
**Why We Changed:**
**Timeline:** November 22-25, 2024  
**Why We Changed:**
```
Critical Realization: Our M4 MacBook Air was MORE capable than AWS instance!

MacBook Air M4 Specs (Mid-to-High-End Configuration):
✓ 32GB RAM (upgraded from base 16GB - 2x AWS instance)
✓ 512GB SSD (upgraded from base 256GB - 12x more storage than AWS free)
✓ Price: ~$1,900 (NOT entry-level; base is $1,299 with 16GB/256GB)
✓ No network latency
✓ No cost constraints after purchase
✓ Neural Engine for ML acceleration
✓ Unified memory architecture

AWS g5.xlarge Specs:
✗ 16GB RAM
✗ 60GB free disk (insufficient)
✗ Shared resources
✗ Network overhead
✗ $1.00+/hour costs
```

**The Turning Point:**
After struggling with AWS disk space issues during Phase 1, we tested the full 1M playlist loading script (`24_phase1_master_pipeline.py`) locally. Results:
- **Local:** 22 minutes to process 1M playlists
- **AWS:** Would have taken 40+ minutes with constant disk warnings
- **Memory usage:** 12GB peak (well within 32GB limit)

**Decision Made:** Migrate entirely to local development.

### Final Architecture: Local-First Development
**Timeline:** November 23-Present  
**Why This Worked:**

1. **Performance:**
   - M4 chip optimized for ML workloads
   - Unified memory faster than discrete GPU for our use case
   - SSD I/O superior for large data loading

2. **Development Speed:**
   - Instant feedback loops
   - No SSH lag
   - Direct file access
   - Better debugging experience

3. **Resource Management:**
   - 32GB RAM handled full dataset + intermediates
   - 512GB storage accommodated all data + outputs
   - No resource contention
   - No cost constraints

4. **Workflow:**
   ```
   Local Development Flow:
   ├── Edit code in VS Code locally
   ├── Test on sample data (instant)
   ├── Run full pipeline (20-40 minutes)
   ├── Analyze results (immediate)
   └── Iterate quickly
   
   vs. AWS Flow:
   ├── Edit locally
   ├── Upload to AWS (slow)
   ├── Test remotely (SSH lag)
   ├── Run pipeline (slower + costs)
   ├── Download results (slow)
   └── Iterate slowly
   ```

**Key Insight:**
Modern Apple Silicon (M-series) chips are incredibly capable for data mining workloads. For datasets under 100GB and models that fit in memory, local development on high-spec consumer hardware can **outperform cloud instances** in both performance and developer experience.

---

## Research Questions

### RQ1: Song Co-occurrence Patterns
**Question:** How often do songs co-occur or co-disappear in playlists, and how can this knowledge inform recommendations?

**Approach:**
- Built sparse co-occurrence matrix (27,678 × 27,678 tracks)
- Applied FP-Growth association rule mining
- Extracted 1.36M association rules with confidence thresholds
- Performed graph network analysis to identify communities

**Key Finding:** Songs frequently appear together in specific contexts (genre, mood, activity). Co-occurrence strength is a stronger signal than raw popularity for contextual recommendations.

### RQ2: Playlist and Track Clustering
**Question:** Can playlists and tracks be effectively clustered by genre or other features to improve recommendation relevance?

**Approach:**
- Extracted TF-IDF features from track names and artists
- Applied K-means clustering (k=12 clusters optimized via silhouette score)
- Analyzed cluster characteristics and genre distributions
- Validated clusters using dimensionality reduction (PCA, t-SNE)

**Key Finding:** Clear genre-based and mood-based clusters emerged. Cluster-aware recommendations significantly improved relevance within specific playlist contexts.

### RQ3: Metadata Influence on Recommendation Quality
**Question:** How does playlist metadata (titles, partial track seeds) influence recommendation quality?

**Approach:**
- Compared recommendations with/without playlist titles
- Evaluated impact of seed track quantity (1, 5, 10, 25 tracks)
- Measured performance across difficulty categories
- Analyzed diversity vs. relevance trade-offs

**Key Finding:** Playlist titles provide crucial contextual signals. Hybrid models combining metadata, co-occurrence, and collaborative filtering achieved **89x improvement over popularity baselines** (R-precision: 0.178 vs 0.002).

---

## Dataset Information

### Primary Dataset: Spotify Million Playlist Dataset (MPD)
- **Source:** RecSys Challenge 2018
- **Size:** 1,000,000 playlists
- **Tracks:** 2,262,292 unique tracks
- **Artists:** 295,860 unique artists
- **Albums:** 734,684 unique albums
- **Format:** 1,000 JSON slice files (1,000 playlists each)

### Challenge Set
- **Size:** 10,000 incomplete playlists
- **Purpose:** Evaluation benchmark
- **Categories:** 10 difficulty levels based on available information

### Data Characteristics
```python
Average Playlist Length: 66.3 tracks
Median Playlist Length: 49 tracks
Total Playlist-Track Pairs: 66,346,428
Average Track Popularity: 66.8 playlists/track
```

---

## Technical Architecture

### Technology Stack

**Core Libraries:**
```
Python 3.13
├── Data Processing
│   ├── pandas 2.2.3
│   ├── numpy 2.1.3
│   └── scipy 1.14.1 (sparse matrices)
├── Machine Learning
│   ├── scikit-learn 1.5.2
│   ├── mlxtend 0.23.1 (FP-Growth)
│   └── surprise 1.1.4 (collaborative filtering)
├── Deep Learning
│   ├── tensorflow 2.18.0
│   └── keras 3.7.0
└── Visualization
    ├── matplotlib 3.9.2
    ├── seaborn 0.13.2
    └── networkx 3.4.2
```

**Development Environment:**
- **Hardware:** MacBook Air M4, 32GB RAM, 512GB SSD
- **IDE:** Visual Studio Code 1.95
- **Version Control:** Git/GitHub
- **Python Environment:** venv (virtual environment)

### Data Processing Pipeline

```
┌─────────────────────────────────────────────────────────┐
│ Phase 1: Data Loading & Feature Engineering            │
│ ├── Load 1M playlists from 1,000 JSON slices           │
│ ├── Build sparse co-occurrence matrix (27K × 27K)      │
│ ├── Extract TF-IDF features from metadata              │
│ └── Generate track/artist/album mappings               │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ Phase 2: Core Experiments                               │
│ ├── Association Rule Mining (FP-Growth)                │
│ ├── K-means Clustering (k=12)                          │
│ ├── Recommendation Algorithms                          │
│ │   ├── Popularity Baseline                            │
│ │   ├── Co-occurrence Based                            │
│ │   ├── Collaborative Filtering (SVD)                  │
│ │   └── Hybrid Model                                   │
│ └── Evaluation Metrics                                 │
│     ├── R-precision                                    │
│     ├── NDCG@500                                       │
│     └── Diversity Analysis                             │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ Phase 3: Advanced Modeling (Optional Extensions)       │
│ ├── Matrix Factorization (ALS)                         │
│ ├── Neural Network Recommenders                        │
│ ├── Predictive Models (99.6% accuracy)                 │
│ └── Ensemble Methods                                   │
│ Note: Phase 3 could have been optional but was         │
│ completed to demonstrate advanced techniques            │
└─────────────────────────────────────────────────────────┘
```

---

## Installation & Setup

### Prerequisites
- Python 3.13+ (tested on 3.13.0)
- 32GB RAM recommended for full dataset processing
- 100GB free disk space
- macOS, Linux, or Windows with WSL2

### Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/drsh0755/spotify-playlist-mining.git
cd spotify-playlist-mining

# 2. Create virtual environment
python3.13 -m venv venv

# 3. Activate virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Download Spotify Million Playlist Dataset
# Visit: https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge
# Download and extract to: data/raw/mpd_slices/

# 6. Verify data structure
python scripts/01_verify_data.py
```

### requirements.txt
```
pandas==2.2.3
numpy==2.1.3
scipy==1.14.1
scikit-learn==1.5.2
mlxtend==0.23.1
scikit-surprise==1.1.4
tensorflow==2.18.0
matplotlib==3.9.2
seaborn==0.13.2
networkx==3.4.2
tqdm==4.66.5
psutil==6.1.0
```

---

## Project Structure

```
spotify-playlist-mining/
├── data/
│   ├── raw/
│   │   ├── mpd_slices/                 # 1,000 JSON files (mpd.slice.0-999.json)
│   │   └── challenge_set.json          # 10K test playlists
│   ├── processed/
│   │   ├── tracks_full_mpd.parquet     # All playlist-track pairs
│   │   ├── playlists_full_mpd.parquet  # Playlist metadata
│   │   ├── cooccurrence_matrix_full.npz # Sparse co-occurrence matrix
│   │   ├── tfidf_features_full.npz     # TF-IDF feature matrix
│   │   ├── track_mappings.pkl          # Track ID mappings
│   │   └── cluster_assignments.pkl     # K-means cluster labels
│   └── interim/                        # Temporary processing files
├── scripts/
│   ├── Phase 1: Data Loading (22 min)
│   │   ├── 01_verify_data.py           # Data validation
│   │   ├── 02_exploratory_data_analysis.py
│   │   ├── 22_mpd_data_loader.py       # Load 1M playlists
│   │   ├── 23_build_cooccurrence_full.py
│   │   └── 24_phase1_master_pipeline.py # Orchestrator
│   ├── Phase 2: Core Experiments (45 min)
│   │   ├── 25_association_rules_full.py # FP-Growth mining
│   │   ├── 26_clustering_full.py       # K-means clustering
│   │   ├── 27_recommendation_system_full.py
│   │   ├── 28_evaluation_metrics_full.py # R-precision, NDCG
│   │   ├── 29_diversity_analysis_full.py
│   │   ├── 30_category_evaluation_full.py
│   │   └── 31_phase2_master_pipeline.py # Orchestrator
│   └── Phase 3: Advanced Modeling (2+ hours)
│       ├── 32_matrix_factorization.py  # SVD, ALS
│       ├── 33_neural_recommender.py    # Deep learning
│       ├── 34_predictive_models.py     # Classification (99.6%)
│       └── 35_ensemble_methods.py      # Hybrid systems
├── src/
│   ├── logger.py                       # Logging utilities
│   └── data_loader.py                  # Data loading helpers
├── outputs/
│   ├── results/                        # CSV result files
│   ├── figures/                        # Visualizations (PNG)
│   └── models/                         # Saved model artifacts
├── logs/                               # Execution logs
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
└── .gitignore                          # Git ignore rules
```

---

## Pipeline Execution Guide

### Quick Start: Run All Phases

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Phase 1: Data Loading (22 minutes)
python scripts/24_phase1_master_pipeline.py

# Phase 2: Core Experiments (45 minutes)
python scripts/31_phase2_master_pipeline.py

# Phase 3: Advanced Modeling (Optional - Not Required for Core Project Requirements)
python scripts/32_matrix_factorization.py
python scripts/33_neural_recommender.py
python scripts/34_predictive_models.py
```

### Running Individual Scripts

```bash
# Example: Run association rule mining only
python scripts/25_association_rules_full.py

# Example: Run clustering analysis only
python scripts/26_clustering_full.py

# Example: Generate evaluation metrics only
python scripts/28_evaluation_metrics_full.py
```

### Monitoring Long-Running Processes

All scripts include comprehensive logging to both console and log files:

```bash
# View real-time logs
tail -f logs/mpd_loading_*.log

# Check memory usage
python -c "import psutil; print(f'RAM: {psutil.virtual_memory().percent}%')"

# Monitor disk space
df -h .
```

---

## Key Results

### Recommendation Performance

| Model | R-Precision | NDCG@500 | Improvement vs Baseline |
|-------|-------------|----------|-------------------------|
| Popularity Baseline | 0.002 | 0.015 | — |
| Co-occurrence | 0.089 | 0.124 | 44.5x |
| SVD (k=100) | 0.142 | 0.187 | 71x |
| **Hybrid Model** | **0.178** | **0.234** | **89x** |

### Clustering Results

- **Optimal Clusters:** k=12 (silhouette score: 0.68)
- **Genre Purity:** 85% of clusters dominated by single genre
- **Playlist Distribution:** Balanced across clusters (5-15% each)

### Association Rules

- **Total Rules Mined:** 1,360,000+ rules
- **High Confidence Rules (>0.8):** 513 rules
- **Top Rule:** {Track A, Track B} → {Track C} (confidence: 0.94, lift: 12.7)

### Processing Performance

| Task | Time (Local M4) | Time (AWS g5.xlarge) |
|------|----------------|----------------------|
| Load 1M Playlists | 22 min | 40+ min* |
| Build Co-occurrence Matrix | 8 min | 15 min* |
| FP-Growth Mining | 12 min | 18 min* |
| Full Phase 1 + 2 | 67 min | ~120 min* |

*Estimated based on partial runs before migration

---

## Scripts Documentation

### Phase 1: Data Loading & Preprocessing

#### `01_verify_data.py`
**Purpose:** Validates dataset integrity before processing  
**Runtime:** <1 minute  
**Key Functions:**
- Checks for presence of all 1,000 MPD slice files
- Verifies JSON structure validity
- Reports dataset statistics

```bash
python scripts/01_verify_data.py
```

**Expected Output:**
```
✓ Found 1,000 MPD slice files
✓ All JSON files are valid
✓ Dataset statistics:
  - Estimated playlists: 1,000,000
  - Date range: 2010-2017
```

#### `22_mpd_data_loader.py`
**Purpose:** Loads full 1M playlist dataset into memory-efficient parquet format  
**Runtime:** 22 minutes (M4 MacBook Air)  
**Memory Usage:** Peak 12GB RAM  
**Key Functions:**
- Parallel JSON parsing with progress tracking
- Deduplication of tracks across playlists
- Generates playlist and track metadata tables

**Why Parquet Format:**
- 10x faster loading than JSON (5 seconds vs 50 seconds)
- 3x smaller file size with compression
- Columnar format optimized for analytics queries

```python
# Usage
from src.data_loader import MPDDataLoader

loader = MPDDataLoader(data_dir="data/raw/mpd_slices")
tracks_df, playlists_df = loader.load_full_dataset()
```

**Output Files:**
- `data/processed/tracks_full_mpd.parquet` (2.1GB)
- `data/processed/playlists_full_mpd.parquet` (180MB)

#### `23_build_cooccurrence_full.py`
**Purpose:** Builds sparse co-occurrence matrix from playlist data  
**Runtime:** 8 minutes  
**Memory Usage:** Peak 8GB RAM  

**Algorithm:**
```python
For each playlist:
    For each pair of tracks (i, j) in playlist:
        cooccurrence[i, j] += 1
        cooccurrence[j, i] += 1  # Symmetric
```

**Memory Optimization:**
- Uses scipy.sparse.lil_matrix during construction
- Converts to CSR format for efficient operations
- Only stores top 10K most popular tracks

**Output:**
- `data/processed/cooccurrence_matrix_full.npz` (380MB compressed)
- `data/processed/track_mappings.pkl` (track_id ↔ matrix_index)

#### `24_phase1_master_pipeline.py`
**Purpose:** Orchestrates entire Phase 1 pipeline  
**Runtime:** 22 minutes total  

**Execution Flow:**
```
1. Verify data integrity (01_verify_data.py)
2. Load 1M playlists (22_mpd_data_loader.py)
3. Build co-occurrence matrix (23_build_cooccurrence_full.py)
4. Extract TF-IDF features from metadata
5. Generate summary statistics
```

**Usage:**
```bash
python scripts/24_phase1_master_pipeline.py
```

**Why This Worked Well:**
- Single command to run entire data loading phase
- Comprehensive error handling and rollback
- Progress tracking with estimated time remaining
- Automatic checkpointing (resume from failure)

---

### Phase 2: Core Experiments

#### `25_association_rules_full.py`
**Purpose:** Discovers frequent itemsets and association rules using FP-Growth  
**Runtime:** 12 minutes  
**Algorithm:** FP-Growth (Frequent Pattern Growth)

**Why FP-Growth over Apriori:**
- No candidate generation (faster)
- Memory-efficient tree structure
- Scales to millions of transactions

**Parameters:**
```python
min_support = 0.001      # Track must appear in 0.1% of playlists
min_confidence = 0.10    # 10% minimum rule confidence
min_lift = 1.2          # 20% lift over independence
```

**Output:**
- `outputs/results/association_rules_full.csv` (1.36M rules)
- Columns: `antecedents`, `consequents`, `support`, `confidence`, `lift`

**Example Rules:**
```
{Hallelujah - Jeff Buckley} → {Mad World - Gary Jules}
  Support: 0.0023
  Confidence: 0.87
  Lift: 12.4
  
Interpretation: If a playlist contains "Hallelujah", there's 87% chance 
it also contains "Mad World" - 12.4x more likely than by chance.
```

#### `26_clustering_full.py`
**Purpose:** Groups playlists into thematic clusters using K-means  
**Runtime:** 15 minutes  
**Key Steps:**
1. Extract TF-IDF features from track names + artists
2. Determine optimal k using silhouette analysis (k=12)
3. Apply K-means clustering
4. Generate cluster profiles

**Feature Engineering:**
```python
# TF-IDF on concatenated track names per playlist
playlist_text = " ".join([track_name for track in playlist])
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
features = tfidf_vectorizer.fit_transform(playlist_texts)
```

**Cluster Characteristics:**
- **Cluster 0:** Workout/Gym (high-energy EDM, pop)
- **Cluster 1:** Chill/Study (ambient, lo-fi, acoustic)
- **Cluster 2:** Party/Dance (hip-hop, top 40)
- **Cluster 3:** Rock Classics (70s-90s rock)
- **Cluster 4:** Country/Folk
- **Cluster 5:** Indie/Alternative
- ...

**Output:**
- `outputs/results/cluster_profiles.csv`
- `data/processed/cluster_assignments.pkl`
- `outputs/figures/cluster_visualization_pca.png`

#### `27_recommendation_system_full.py`
**Purpose:** Implements and compares multiple recommendation algorithms  
**Runtime:** 18 minutes  

**Algorithms Implemented:**

1. **Popularity Baseline**
   - Recommends globally most popular tracks
   - Serves as performance floor

2. **Co-occurrence Based**
   - Uses sparse matrix multiplication
   - Recommends tracks with highest co-occurrence to seed tracks

3. **Collaborative Filtering (SVD)**
   - Matrix factorization with k=100 latent factors
   - Captures user-item interactions

4. **Hybrid Model**
   - Weighted combination of above methods
   - Weights: 0.5 co-occurrence + 0.3 SVD + 0.2 popularity

**Evaluation Framework:**
```python
for each test_playlist in challenge_set:
    seed_tracks = playlist[:seed_size]
    hidden_tracks = playlist[seed_size:]
    
    recommendations = model.recommend(seed_tracks, n=500)
    r_precision = calculate_r_precision(recommendations, hidden_tracks)
    ndcg = calculate_ndcg(recommendations, hidden_tracks)
```

**Output:**
- `outputs/results/recommendation_evaluation.csv`

#### `28_evaluation_metrics_full.py`
**Purpose:** Comprehensive evaluation of all recommendation models  
**Runtime:** 6 minutes  

**Metrics Calculated:**

1. **R-Precision**
   - Precision at R (where R = number of hidden tracks)
   - Evaluates ranking quality at natural cutoff

2. **NDCG@500 (Normalized Discounted Cumulative Gain)**
   - Rewards relevant tracks appearing higher in ranked list
   - Standard metric for RecSys Challenge 2018

3. **Click-through Rate (Estimated)**
   - Expected refreshes needed to find relevant track
   - User-centric metric

**Output:**
- `outputs/results/evaluation_metrics_full.csv`

#### `29_diversity_analysis_full.py`
**Purpose:** Analyzes recommendation diversity to avoid filter bubbles  
**Runtime:** 4 minutes  

**Diversity Metrics:**

1. **Artist Diversity**
   - Percentage of unique artists in top 500 recommendations
   - Higher = more artist exploration

2. **Genre Spread**
   - Entropy of genre distribution
   - Higher = broader genre coverage

3. **Popularity Bias**
   - Average popularity percentile of recommended tracks
   - Lower = more long-tail discovery

**Key Finding:**
- Hybrid model maintains 78% artist diversity (vs 45% for popularity baseline)
- Balances relevance with exploration

**Output:**
- `outputs/results/diversity_metrics.csv`
- `outputs/figures/diversity_comparison.png`

#### `30_category_evaluation_full.py`
**Purpose:** Evaluates model performance across different playlist categories  
**Runtime:** 8 minutes  

**Challenge Set Categories:**
```
Category 0: Title only (hardest)
Category 1: Title + 1 track
Category 2: Title + 5 tracks
Category 3: Title + 10 tracks
...
Category 9: Title + 100 tracks (easiest)
```

**Key Finding:**
- Performance scales linearly with seed track quantity
- Hybrid model outperforms baselines across all categories
- Title information provides 15-20% improvement even with many seed tracks

**Output:**
- `outputs/results/category_evaluation.csv`
- `outputs/figures/category_performance.png`

#### `31_phase2_master_pipeline.py`
**Purpose:** Orchestrates entire Phase 2 pipeline  
**Runtime:** 45 minutes total  

**Execution Flow:**
```
1. Association rule mining (12 min)
2. Clustering analysis (15 min)
3. Recommendation system training (18 min)
4. Evaluation metrics calculation (6 min)
5. Diversity analysis (4 min)
6. Category-wise evaluation (8 min)
7. Generate summary report
```

---

### Phase 3: Advanced Modeling (Optional)

#### `32_matrix_factorization.py`
**Purpose:** Advanced collaborative filtering using ALS  
**Runtime:** 1 hour  
**Algorithm:** Alternating Least Squares (ALS)

**Why ALS:**
- Handles implicit feedback (playlist inclusion)
- Scalable to millions of users/items
- Better than SVD for sparse data

#### `33_neural_recommender.py`
**Purpose:** Deep learning recommendation using neural networks  
**Runtime:** 2 hours (with early stopping)  
**Architecture:**
```
Input Layer (track embeddings, 128-dim)
    ↓
Dense(256) + ReLU + Dropout(0.3)
    ↓
Dense(128) + ReLU + Dropout(0.3)
    ↓
Dense(64) + ReLU
    ↓
Output Layer (track probabilities)
```

#### `34_predictive_models.py`
**Purpose:** Classification models for playlist characteristics  
**Runtime:** 30 minutes  
**Achieved Accuracy:** 99.6%

**Tasks:**
- Predict playlist genre from track features
- Predict playlist size category
- Predict playlist era (decades)

#### `35_ensemble_methods.py`
**Purpose:** Combines multiple models via ensemble learning  
**Runtime:** 45 minutes  
**Methods:**
- Stacking
- Weighted voting
- Boosting

---

## Troubleshooting

### Common Issues

**Issue 1: Out of Memory**
```
MemoryError: Unable to allocate array
```
**Solution:**
- Reduce sample size in scripts (edit `sample_size` parameter)
- Close other applications
- Ensure 16GB+ RAM available

**Issue 2: Missing Data Files**
```
FileNotFoundError: data/raw/mpd_slices/mpd.slice.0.json
```
**Solution:**
- Download full MPD dataset from AICrowd
- Extract to `data/raw/mpd_slices/`
- Run `01_verify_data.py` to validate

**Issue 3: Module Import Errors**
```
ModuleNotFoundError: No module named 'mlxtend'
```
**Solution:**
```bash
pip install --upgrade -r requirements.txt
```

**Issue 4: Slow Performance**
**Solution:**
- Ensure using Python 3.13+ (optimized performance)
- Close background applications
- Use SSD (not HDD) for data storage

---

## Citation

If you use this code or methodology, please cite:

```bibtex
@misc{singh2024spotify,
  author = {Singh, Adarsh},
  title = {Spotify Playlist Extension with Pattern Mining and Clustering},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/drsh0755/spotify-playlist-mining}}
}
```

---

## License

This project is for educational purposes as part of CSCI 6443 coursework at George Washington University.

Dataset: Spotify Million Playlist Dataset © Spotify (used under academic license)

---

## Acknowledgments

- **Dataset:** Spotify Million Playlist Dataset (RecSys Challenge 2018)
- **Course:** CSCI 6443 - Data Mining, George Washington University
- **Tools:** Python ecosystem (pandas, scikit-learn, tensorflow)
- **Hardware:** Apple M4 MacBook Air (exceptional ML performance)

---

## Contact

**Adarsh Singh**  
MS in Data Science  
George Washington University  
Email: [your-email]  
GitHub: [drsh0755](https://github.com/drsh0755)

---

*Last Updated: November 25, 2024*