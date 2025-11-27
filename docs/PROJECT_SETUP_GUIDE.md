# Complete Project Setup Guide

**Quick reference for setting up the Spotify Playlist Extension project**

---

## 📁 Step 1: Documentation Structure

### Where to put documentation files:

```
spotify-playlist-mining/
├── README.md                          # ← Use comprehensive_README.md
├── requirements.txt                   # ← Python dependencies
├── requirements-minimal.txt           # ← Phases 1-2 only
├── requirements-dev.txt              # ← Dev tools
├── .gitignore
│
├── docs/                             # ← All detailed docs here
│   ├── README.md                     # Documentation index
│   ├── DEVELOPMENT_JOURNEY.md        # Development pivots
│   ├── SCRIPTS_REFERENCE.md          # Script documentation
│   └── DOCUMENTATION_UPDATES.md      # Changelog
│
├── data/                             # Data (add to .gitignore)
├── scripts/                          # Python scripts
├── src/                              # Reusable modules
├── outputs/                          # Results
└── logs/                             # Logs (add to .gitignore)
```

### Quick setup commands:

```bash
# Navigate to your project
cd ~/Desktop/CSCI\ 6443\ Data\ Mining\ -\ Project

# Create docs folder
mkdir -p docs

# You'll download these from Claude and place them:
# - comprehensive_README.md → rename to README.md (root)
# - DEVELOPMENT_JOURNEY.md → docs/
# - SCRIPTS_REFERENCE.md → docs/
# - DOCUMENTATION_UPDATES.md → docs/
# - requirements.txt → (root)
```

---

## 📦 Step 2: Requirements Files

### Three versions available:

#### **1. requirements.txt** (Full - Recommended)
**Use for:** Complete project including Phase 3 deep learning
```bash
pip install -r requirements.txt
```

**Includes:**
- pandas, numpy, scipy (data processing)
- scikit-learn, scikit-surprise (ML)
- mlxtend (FP-Growth)
- tensorflow, keras (deep learning)
- matplotlib, seaborn, networkx (visualization)
- tqdm, psutil (utilities)

**Size:** ~2.5GB installed

---

#### **2. requirements-minimal.txt** (Lightweight)
**Use for:** Phases 1-2 only (no deep learning)
```bash
pip install -r requirements-minimal.txt
```

**Includes:**
- Everything except tensorflow/keras
- Sufficient for all core experiments
- Association rules, clustering, recommendations

**Size:** ~500MB installed

---

#### **3. requirements-dev.txt** (Development)
**Use for:** If contributing or developing
```bash
pip install -r requirements-dev.txt
```

**Includes:**
- All from requirements.txt
- Plus: jupyter, pytest, black, flake8, sphinx

---

## 🚀 Step 3: Complete Setup Workflow

### Fresh project setup:

```bash
# 1. Navigate to project directory
cd ~/Desktop/CSCI\ 6443\ Data\ Mining\ -\ Project

# 2. Create/activate virtual environment
python3.13 -m venv venv
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies (choose one):
pip install -r requirements.txt          # Full version
# OR
pip install -r requirements-minimal.txt  # Lightweight

# 5. Verify installation
python -c "import pandas, numpy, scipy, sklearn, mlxtend; print('✓ All packages installed')"

# 6. Verify data
python scripts/01_verify_data.py

# 7. Ready to run!
python scripts/24_phase1_master_pipeline.py
```

---

## 🔄 Step 4: When to Update Requirements

### Add new packages:

```bash
# Install new package
pip install some-new-package

# Update requirements.txt
pip freeze > requirements-temp.txt

# Manually merge with requirements.txt (keep versions organized)
# Then delete requirements-temp.txt
```

### Why not just `pip freeze`?

❌ **Don't do this:**
```bash
pip freeze > requirements.txt  # Creates messy file with ALL dependencies
```

✅ **Do this:**
- Manually maintain requirements.txt with main packages only
- Let pip handle sub-dependencies automatically
- Keep it clean and readable

---

## 📝 Step 5: Documentation Placement

### After downloading from Claude:

```bash
# 1. Main README
mv comprehensive_README.md README.md

# 2. Create docs folder
mkdir -p docs

# 3. Move detailed docs
mv DEVELOPMENT_JOURNEY.md docs/
mv SCRIPTS_REFERENCE.md docs/
mv DOCUMENTATION_UPDATES.md docs/

# 4. Create docs index
cat > docs/README.md << 'DOCINDEX'
# Documentation Index

## Files in this Directory

- **DEVELOPMENT_JOURNEY.md** - Development timeline and pivots
- **SCRIPTS_REFERENCE.md** - Detailed script documentation
- **DOCUMENTATION_UPDATES.md** - Changelog

## Navigation

- [Main README](../README.md) - Project overview
- [Development Journey](./DEVELOPMENT_JOURNEY.md) - Technical decisions
- [Scripts Reference](./SCRIPTS_REFERENCE.md) - Script details
DOCINDEX

# 5. Verify structure
tree -L 2 .  # or ls -R if tree not installed
```

---

## 🎯 Step 6: Verify Everything Works

### Quick verification checklist:

```bash
# ✓ Virtual environment active
echo $VIRTUAL_ENV  # Should show venv path

# ✓ Packages installed
pip list | grep -E "pandas|numpy|scikit-learn|mlxtend"

# ✓ Python version
python --version  # Should be 3.13+

# ✓ Data present
ls data/raw/mpd_slices/ | wc -l  # Should show 1000

# ✓ Scripts present
ls scripts/*.py | wc -l  # Should show 35+

# ✓ Documentation present
ls docs/*.md  # Should show 3-4 files
```

---

## 🐛 Troubleshooting

### Issue 1: ModuleNotFoundError
```bash
# Solution: Install requirements
pip install -r requirements.txt
```

### Issue 2: Wrong Python version
```bash
# Check version
python --version

# If < 3.13, install Python 3.13 from python.org
# Then recreate venv:
rm -rf venv
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Issue 3: Apple Silicon compatibility
```bash
# Make sure you're using Python.org Python, not Homebrew
which python3.13
# Should show: /Library/Frameworks/Python.framework/...
# NOT: /opt/homebrew/...

# If using Homebrew Python, uninstall and get from python.org
```

### Issue 4: TensorFlow installation fails
```bash
# Option 1: Use minimal requirements instead
pip install -r requirements-minimal.txt

# Option 2: Install TensorFlow separately
pip install tensorflow-macos tensorflow-metal  # For Apple Silicon
```

---

## 📊 Disk Space Requirements

| Component | Size | Location |
|-----------|------|----------|
| Raw data (MPD) | 35GB | data/raw/mpd_slices/ |
| Processed data | 3GB | data/processed/ |
| Python packages (full) | 2.5GB | venv/ |
| Python packages (minimal) | 500MB | venv/ |
| Outputs | 1-5GB | outputs/ |
| Logs | 100MB | logs/ |
| **Total (full)** | **~45GB** | |
| **Total (minimal)** | **~42GB** | |

**Recommendation:** Have at least 100GB free disk space

---

## 🔐 Git Setup (Optional)

### Initialize Git:

```bash
# Initialize repo
git init

# Add files (respects .gitignore)
git add .

# First commit
git commit -m "Initial commit: Spotify playlist mining project"

# Connect to GitHub
git remote add origin https://github.com/YOUR_USERNAME/spotify-playlist-mining.git
git push -u origin main
```

### What gets committed:

✅ **Committed:**
- All Python scripts
- README.md and docs/
- requirements.txt files
- .gitignore
- src/ modules

❌ **Not committed (via .gitignore):**
- data/ (too large)
- venv/ (environment-specific)
- logs/ (temporary)
- outputs/ (results, can regenerate)
- *.pyc, __pycache__/

---

## 📚 Next Steps

After setup:

1. **Verify data:** `python scripts/01_verify_data.py`
2. **Run Phase 1:** `python scripts/24_phase1_master_pipeline.py` (22 min)
3. **Run Phase 2:** `python scripts/31_phase2_master_pipeline.py` (45 min)
4. **Analyze results:** Check `outputs/results/`

---

## 🆘 Getting Help

If something doesn't work:

1. Check this guide's troubleshooting section
2. Review SCRIPTS_REFERENCE.md for script-specific issues
3. Check logs in `logs/` directory
4. Review DEVELOPMENT_JOURNEY.md for context on decisions

---

*Last updated: November 25, 2024*