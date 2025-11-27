# Spotify Playlist Extension - Complete Documentation

**Project:** CSCI 6443 Data Mining Final Project  
**Author:** Adarsh Singh  
**Institution:** George Washington University  
**Date:** November 2024

---

## 📚 Documentation Structure

This project includes comprehensive documentation across multiple files:

### 1. [comprehensive_README.md](computer:///home/claude/comprehensive_README.md)
**Primary documentation file**
- Complete project overview
- Development journey with all pivots explained
- Research questions and methodology
- Installation and setup instructions
- Results and key findings
- **Start here for overview**

### 2. [DEVELOPMENT_JOURNEY.md](computer:///home/claude/DEVELOPMENT_JOURNEY.md)
**Detailed development pivots and reasoning**
- Timeline of infrastructure changes
- AWS + PyCharm → VS Code → Local development
- Why each pivot was made
- Performance comparisons
- Cost analysis
- **Read this to understand technical decisions**

### 3. [SCRIPTS_REFERENCE.md](computer:///home/claude/SCRIPTS_REFERENCE.md)
**Complete script documentation**
- All 35 scripts explained in detail
- Usage examples for each script
- Runtime and memory requirements
- Troubleshooting guide
- **Use this as reference during execution**

---

## 🚀 Quick Start Guide

### For Someone New to This Project:

1. **Understand the project:** Read sections 1-4 of comprehensive_README.md
2. **Set up environment:** Follow Installation & Setup in comprehensive_README.md
3. **Run the pipeline:** Use the master pipeline scripts (24 and 31)
4. **Understand what ran:** Reference SCRIPTS_REFERENCE.md
5. **Learn from decisions:** Read DEVELOPMENT_JOURNEY.md

### For Reproducing Results:

```bash
# 1. Clone and setup
git clone https://github.com/drsh0755/spotify-playlist-mining.git
cd spotify-playlist-mining
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Download dataset (see comprehensive_README.md for links)
# Place in data/raw/mpd_slices/

# 3. Run pipelines
python scripts/24_phase1_master_pipeline.py    # 22 minutes
python scripts/31_phase2_master_pipeline.py    # 65 minutes

# 4. View results
ls outputs/results/
cat outputs/phase2_summary_report.txt
```

---

## 📖 What Each Document Covers

### comprehensive_README.md Highlights:
- ✅ Complete project overview
- ✅ All three research questions answered
- ✅ Development journey (AWS→Local) with detailed reasoning
- ✅ Technical architecture and stack
- ✅ Installation instructions
- ✅ Project structure
- ✅ Pipeline execution guide
- ✅ Key results (89x improvement over baseline!)
- ✅ Scripts documentation overview
- ✅ Troubleshooting guide

### DEVELOPMENT_JOURNEY.md Highlights:
- ✅ Timeline of all pivots
- ✅ Why we chose AWS + PyCharm initially
- ✅ Problems encountered with AWS
- ✅ Why we switched to VS Code
- ✅ Why we ultimately went local
- ✅ M4 MacBook vs AWS performance comparison
- ✅ Cost analysis ($150 saved)
- ✅ Key learnings about cloud vs local
- ✅ Recommendations for future projects

### SCRIPTS_REFERENCE.md Highlights:
- ✅ All 35 scripts documented
- ✅ Quick reference table
- ✅ Detailed algorithm explanations
- ✅ Usage examples for each script
- ✅ Expected outputs
- ✅ Runtime and memory requirements
- ✅ Monitoring and debugging tips
- ✅ Common error patterns and fixes

---

## 🎯 Use Cases for Each Document

### You Should Read comprehensive_README.md If:
- You're new to the project
- You need to understand what was accomplished
- You want to reproduce the results
- You're writing about this project
- You need to cite the methodology

### You Should Read DEVELOPMENT_JOURNEY.md If:
- You're curious why certain decisions were made
- You're planning a similar project
- You want to understand cloud vs local trade-offs
- You're interested in Apple Silicon performance
- You need to justify architecture decisions

### You Should Read SCRIPTS_REFERENCE.md If:
- You're running the scripts
- You need to understand what a specific script does
- You're debugging an error
- You want to modify or extend a script
- You need technical details about algorithms

---

## 📊 Key Achievements Documented

### Research Accomplishments:
1. **RQ1 (Co-occurrence):** 1.36M association rules mined using FP-Growth
2. **RQ2 (Clustering):** 12 optimal clusters discovered (k-means, silhouette=0.68)
3. **RQ3 (Recommendations):** 89x improvement over popularity baseline

### Technical Accomplishments:
1. Processed 1M playlists, 2.3M unique tracks
2. Built 10K×10K sparse co-occurrence matrix
3. Implemented 4+ recommendation algorithms
4. Achieved R-precision: 0.178, NDCG: 0.234

### Infrastructure Journey:
1. Started: AWS g5.xlarge + PyCharm
2. Pivoted: AWS + VS Code
3. Final: Local M4 MacBook (superior performance)
4. Saved: $150 in costs, 4x productivity increase

---

## 🔗 Important Links

**GitHub Repository:**
https://github.com/drsh0755/spotify-playlist-mining

**Dataset Source:**
https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge

**RecSys Challenge 2018:**
https://recsys-challenge.spotify.com/

---

## 💡 Tips for Readers

### If You Have 5 Minutes:
Read the "Project Overview" and "Key Results" sections of comprehensive_README.md

### If You Have 15 Minutes:
Read comprehensive_README.md sections 1-6, focusing on:
- Development Journey & Pivots
- Research Questions
- Key Results

### If You Have 30 Minutes:
Read all of comprehensive_README.md to understand the complete project

### If You Have 1 Hour:
Read comprehensive_README.md + DEVELOPMENT_JOURNEY.md for full context

### If You're Running the Code:
Keep SCRIPTS_REFERENCE.md open for reference while executing

---

## 📝 Document Maintenance

**Last Updated:** November 25, 2024

**Version:** 1.0

**Status:** Complete and ready for submission

**Future Updates:**
- Add Phase 3 advanced modeling details (if completed)
- Include final presentation slides
- Add demo application documentation

---

## ✉️ Contact

**Adarsh Singh**  
MS in Data Science  
George Washington University

For questions about the documentation or project:
- Check the comprehensive_README.md first
- Review SCRIPTS_REFERENCE.md for technical issues
- See DEVELOPMENT_JOURNEY.md for architectural questions

---

## 🙏 Acknowledgments

Documentation created with attention to:
- Clarity for future readers
- Reproducibility of results
- Transparency about decisions
- Learning from challenges
- Comprehensive technical details

Special thanks to the open-source community for the tools that made this project possible.

---

*This index serves as a navigation guide to all project documentation.*
