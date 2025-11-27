"""
Spotify Playlist Extension Dashboard
Interactive web-based dashboard for exploring playlist mining results

Author: Adarsh Singh
Date: November 2024
"""

import streamlit as st
import sys
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Spotify Playlist Mining Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1DB954;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1DB954;
    }
    .metric-label {
        font-size: 1rem;
        color: #555;
        margin-top: 0.5rem;
    }
    .stButton>button {
        background-color: #1DB954;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.5rem 2rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1ed760;
    }
</style>
""", unsafe_allow_html=True)

# Main page
st.markdown('<div class="main-header">🎵 Spotify Playlist Extension</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Pattern Mining and Clustering for Intelligent Recommendations</div>', unsafe_allow_html=True)

st.markdown("---")

# Project overview
st.markdown("## 📊 Project Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">1M</div>
        <div class="metric-label">Playlists Analyzed</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">2.3M</div>
        <div class="metric-label">Unique Tracks</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">89x</div>
        <div class="metric-label">Performance Improvement</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Research questions
st.markdown("## 🔬 Research Questions")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Question 1")
    st.markdown("**Co-occurrence Patterns**")
    st.info("How often do songs co-occur or co-disappear in playlists, and how can this inform recommendations?")

with col2:
    st.markdown("### Question 2")
    st.markdown("**Clustering Effectiveness**")
    st.info("Can playlists and tracks be effectively clustered by genre or features to improve recommendations?")

with col3:
    st.markdown("### Question 3")
    st.markdown("**Metadata Influence**")
    st.info("How does playlist metadata (titles, partial tracks) influence recommendation quality?")

st.markdown("---")

# Navigation
st.markdown("## 🧭 Dashboard Navigation")

st.markdown("""
Use the sidebar to navigate between pages:

1. **📊 Overview** - Dataset statistics and key findings
2. **🎯 Model Performance** - Compare recommendation algorithms
3. **🎵 Recommendations** - Try the recommendation system
4. **🗂️ Clusters** - Explore playlist clustering
5. **🔗 Association Rules** - Discover track relationships
6. **📈 Advanced Analytics** - Deep dive into models
7. **⏱️ Timeline** - Project milestones and progress
""")

st.markdown("---")

# Key findings
st.markdown("## 🎯 Key Findings")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### ✅ Achievements")
    st.success("✓ 89x improvement over popularity baseline")
    st.success("✓ Identified distinct playlist clusters")
    st.success("✓ Generated meaningful association rules")
    st.success("✓ High classification accuracy (>85%)")

with col2:
    st.markdown("### 🔍 Techniques Used")
    st.info("• Association rule mining (Apriori)")
    st.info("• K-means clustering with TF-IDF")
    st.info("• Matrix factorization (SVD)")
    st.info("• Neural network embeddings")

st.markdown("---")

# Footer
st.markdown("""
<div style='text-align: center; color: #777; padding: 2rem;'>
    <p><strong>Spotify Playlist Extension Project</strong></p>
    <p>CSCI 6443 Data Mining • George Washington University • Fall 2024</p>
    <p>Adarsh Singh</p>
</div>
""", unsafe_allow_html=True)