#!/bin/bash
# Script to check what data files you have for the dashboard

echo "======================================================================="
echo "CHECKING YOUR DATA FILES FOR DASHBOARD"
echo "======================================================================="
echo ""

# Use current directory (don't change directories)
echo "Current directory: $(pwd)"
echo ""

# Function to check file and show size
check_file() {
    if [ -f "$1" ]; then
        size=$(ls -lh "$1" | awk '{print $5}')
        echo "✅ FOUND: $1 ($size)"
        return 0
    else
        echo "❌ MISSING: $1"
        return 1
    fi
}

echo "======================================================================="
echo "1. RECOMMENDATION RESULTS"
echo "======================================================================="
check_file "outputs/results/recommendations_baseline.pkl"
check_file "outputs/results/recommendations_cooccurrence.pkl"
check_file "outputs/results/recommendations_svd.pkl"
check_file "outputs/results/recommendations_hybrid.pkl"
check_file "outputs/results/evaluation_results.pkl"
echo ""

echo "======================================================================="
echo "2. CLUSTER DATA"
echo "======================================================================="
check_file "data/processed/cluster_assignments.pkl"
check_file "data/processed/tfidf_features_full.npz"
check_file "outputs/results/cluster_stats.pkl"
echo ""

echo "======================================================================="
echo "3. ASSOCIATION RULES"
echo "======================================================================="
check_file "outputs/results/association_rules_full.csv"
check_file "data/processed/cooccurrence_matrix_full.npz"
echo ""

echo "======================================================================="
echo "4. BASIC DATA FILES"
echo "======================================================================="
check_file "data/processed/tracks_full_mpd.parquet"
check_file "data/processed/playlists_full_mpd.parquet"
check_file "data/processed/track_mappings.pkl"
echo ""

echo "======================================================================="
echo "5. ADVANCED MODELS"
echo "======================================================================="
check_file "outputs/models/svd_model.pkl"
check_file "outputs/models/neural_embeddings.pkl"
check_file "outputs/models/rf_classifier.pkl"
check_file "outputs/models/rf_regressor.pkl"
echo ""

echo "======================================================================="
echo "SUMMARY"
echo "======================================================================="
echo ""

# Count what we have
RECS=0
CLUSTERS=0
RULES=0
BASIC=0
MODELS=0

[ -f "outputs/results/recommendations_hybrid.pkl" ] && RECS=1
[ -f "data/processed/cluster_assignments.pkl" ] && CLUSTERS=1
[ -f "outputs/results/association_rules_full.csv" ] && RULES=1
[ -f "data/processed/tracks_full_mpd.parquet" ] && BASIC=1
[ -f "outputs/models/svd_model.pkl" ] && MODELS=1

echo "Data availability:"
echo "  Recommendation Results: $([[ $RECS -eq 1 ]] && echo '✅ YES' || echo '❌ NO')"
echo "  Cluster Assignments: $([[ $CLUSTERS -eq 1 ]] && echo '✅ YES' || echo '❌ NO')"
echo "  Association Rules: $([[ $RULES -eq 1 ]] && echo '✅ YES' || echo '❌ NO')"
echo "  Basic Data Files: $([[ $BASIC -eq 1 ]] && echo '✅ YES' || echo '❌ NO')"
echo "  Advanced Models: $([[ $MODELS -eq 1 ]] && echo '✅ YES' || echo '❌ NO')"
echo ""

echo "======================================================================="
echo "RECOMMENDATION"
echo "======================================================================="
echo ""

if [[ $BASIC -eq 1 ]]; then
    echo "✅ You have basic data - can build dashboard with simulated results"
    echo ""
    echo "Best option: Option B or C (Essential/Demo Dashboard)"
    echo "  - Will use your actual track/playlist data"
    echo "  - Will simulate recommendation results for demo"
    echo "  - Looks professional and works great"
else
    echo "⚠️  Missing basic data files"
    echo ""
    echo "Best option: Option C (Demo Dashboard)"
    echo "  - Will use completely simulated data"
    echo "  - Still looks professional"
    echo "  - Shows what the system could do"
fi

if [[ $RECS -eq 1 && $CLUSTERS -eq 1 && $RULES -eq 1 && $MODELS -eq 1 ]]; then
    echo ""
    echo "🎉 WOW! You have EVERYTHING!"
    echo ""
    echo "Best option: Option A (Full Dashboard)"
    echo "  - Can use all your actual results"
    echo "  - Most impressive"
    echo "  - Show off everything you built"
fi

echo ""
echo "======================================================================="
