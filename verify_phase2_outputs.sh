#!/bin/bash

echo "=== Checking Phase 2 Outputs ==="
echo ""

cd ~/Documents/George\ Washington\ University/Fall25/Data\ Mining_CSCI_6443/CSCI\ 6443\ Data\ Mining\ -\ Project 2>/dev/null || {
    echo "❌ Project directory not found"
    exit 1
}

echo "✓ Found project directory: $(pwd)"
echo ""

check_file() {
    if [ -f "$1" ]; then
        size=$(ls -lh "$1" | awk '{print $5}')
        echo "  ✅ Found: $(basename $1) ($size)"
        return 0
    else
        echo "  ❌ Missing: $(basename $1)"
        return 1
    fi
}

found=0
total=0

echo "Script 25 - Association Rules:"
total=$((total + 1))
check_file "data/processed/association_rules_full.csv" && found=$((found + 1))

echo ""
echo "Script 26 - Clustering:"
total=$((total + 2))
check_file "data/processed/track_clusters_full.csv" && found=$((found + 1))
check_file "data/processed/cluster_profiles_full.csv" && found=$((found + 1))

echo ""
echo "Script 27 - Recommendations:"
total=$((total + 1))
check_file "data/processed/sample_recommendations_full.csv" && found=$((found + 1))

echo ""
echo "Script 28 - Evaluation:"
total=$((total + 1))
check_file "data/processed/evaluation_metrics_full.json" && found=$((found + 1))

echo ""
echo "Script 29 - Diversity:"
total=$((total + 1))
check_file "data/processed/diversity_metrics_full.json" && found=$((found + 1))

echo ""
echo "Script 30 - Category Evaluation:"
total=$((total + 3))
check_file "data/processed/category_evaluation_full.json" && found=$((found + 1))
check_file "data/processed/category_by_genre_full.csv" && found=$((found + 1))
check_file "data/processed/category_by_size_full.csv" && found=$((found + 1))

echo ""
echo "=== Summary ==="
echo "Files found: $found / $total"
echo ""

if [ $found -eq $total ]; then
    echo "🎉 Perfect! All Phase 2 outputs exist!"
    echo "✅ Dashboard will use 100% REAL DATA"
elif [ $found -ge 7 ]; then
    echo "✅ Most files present! Dashboard will work well."
    echo "⚠️  Some optional files missing (will use simulated data for those)"
else
    echo "⚠️  Many files missing."
    echo "Options:"
    echo "  1. Re-run individual scripts that failed"
    echo "  2. Run script 31 to regenerate all Phase 2 outputs"
fi
