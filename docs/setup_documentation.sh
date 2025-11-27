#!/bin/bash
# Setup documentation structure for Spotify Playlist Mining project

echo "Setting up documentation structure..."
echo ""

# Navigate to your project directory (update this path)
PROJECT_DIR="$HOME/Desktop/CSCI 6443 Data Mining - Project"

cd "$PROJECT_DIR" || exit 1

# Create docs directory if it doesn't exist
mkdir -p docs

# Copy main README (comprehensive_README.md becomes README.md)
echo "1. Setting up main README.md..."
cp comprehensive_README.md README.md

# Move other docs to docs/ folder
echo "2. Moving detailed documentation to docs/..."
mv DEVELOPMENT_JOURNEY.md docs/ 2>/dev/null || echo "   (DEVELOPMENT_JOURNEY.md not found, will download)"
mv SCRIPTS_REFERENCE.md docs/ 2>/dev/null || echo "   (SCRIPTS_REFERENCE.md not found, will download)"
mv DOCUMENTATION_UPDATES.md docs/ 2>/dev/null || echo "   (DOCUMENTATION_UPDATES.md not found, will download)"

# Create DOCUMENTATION_INDEX.md in docs
echo "3. Creating docs/README.md (index)..."
cat > docs/README.md << 'DOCINDEX'
# Documentation Index

This folder contains detailed project documentation.

## Files

### [DEVELOPMENT_JOURNEY.md](./DEVELOPMENT_JOURNEY.md)
Complete development timeline, pivots, and technical decisions including:
- AWS → Local migration rationale
- Iterative problem-solving (6 major iterations documented)
- Hardware comparisons
- Cost analysis

### [SCRIPTS_REFERENCE.md](./SCRIPTS_REFERENCE.md)
Comprehensive reference for all 35 scripts including:
- Purpose and usage
- Runtime and memory requirements
- Algorithm explanations
- Troubleshooting guide

### [DOCUMENTATION_UPDATES.md](./DOCUMENTATION_UPDATES.md)
Changelog of documentation corrections and updates.

## Quick Links

- [Main README](../README.md) - Project overview
- [Setup Guide](../README.md#installation--setup) - Installation instructions
- [Scripts Reference](./SCRIPTS_REFERENCE.md) - Detailed script docs

## Navigation

**New to the project?** Start with the main [README.md](../README.md)

**Running the code?** See [SCRIPTS_REFERENCE.md](./SCRIPTS_REFERENCE.md)

**Curious about decisions?** Read [DEVELOPMENT_JOURNEY.md](./DEVELOPMENT_JOURNEY.md)
DOCINDEX

echo ""
echo "✅ Documentation structure created!"
echo ""
echo "Structure:"
echo "├── README.md (main project documentation)"
echo "├── requirements.txt (Python dependencies)"
echo "└── docs/"
echo "    ├── README.md (documentation index)"
echo "    ├── DEVELOPMENT_JOURNEY.md"
echo "    ├── SCRIPTS_REFERENCE.md"
echo "    └── DOCUMENTATION_UPDATES.md"
echo ""
echo "Next steps:"
echo "1. Download documentation files from Claude"
echo "2. Place them in the docs/ folder"
echo "3. Run this script: bash setup_documentation.sh"
