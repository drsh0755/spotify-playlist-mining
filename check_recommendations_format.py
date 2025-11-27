"""
Check the format of sample_recommendations_full.csv
"""
import pandas as pd
from pathlib import Path

print("=" * 60)
print("CHECKING SAMPLE_RECOMMENDATIONS_FULL.CSV FORMAT")
print("=" * 60)

recs_file = Path("data/processed/sample_recommendations_full.csv")

if not recs_file.exists():
    print(f"❌ File not found: {recs_file}")
    exit(1)

print(f"✅ File exists: {recs_file}")
print(f"File size: {recs_file.stat().st_size / 1024:.1f} KB")

print("\n" + "=" * 60)
print("LOADING DATA...")
print("=" * 60)

recs = pd.read_csv(recs_file)
print(f"✅ Loaded {len(recs):,} rows")
print(f"\nColumns: {list(recs.columns)}")
print(f"\nFirst 5 rows:")
print(recs.head())

print("\n" + "=" * 60)
print("COLUMN DETAILS")
print("=" * 60)

for col in recs.columns:
    print(f"\n{col}:")
    print(f"  Type: {recs[col].dtype}")
    print(f"  Unique values: {recs[col].nunique():,}")
    print(f"  Sample: {recs[col].iloc[0]}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
