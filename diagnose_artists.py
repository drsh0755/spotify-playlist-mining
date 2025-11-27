"""
Diagnose the Top 10 Artists issue
"""
import pandas as pd
from pathlib import Path

print("=" * 60)
print("DIAGNOSING TOP 10 ARTISTS ISSUE")
print("=" * 60)

tracks_path = Path("data/processed/tracks_full_mpd.parquet")

if not tracks_path.exists():
    print(f"❌ File not found: {tracks_path}")
    exit(1)

print(f"✅ File exists: {tracks_path}")
print(f"File size: {tracks_path.stat().st_size / 1e9:.2f} GB")

print("\n" + "=" * 60)
print("LOADING DATA...")
print("=" * 60)

tracks = pd.read_parquet(tracks_path)
print(f"✅ Loaded {len(tracks):,} rows")
print(f"Columns: {list(tracks.columns)}")
print(f"\nFirst few rows:")
print(tracks.head(3))

print("\n" + "=" * 60)
print("CHECKING ARTIST_NAME COLUMN")
print("=" * 60)

if 'artist_name' in tracks.columns:
    print("✅ 'artist_name' column exists")
    print(f"Total values: {len(tracks['artist_name']):,}")
    print(f"Non-null values: {tracks['artist_name'].notna().sum():,}")
    print(f"Null values: {tracks['artist_name'].isna().sum():,}")
    print(f"Unique artists: {tracks['artist_name'].nunique():,}")
    
    print("\n" + "=" * 60)
    print("ATTEMPTING value_counts()...")
    print("=" * 60)
    
    try:
        artist_series = tracks['artist_name'].dropna()
        print(f"After dropna: {len(artist_series):,} values")
        
        top_artists = artist_series.value_counts().head(10)
        print("✅ SUCCESS! Top 10 artists:")
        print(top_artists)
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
else:
    print("❌ 'artist_name' column NOT FOUND!")
    print(f"Available columns: {list(tracks.columns)}")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
