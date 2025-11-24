"""
Spotify Data Loader Module

Provides unified interface for loading and accessing Spotify playlist data.
Handles both challenge set and full MPD dataset with efficient caching.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from scipy.sparse import csr_matrix, lil_matrix
import logging


class SpotifyDataLoader:
    """
    Unified data loader for Spotify Million Playlist Dataset
    
    Features:
    - Load challenge set or MPD slices
    - Efficient caching with pickle
    - Track/playlist indexing
    - Feature extraction
    - Co-occurrence matrix generation
    """
    
    def __init__(self, data_dir: Optional[Path] = None, cache_dir: Optional[Path] = None):
        """
        Initialize data loader
        
        Args:
            data_dir: Root data directory (default: project_root/data)
            cache_dir: Cache directory for processed data (default: data/processed)
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"
        if cache_dir is None:
            cache_dir = data_dir / "processed"
        
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.playlists = []
        self.track_to_id = {}  # track_uri -> internal_id
        self.id_to_track = {}  # internal_id -> track_uri
        self.track_to_artist = {}  # track_uri -> artist_name
        self.track_counter = Counter()  # track_uri -> count
        
        self.logger = logging.getLogger(__name__)
    
    def load_challenge_set(self, use_cache: bool = True) -> List[Dict]:
        """
        Load challenge set with optional caching
        
        Args:
            use_cache: Use cached data if available
            
        Returns:
            List of playlist dictionaries
        """
        cache_file = self.cache_dir / "challenge_set_processed.pkl"
        
        # Try loading from cache
        if use_cache and cache_file.exists():
            self.logger.info(f"Loading from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            self.playlists = cached_data['playlists']
            self.track_to_id = cached_data['track_to_id']
            self.id_to_track = cached_data['id_to_track']
            self.track_to_artist = cached_data['track_to_artist']
            self.track_counter = cached_data['track_counter']
            
            self.logger.info(f"Loaded {len(self.playlists):,} playlists from cache")
            return self.playlists
        
        # Load from raw JSON
        raw_path = self.data_dir / "raw" / "challenge_set.json"
        self.logger.info(f"Loading from raw file: {raw_path}")
        
        with open(raw_path, 'r') as f:
            data = json.load(f)
        
        self.playlists = data.get('playlists', [])
        self.logger.info(f"Loaded {len(self.playlists):,} playlists")
        
        # Build indices
        self._build_indices()
        
        # Cache the processed data
        if use_cache:
            self.logger.info(f"Caching processed data to: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'playlists': self.playlists,
                    'track_to_id': self.track_to_id,
                    'id_to_track': self.id_to_track,
                    'track_to_artist': self.track_to_artist,
                    'track_counter': self.track_counter
                }, f)
        
        return self.playlists
    
    def load_mpd_slices(self, n_slices: int = 10, use_cache: bool = True) -> List[Dict]:
        """
        Load N slices from full MPD dataset
        
        Args:
            n_slices: Number of slices to load (each has ~1000 playlists)
            use_cache: Use cached data if available
            
        Returns:
            List of playlist dictionaries
        """
        cache_file = self.cache_dir / f"mpd_{n_slices}_slices_processed.pkl"
        
        # Try loading from cache
        if use_cache and cache_file.exists():
            self.logger.info(f"Loading from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            self.playlists = cached_data['playlists']
            self.track_to_id = cached_data['track_to_id']
            self.id_to_track = cached_data['id_to_track']
            self.track_to_artist = cached_data['track_to_artist']
            self.track_counter = cached_data['track_counter']
            
            self.logger.info(f"Loaded {len(self.playlists):,} playlists from cache")
            return self.playlists
        
        # Load from raw slices
        mpd_dir = self.data_dir / "raw" / "mpd_slices"
        slice_files = sorted(mpd_dir.glob("mpd.slice.*.json"))[:n_slices]
        
        self.logger.info(f"Loading {len(slice_files)} MPD slices")
        self.playlists = []
        
        for slice_file in slice_files:
            self.logger.info(f"Loading: {slice_file.name}")
            with open(slice_file, 'r') as f:
                data = json.load(f)
            self.playlists.extend(data.get('playlists', []))
        
        self.logger.info(f"Loaded {len(self.playlists):,} playlists total")
        
        # Build indices
        self._build_indices()
        
        # Cache the processed data
        if use_cache:
            self.logger.info(f"Caching processed data to: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'playlists': self.playlists,
                    'track_to_id': self.track_to_id,
                    'id_to_track': self.id_to_track,
                    'track_to_artist': self.track_to_artist,
                    'track_counter': self.track_counter
                }, f)
        
        return self.playlists
    
    def _build_indices(self):
        """Build track indices and metadata"""
        self.logger.info("Building track indices...")
        
        track_id = 0
        self.track_to_id = {}
        self.id_to_track = {}
        self.track_to_artist = {}
        self.track_counter = Counter()
        
        for playlist in self.playlists:
            for track in playlist.get('tracks', []):
                track_uri = track.get('track_uri')
                if not track_uri:
                    continue
                
                # Assign ID if new track
                if track_uri not in self.track_to_id:
                    self.track_to_id[track_uri] = track_id
                    self.id_to_track[track_id] = track_uri
                    track_id += 1
                
                # Store artist info
                if 'artist_name' in track:
                    self.track_to_artist[track_uri] = track.get('artist_name')
                
                # Count occurrences
                self.track_counter[track_uri] += 1
        
        self.logger.info(f"Built indices for {len(self.track_to_id):,} unique tracks")
    
    def get_tracks_dataframe(self) -> pd.DataFrame:
        """
        Get tracks as a pandas DataFrame
        
        Returns:
            DataFrame with columns: track_id, track_uri, artist_name, playlist_count
        """
        data = []
        for track_uri, track_id in self.track_to_id.items():
            data.append({
                'track_id': track_id,
                'track_uri': track_uri,
                'artist_name': self.track_to_artist.get(track_uri, 'Unknown'),
                'playlist_count': self.track_counter[track_uri]
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('playlist_count', ascending=False).reset_index(drop=True)
    
    def get_playlists_dataframe(self) -> pd.DataFrame:
        """
        Get playlists as a pandas DataFrame
        
        Returns:
            DataFrame with playlist metadata
        """
        data = []
        for i, playlist in enumerate(self.playlists):
            data.append({
                'playlist_idx': i,
                'pid': playlist.get('pid'),
                'name': playlist.get('name', ''),
                'num_tracks': len(playlist.get('tracks', [])),
                'num_samples': playlist.get('num_samples', 0),
                'num_followers': playlist.get('num_followers', 0),
                'modified_at': playlist.get('modified_at', 0)
            })
        
        return pd.DataFrame(data)
    
    def get_cooccurrence_matrix(self, min_support: int = 2) -> Tuple[csr_matrix, Dict]:
        """
        Build track co-occurrence matrix
        
        Args:
            min_support: Minimum number of playlists a track must appear in
            
        Returns:
            Sparse co-occurrence matrix and metadata dict
        """
        self.logger.info(f"Building co-occurrence matrix (min_support={min_support})...")
        
        # Filter tracks by support
        valid_tracks = {uri for uri, count in self.track_counter.items() if count >= min_support}
        self.logger.info(f"Using {len(valid_tracks):,} tracks with support >= {min_support}")
        
        # Create filtered track mapping
        filtered_track_to_id = {uri: i for i, uri in enumerate(sorted(valid_tracks))}
        n_tracks = len(filtered_track_to_id)
        
        # Build co-occurrence matrix
        cooccur = lil_matrix((n_tracks, n_tracks), dtype=np.int32)
        
        for playlist in self.playlists:
            track_uris = [t.get('track_uri') for t in playlist.get('tracks', [])]
            track_uris = [uri for uri in track_uris if uri in valid_tracks]
            
            # Update co-occurrence for all pairs
            for i, uri1 in enumerate(track_uris):
                id1 = filtered_track_to_id[uri1]
                for uri2 in track_uris[i:]:
                    id2 = filtered_track_to_id[uri2]
                    if id1 != id2:
                        cooccur[id1, id2] += 1
                        cooccur[id2, id1] += 1
        
        # Convert to CSR for efficient operations
        cooccur_csr = cooccur.tocsr()
        
        metadata = {
            'n_tracks': n_tracks,
            'track_to_id': filtered_track_to_id,
            'id_to_track': {i: uri for uri, i in filtered_track_to_id.items()},
            'min_support': min_support
        }
        
        self.logger.info(f"Co-occurrence matrix built: {n_tracks} x {n_tracks}")
        self.logger.info(f"Non-zero entries: {cooccur_csr.nnz:,}")
        
        return cooccur_csr, metadata
    
    def get_playlist_features(self) -> pd.DataFrame:
        """
        Extract features from playlists for clustering
        
        Returns:
            DataFrame with playlist features
        """
        features = []
        
        for i, playlist in enumerate(self.playlists):
            tracks = playlist.get('tracks', [])
            
            # Basic features
            feature_dict = {
                'playlist_idx': i,
                'num_tracks': len(tracks),
                'name_length': len(playlist.get('name', '')),
                'has_description': 1 if playlist.get('description') else 0,
                'num_followers': playlist.get('num_followers', 0)
            }
            
            # Track-based features
            if tracks:
                artists = [t.get('artist_name', '') for t in tracks]
                feature_dict['unique_artists'] = len(set(artists))
                feature_dict['artist_diversity'] = len(set(artists)) / len(tracks) if tracks else 0
            else:
                feature_dict['unique_artists'] = 0
                feature_dict['artist_diversity'] = 0
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def save_processed_data(self, output_path: Optional[Path] = None):
        """Save all processed data to a single file"""
        if output_path is None:
            output_path = self.cache_dir / "full_processed_data.pkl"
        
        self.logger.info(f"Saving processed data to: {output_path}")
        
        with open(output_path, 'wb') as f:
            pickle.dump({
                'playlists': self.playlists,
                'track_to_id': self.track_to_id,
                'id_to_track': self.id_to_track,
                'track_to_artist': self.track_to_artist,
                'track_counter': self.track_counter
            }, f)
        
        self.logger.info("✓ Data saved successfully")
