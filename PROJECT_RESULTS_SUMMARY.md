# Spotify Playlist Extension: Complete Results Summary

**Generated:** 2025-11-26 21:57:57

---

## 1. Dataset Statistics

- **Total Playlists:** 1,000,000
- **Total Track Entries:** 66,346,428
- **Unique Tracks:** 2,262,292
- **Unique Artists:** 287,742
- **Unique Albums:** 571,629
- **Average Playlist Length:** 66.3 tracks


## 2. Research Question 1: Song Co-occurrence Patterns

### 2.1 Association Rule Mining Results

- **Total Association Rules Generated:** 10,000
- **High Confidence Rules (>80%):** 9,999
- **High Lift Rules (>2.0):** 10,000
- **Average Confidence:** 2.799
- **Average Lift:** 1282.626
- **Average Support:** 0.006701

### 2.2 Top 10 Association Rules

```

Rule 1:
  antecedent: spotify:track:00t8f0jxCP556HvIoMulCZ
  consequent: spotify:track:3F14hbIqy1JKKLTb1mzdYV
  support: 0.0208147060855872
  confidence: 13.46039268788084
  lift: 10960.4317073213
  count: 19881

Rule 2:
  antecedent: spotify:track:3F14hbIqy1JKKLTb1mzdYV
  consequent: spotify:track:00t8f0jxCP556HvIoMulCZ
  support: 0.0208147060855872
  confidence: 16.948849104859335
  lift: 10960.4317073213
  count: 19881

Rule 3:
  antecedent: spotify:track:05Z77MHLPcBA2Wd5eDSJzl
  consequent: spotify:track:3iJmiaoZWzykf3Hl4IeYtI
  support: 0.013844014816645
  confidence: 11.629727352682496
  lift: 10900.92349665934
  count: 13223

Rule 4:
  antecedent: spotify:track:3iJmiaoZWzykf3Hl4IeYtI
  consequent: spotify:track:05Z77MHLPcBA2Wd5eDSJzl
  support: 0.013844014816645
  confidence: 12.976447497546614
  lift: 10900.92349665934
  count: 13223

Rule 5:
  antecedent: spotify:track:00t8f0jxCP556HvIoMulCZ
  consequent: spotify:track:0KkrT0y1iht0tqgh9vrGd3
  support: 0.0175188610698723
  confidence: 11.329045362220718
  lift: 10094.073736345355
  count: 16733

Rule 6:
  antecedent: spotify:track:0KkrT0y1iht0tqgh9vrGd3
  consequent: spotify:track:00t8f0jxCP556HvIoMulCZ
  support: 0.0175188610698723
  confidence: 15.609141791044776
  lift: 10094.073736345355
  count: 16733

Rule 7:
  antecedent: spotify:track:01Ch5LzVStxCFPKkT1xg6k
  consequent: spotify:track:2c2csx4OTYtbkzvbSTXlGY
  support: 0.0196913129147289
  confidence: 10.317059791552389
  lift: 8942.157099294855
  count: 18808

Rule 8:
  antecedent: spotify:track:2c2csx4OTYtbkzvbSTXlGY
  consequent: spotify:track:01Ch5LzVStxCFPKkT1xg6k
  support: 0.0196913129147289
  confidence: 17.067150635208712
  lift: 8942.157099294855
  count: 18808

Rule 9:
  antecedent: spotify:track:05Z77MHLPcBA2Wd5eDSJzl
  consequent: spotify:track:0XJDfLHO7C6b0GpjeB5mpu
  support: 0.0161379145718647
  confidence: 13.556728232189974
  lift: 8690.335917550601
  count: 15414

Rule 10:
  antecedent: spotify:track:0XJDfLHO7C6b0GpjeB5mpu
  consequent: spotify:track:05Z77MHLPcBA2Wd5eDSJzl
  support: 0.0161379145718647
  confidence: 10.34496644295302
  lift: 8690.335917550601
  count: 15414
```


## 3. Research Question 2: Clustering Results

- **Number of Clusters:** 5

### 3.1 Cluster Profiles

| Cluster ID | Size | Description |
|------------|------|-------------|
| 0 | N/A | - |
| 1 | N/A | - |
| 2 | N/A | - |
| 3 | N/A | - |
| 4 | N/A | - |


## 4. Research Question 3: Recommendation Performance

### 4.1 Overall Performance Metrics

```json
{
  "r_precision_mean": 0.1333418002592439,
  "r_precision_median": 0.08333333333333333,
  "r_precision_std": 0.1494684155698968,
  "playlists_evaluated": 4998,
  "ndcg_mean": 1.0,
  "ndcg_median": 1.0,
  "total_unique_tracks": 2262292,
  "total_unique_artists": 295860,
  "total_unique_albums": 734684,
  "tracks_in_1_playlist": 1073419,
  "tracks_in_10plus_playlists": 372433,
  "tracks_in_100plus_playlists": 70229,
  "tracks_in_1000plus_playlists": 10333
}
```

### 4.2 Model Comparison

```json
[
  {
    "model": "Popularity Baseline",
    "precision@10": 0.14609053497942387,
    "test_size": 972
  },
  {
    "model": "Co-occurrence",
    "precision@10": 0.026123128119800332,
    "test_size": 601
  },
  {
    "model": "SVD",
    "precision@10": 0.05005382131324004,
    "test_size": 929
  },
  {
    "model": "Neural (PCA)",
    "precision@10": 0.010441334768568353,
    "test_size": 929
  }
]
```

### 4.3 Diversity Metrics

```json
{
  "artist_diversity_mean": 0.643378118334969,
  "artist_diversity_median": 0.6701030927835051,
  "album_diversity_mean": 0.7929687265246611,
  "album_diversity_median": 0.8461538461538461,
  "avg_genres_per_playlist": 0.42681,
  "median_genres_per_playlist": 0.0,
  "genre_distribution": {
    "genre_workout": 21275,
    "genre_party": 31215,
    "genre_chill": 38350,
    "genre_rock": 26867,
    "genre_hip_hop": 26317,
    "genre_country": 24457,
    "genre_pop": 10387,
    "genre_indie": 6510,
    "genre_classical": 2697,
    "genre_jazz": 6454,
    "genre_electronic": 3416,
    "genre_latin": 5285,
    "genre_mood": 10175,
    "genre_tag_count": 213405
  },
  "popularity_gini": 0.9212534515246544,
  "unique_tracks": 2262292,
  "most_popular_track_count": 46574,
  "median_track_count": 2,
  "size_diversity_correlation": -0.3774200863174593
}
```

### 4.4 Category-wise Evaluation

```json
{
  "by_genre": {
    "workout": {
      "num_playlists": 21275,
      "avg_tracks_per_playlist": 62.066180963572265,
      "unique_tracks": 155852,
      "unique_artists": 33499
    },
    "party": {
      "num_playlists": 31215,
      "avg_tracks_per_playlist": 74.39083773826685,
      "unique_tracks": 257099,
      "unique_artists": 58790
    },
    "chill": {
      "num_playlists": 38350,
      "avg_tracks_per_playlist": 68.9006518904824,
      "unique_tracks": 346691,
      "unique_artists": 61738
    },
    "rock": {
      "num_playlists": 26867,
      "avg_tracks_per_playlist": 81.19231771317973,
      "unique_tracks": 214655,
      "unique_artists": 32751
    },
    "hip_hop": {
      "num_playlists": 26317,
      "avg_tracks_per_playlist": 79.63263289888665,
      "unique_tracks": 151192,
      "unique_artists": 27217
    },
    "country": {
      "num_playlists": 24457,
      "avg_tracks_per_playlist": 87.2474547164411,
      "unique_tracks": 90649,
      "unique_artists": 14550
    },
    "pop": {
      "num_playlists": 10387,
      "avg_tracks_per_playlist": 71.76874939828632,
      "unique_tracks": 113901,
      "unique_artists": 24223
    },
    "indie": {
      "num_playlists": 6510,
      "avg_tracks_per_playlist": 85.52995391705069,
      "unique_tracks": 93834,
      "unique_artists": 19729
    },
    "classical": {
      "num_playlists": 2697,
      "avg_tracks_per_playlist": 56.09454949944383,
      "unique_tracks": 62655,
      "unique_artists": 8224
    },
    "jazz": {
      "num_playlists": 6454,
      "avg_tracks_per_playlist": 62.98868918500155,
      "unique_tracks": 136126,
      "unique_artists": 23819
    },
    "electronic": {
      "num_playlists": 3416,
      "avg_tracks_per_playlist": 68.15310304449649,
      "unique_tracks": 74446,
      "unique_artists": 19187
    },
    "latin": {
      "num_playlists": 5285,
      "avg_tracks_per_playlist": 62.582970671712395,
      "unique_tracks": 46074,
      "unique_artists": 9643
    },
    "mood": {
      "num_playlists": 10175,
      "avg_tracks_per_playlist": 54.321375921375925,
      "unique_tracks": 113604,
      "unique_artists": 25668
    },
    "tag_count": {
      "num_playlists": 194517,
      "avg_tracks_per_playlist": 72.93552234509066,
      "unique_tracks": 949593,
      "unique_artists": 140971
    }
  },
  "by_size": {
    "tiny": {
      "num_playlists": 46675,
      "avg_tracks": 7.881328334226031,
      "avg_artist_diversity": 0.8255469346460241,
      "unique_tracks": 120297
    },
    "small": {
      "num_playlists": 198977,
      "avg_tracks": 18.156088392125724,
      "avg_artist_diversity": 0.7480899360052131,
      "unique_tracks": 481204
    },
    "medium": {
      "num_playlists": 266794,
      "avg_tracks": 37.08701095227029,
      "avg_artist_diversity": 0.6745886970050651,
      "unique_tracks": 805242
    },
    "large": {
      "num_playlists": 271072,
      "avg_tracks": 71.8289236807933,
      "avg_artist_diversity": 0.6059437792304518,
      "unique_tracks": 1162834
    },
    "huge": {
      "num_playlists": 216482,
      "avg_tracks": 152.44002272706277,
      "avg_artist_diversity": 0.516266548773249,
      "unique_tracks": 1650496
    }
  },
  "by_popularity": {
    "few_followers": {
      "num_playlists": 754219,
      "avg_tracks": 61.18661688448581,
      "avg_artist_diversity": 0.6467660593558118,
      "avg_followers": 1.0
    },
    "some_followers": {
      "num_playlists": 239269,
      "avg_tracks": 81.73989944372234,
      "avg_artist_diversity": 0.6335968380142345,
      "avg_followers": 2.773251027086668
    },
    "many_followers": {
      "num_playlists": 5696,
      "avg_tracks": 100.42661516853933,
      "avg_artist_diversity": 0.6106337977968148,
      "avg_followers": 23.448911516853933
    },
    "viral": {
      "num_playlists": 816,
      "avg_tracks": 83.9031862745098,
      "avg_artist_diversity": 0.6085978757470945,
      "avg_followers": 1282.3639705882354
    }
  }
}
```

## 5. Advanced Analysis

### 5.1 Graph Network Analysis

```json
{
  "structure": {
    "num_nodes": 10221,
    "num_edges": 77514899,
    "density": 0.7420632112505411,
    "avg_degree": 266203.2621074259,
    "median_degree": 144297.0,
    "max_degree": 3563509,
    "min_degree": 13194
  },
  "connectivity": {
    "avg_edge_weight": 35.101168641140845,
    "median_edge_weight": 6.0,
    "max_edge_weight": 45394,
    "min_edge_weight": 1,
    "avg_clustering": null
  },
  "hub_tracks": [
    {
      "rank": 1,
      "track_uri": "spotify:track:7KXjTSCq5nL1LoYtL7XAwS",
      "degree": 3563509
    },
    {
      "rank": 2,
      "track_uri": "spotify:track:1xznGGDReH1oQq0xzbwXa3",
      "degree": 3472640
    },
    {
      "rank": 3,
      "track_uri": "spotify:track:7yyRTcZmCiyzzJlNzGC9Ol",
      "degree": 3339992
    },
    {
      "rank": 4,
      "track_uri": "spotify:track:3a1lNhkSLSkpJE4MSHpDu9",
      "degree": 3128377
    },
    {
      "rank": 5,
      "track_uri": "spotify:track:7BKLCZ1jbUBVqRi2FVlTVw",
      "degree": 3048401
    },
    {
      "rank": 6,
      "track_uri": "spotify:track:4Km5HrUvYTaSUfiSGPJeQR",
      "degree": 2921634
    },
    {
      "rank": 7,
      "track_uri": "spotify:track:5hTpBe8h35rJ67eAWHQsJx",
      "degree": 2864453
    },
    {
      "rank": 8,
      "track_uri": "spotify:track:2EEeOnHehOozLq4aS0n6SL",
      "degree": 2794729
    },
    {
      "rank": 9,
      "track_uri": "spotify:track:0SGkqnVQo9KPytSri1H6cF",
      "degree": 2784654
    },
    {
      "rank": 10,
      "track_uri": "spotify:track:5dNfHmqgr128gMY2tc5CeJ",
      "degree": 2744196
    },
    {
      "rank": 11,
      "track_uri": "spotify:track:62vpWI1CHwFy7tMIcSStl8",
      "degree": 2730912
    },
    {
      "rank": 12,
      "track_uri": "spotify:track:5XJJdNPkwmbUwE79gv0NxK",
      "degree": 2724424
    },
    {
      "rank": 13,
      "track_uri": "spotify:track:7GX5flRQZVHRAGd6B4TmDO",
      "degree": 2722619
    },
    {
      "rank": 14,
      "track_uri": "spotify:track:0v9Wz8o0BT8DU38R4ddjeH",
      "degree": 2651832
    },
    {
      "rank": 15,
      "track_uri": "spotify:track:27GmP9AWRs744SzKcpJsTZ",
      "degree": 2618852
    },
    {
      "rank": 16,
      "track_uri": "spotify:track:0VgkVdmE4gld66l8iyGjgx",
      "degree": 2593459
    },
    {
      "rank": 17,
      "track_uri": "spotify:track:343YBumqHu19cGoGARUTsd",
      "degree": 2572375
    },
    {
      "rank": 18,
      "track_uri": "spotify:track:6gBFPUFcJLzWGx4lenP6h2",
      "degree": 2541545
    },
    {
      "rank": 19,
      "track_uri": "spotify:track:3DXncPQOG4VBw3QHh3S817",
      "degree": 2470489
    },
    {
      "rank": 20,
      "track_uri": "spotify:track:2d8JP84HNLKhmd6IYOoupQ",
      "degree": 2421312
    }
  ]
}
```

### 5.2 Temporal/Sequential Analysis

```json
{
  "playlist_patterns": {
    "mean_size": 66.346428,
    "median_size": 49.0,
    "std_size": 53.669357999148865,
    "min_size": 5,
    "max_size": 376,
    "total_playlists": 1000000
  },
  "top_tracks": [
    {
      "rank": 1,
      "track_uri": "spotify:track:7KXjTSCq5nL1LoYtL7XAwS",
      "total_occurrences": 46574,
      "num_playlists": 45394,
      "percentage": 0.07019820268244133
    },
    {
      "rank": 2,
      "track_uri": "spotify:track:1xznGGDReH1oQq0xzbwXa3",
      "total_occurrences": 43447,
      "num_playlists": 41707,
      "percentage": 0.0654850627376654
    },
    {
      "rank": 3,
      "track_uri": "spotify:track:7yyRTcZmCiyzzJlNzGC9Ol",
      "total_occurrences": 41309,
      "num_playlists": 40659,
      "percentage": 0.062262583299887674
    },
    {
      "rank": 4,
      "track_uri": "spotify:track:7BKLCZ1jbUBVqRi2FVlTVw",
      "total_occurrences": 41079,
      "num_playlists": 40629,
      "percentage": 0.06191591806570205
    },
    {
      "rank": 5,
      "track_uri": "spotify:track:3a1lNhkSLSkpJE4MSHpDu9",
      "total_occurrences": 39987,
      "num_playlists": 39577,
      "percentage": 0.06027001182339462
    },
    {
      "rank": 6,
      "track_uri": "spotify:track:5hTpBe8h35rJ67eAWHQsJx",
      "total_occurrences": 35202,
      "num_playlists": 34765,
      "percentage": 0.0530578677121849
    },
    {
      "rank": 7,
      "track_uri": "spotify:track:2EEeOnHehOozLq4aS0n6SL",
      "total_occurrences": 35138,
      "num_playlists": 34672,
      "percentage": 0.05296140434267238
    },
    {
      "rank": 8,
      "track_uri": "spotify:track:4Km5HrUvYTaSUfiSGPJeQR",
      "total_occurrences": 34999,
      "num_playlists": 34157,
      "percentage": 0.052751897962012365
    },
    {
      "rank": 9,
      "track_uri": "spotify:track:7GX5flRQZVHRAGd6B4TmDO",
      "total_occurrences": 34922,
      "num_playlists": 34048,
      "percentage": 0.05263584047056761
    },
    {
      "rank": 10,
      "track_uri": "spotify:track:152lZdxL1OR0ZMW6KquMif",
      "total_occurrences": 34657,
      "num_playlists": 33954,
      "percentage": 0.05223642183117982
    },
    {
      "rank": 11,
      "track_uri": "spotify:track:0SGkqnVQo9KPytSri1H6cF",
      "total_occurrences": 33699,
      "num_playlists": 32930,
      "percentage": 0.05079248576878924
    },
    {
      "rank": 12,
      "track_uri": "spotify:track:5dNfHmqgr128gMY2tc5CeJ",
      "total_occurrences": 32391,
      "num_playlists": 31509,
      "percentage": 0.04882101565437705
    },
    {
      "rank": 13,
      "track_uri": "spotify:track:62vpWI1CHwFy7tMIcSStl8",
      "total_occurrences": 32336,
      "num_playlists": 32143,
      "percentage": 0.04873811744620223
    },
    {
      "rank": 14,
      "track_uri": "spotify:track:0VgkVdmE4gld66l8iyGjgx",
      "total_occurrences": 32059,
      "num_playlists": 31884,
      "percentage": 0.048320611925030836
    },
    {
      "rank": 15,
      "track_uri": "spotify:track:0v9Wz8o0BT8DU38R4ddjeH",
      "total_occurrences": 31492,
      "num_playlists": 31340,
      "percentage": 0.04746600676075583
    },
    {
      "rank": 16,
      "track_uri": "spotify:track:3DXncPQOG4VBw3QHh3S817",
      "total_occurrences": 31374,
      "num_playlists": 30800,
      "percentage": 0.047288152423217114
    },
    {
      "rank": 17,
      "track_uri": "spotify:track:27GmP9AWRs744SzKcpJsTZ",
      "total_occurrences": 31119,
      "num_playlists": 30817,
      "percentage": 0.04690380618531566
    },
    {
      "rank": 18,
      "track_uri": "spotify:track:6gBFPUFcJLzWGx4lenP6h2",
      "total_occurrences": 31106,
      "num_playlists": 30948,
      "percentage": 0.046884212063383425
    },
    {
      "rank": 19,
      "track_uri": "spotify:track:343YBumqHu19cGoGARUTsd",
      "total_occurrences": 30678,
      "num_playlists": 29804,
      "percentage": 0.04623911327976843
    },
    {
      "rank": 20,
      "track_uri": "spotify:track:5CtI0qwDJkDQGwXD1H1cLb",
      "total_occurrences": 30485,
      "num_playlists": 29924,
      "percentage": 0.045948215931082226
    }
  ],
  "diversity_by_size": {
    "track_diversity": {
      "0-10": 0.9954,
      "11-25": 0.9945,
      "26-50": 0.9921,
      "51-100": 0.9889,
      "101-200": 0.9845,
      "200+": 0.9799
    },
    "artist_diversity": {
      "0-10": 0.8325,
      "11-25": 0.7532,
      "26-50": 0.6775,
      "51-100": 0.6079,
      "101-200": 0.5279,
      "200+": 0.4608
    },
    "size": {
      "0-10": 36008,
      "11-25": 196613,
      "26-50": 270928,
      "51-100": 275502,
      "101-200": 186054,
      "200+": 34895
    }
  },
  "top_artists": [
    {
      "rank": 1,
      "artist_uri": "spotify:artist:3TVXtAsR1Inumwj472S9r4",
      "total_tracks": 846937,
      "num_playlists": 203345
    },
    {
      "rank": 2,
      "artist_uri": "spotify:artist:5K4W6rqBFWDnAN6FQUkS6x",
      "total_tracks": 413297,
      "num_playlists": 141223
    },
    {
      "rank": 3,
      "artist_uri": "spotify:artist:2YZyLoL8N0Wb9xBt1NhZWg",
      "total_tracks": 353624,
      "num_playlists": 120901
    },
    {
      "rank": 4,
      "artist_uri": "spotify:artist:5pKCCKE2ajJHZ9KAiaK11H",
      "total_tracks": 339570,
      "num_playlists": 150344
    },
    {
      "rank": 5,
      "artist_uri": "spotify:artist:1Xyo4u8uXC1ZmMpatF05PJ",
      "total_tracks": 316603,
      "num_playlists": 125236
    },
    {
      "rank": 6,
      "artist_uri": "spotify:artist:7dGJo4pcD2V6oG8kP0tJRR",
      "total_tracks": 294667,
      "num_playlists": 76657
    },
    {
      "rank": 7,
      "artist_uri": "spotify:artist:6eUKZXaKkcviH0Ku9w2n3V",
      "total_tracks": 272116,
      "num_playlists": 111422
    },
    {
      "rank": 8,
      "artist_uri": "spotify:artist:1RyvyyTE3xzB2ZywiAwp0i",
      "total_tracks": 249986,
      "num_playlists": 96258
    },
    {
      "rank": 9,
      "artist_uri": "spotify:artist:1uNFoZAHBGtllmzznpCI3s",
      "total_tracks": 243119,
      "num_playlists": 92492
    },
    {
      "rank": 10,
      "artist_uri": "spotify:artist:6l3HvQ5sa6mXTsMTB19rO5",
      "total_tracks": 241556,
      "num_playlists": 85749
    },
    {
      "rank": 11,
      "artist_uri": "spotify:artist:6vWDO969PvNqNYHIOW5v0m",
      "total_tracks": 230857,
      "num_playlists": 97468
    },
    {
      "rank": 12,
      "artist_uri": "spotify:artist:69GGBxA162lTqCwzJG5jLp",
      "total_tracks": 223509,
      "num_playlists": 111562
    },
    {
      "rank": 13,
      "artist_uri": "spotify:artist:7bXgB6jMjp9ATFy66eO08Z",
      "total_tracks": 212751,
      "num_playlists": 95501
    },
    {
      "rank": 14,
      "artist_uri": "spotify:artist:7CajNmpbOovFoOoasH2HaY",
      "total_tracks": 203047,
      "num_playlists": 110881
    },
    {
      "rank": 15,
      "artist_uri": "spotify:artist:3YQKmKGau1PzlVlkL1iodx",
      "total_tracks": 198905,
      "num_playlists": 70699
    },
    {
      "rank": 16,
      "artist_uri": "spotify:artist:4O15NlyKLIASxsJ0PrXPfz",
      "total_tracks": 197855,
      "num_playlists": 65549
    },
    {
      "rank": 17,
      "artist_uri": "spotify:artist:246dkjvS1zLTtiykXe5h60",
      "total_tracks": 195907,
      "num_playlists": 93227
    },
    {
      "rank": 18,
      "artist_uri": "spotify:artist:0c173mlxpT3dSFRgMO8XPh",
      "total_tracks": 192473,
      "num_playlists": 97174
    },
    {
      "rank": 19,
      "artist_uri": "spotify:artist:04gDigrS5kc9YWfZHwBETP",
      "total_tracks": 187029,
      "num_playlists": 97256
    },
    {
      "rank": 20,
      "artist_uri": "spotify:artist:3nFkdlSjzX9mRTtwJOzDYB",
      "total_tracks": 185520,
      "num_playlists": 82530
    }
  ],
  "sequential_patterns": {
    "playlists_with_repetitions": 14621,
    "percentage": 29.242
  }
}
```

### 5.3 Genre Cross-Pollination

```json
{
  "genre_distribution": {},
  "multi_genre_analysis": {
    "mean_genres_per_playlist": 0.0,
    "median_genres_per_playlist": 0.0,
    "max_genres": 0,
    "multi_genre_playlists": 0,
    "multi_genre_percentage": 0.0
  },
  "top_genre_pairs": [],
  "genre_exclusivity": {},
  "correlations": {}
}
```

### 5.4 Recommendation Explainability

```json
{
  "sample_explanations": [
    {
      "type": "co-occurrence",
      "seed_track": "spotify:track:7DkzLcSXS4KhSvbvIJja3M",
      "recommended_track": "spotify:track:3b7CDTKB0SRTmQ6ytYi5vZ",
      "cooccurrence_count": 1131,
      "seed_total_connections": 185663,
      "rec_total_connections": 682042,
      "cooccurrence_strength": 0.006091682241480532,
      "explanation": "These tracks appeared together in 1,131 playlists. When 'spotify:track:7DkzLcSXS4KhSvbvIJja3M' is in a playlist, there's a 0.61% chance 'spotify:track:3b7CDTKB0SRTmQ6ytYi5vZ' is also there."
    },
    {
      "type": "co-occurrence",
      "seed_track": "spotify:track:3TxpOBmKTHuKD4mNdgkPYt",
      "recommended_track": "spotify:track:08zJpaUQVi9FrKv2e32Bah",
      "cooccurrence_count": 577,
      "seed_total_connections": 185193,
      "rec_total_connections": 1381031,
      "cooccurrence_strength": 0.003115668518788507,
      "explanation": "These tracks appeared together in 577 playlists. When 'spotify:track:3TxpOBmKTHuKD4mNdgkPYt' is in a playlist, there's a 0.31% chance 'spotify:track:08zJpaUQVi9FrKv2e32Bah' is also there."
    },
    {
      "type": "co-occurrence",
      "seed_track": "spotify:track:0LWQWOFoz5GJLqcHk1fRO2",
      "recommended_track": "spotify:track:2KpCpk6HjXXLb7nnXoXA5O",
      "cooccurrence_count": 4637,
      "seed_total_connections": 1402187,
      "rec_total_connections": 2310066,
      "cooccurrence_strength": 0.003306976886820374,
      "explanation": "These tracks appeared together in 4,637 playlists. When 'spotify:track:0LWQWOFoz5GJLqcHk1fRO2' is in a playlist, there's a 0.33% chance 'spotify:track:2KpCpk6HjXXLb7nnXoXA5O' is also there."
    },
    {
      "type": "co-occurrence",
      "seed_track": "spotify:track:01ZepwW5W3Z4fwl4bzaHyY",
      "recommended_track": "spotify:track:7nDoBWDvf02SyD8kEQuuPO",
      "cooccurrence_count": 1181,
      "seed_total_connections": 184126,
      "rec_total_connections": 596842,
      "cooccurrence_strength": 0.006414086006321758,
      "explanation": "These tracks appeared together in 1,181 playlists. When 'spotify:track:01ZepwW5W3Z4fwl4bzaHyY' is in a playlist, there's a 0.64% chance 'spotify:track:7nDoBWDvf02SyD8kEQuuPO' is also there."
    },
    {
      "type": "co-occurrence",
      "seed_track": "spotify:track:7uEcCGtM1FBBGIhPozhJjv",
      "recommended_track": "spotify:track:3G7tRC24Uh09Hmp1KZ7LQ2",
      "cooccurrence_count": 648,
      "seed_total_connections": 102437,
      "rec_total_connections": 354578,
      "cooccurrence_strength": 0.006325839296347999,
      "explanation": "These tracks appeared together in 648 playlists. When 'spotify:track:7uEcCGtM1FBBGIhPozhJjv' is in a playlist, there's a 0.63% chance 'spotify:track:3G7tRC24Uh09Hmp1KZ7LQ2' is also there."
    },
    {
      "type": "co-occurrence",
      "seed_track": "spotify:track:34FWzxRaGdAZyGQz0krlHF",
      "recommended_track": "spotify:track:4yugZvBYaoREkJKtbG08Qr",
      "cooccurrence_count": 621,
      "seed_total_connections": 142962,
      "rec_total_connections": 825483,
      "cooccurrence_strength": 0.004343811642254585,
      "explanation": "These tracks appeared together in 621 playlists. When 'spotify:track:34FWzxRaGdAZyGQz0krlHF' is in a playlist, there's a 0.43% chance 'spotify:track:4yugZvBYaoREkJKtbG08Qr' is also there."
    },
    {
      "type": "co-occurrence",
      "seed_track": "spotify:track:6V9kwssTrwkKT72imgowj9",
      "recommended_track": "spotify:track:6RUKPb4LETWmmr3iAEQktW",
      "cooccurrence_count": 1112,
      "seed_total_connections": 157710,
      "rec_total_connections": 1539297,
      "cooccurrence_strength": 0.007050916238665906,
      "explanation": "These tracks appeared together in 1,112 playlists. When 'spotify:track:6V9kwssTrwkKT72imgowj9' is in a playlist, there's a 0.71% chance 'spotify:track:6RUKPb4LETWmmr3iAEQktW' is also there."
    },
    {
      "type": "co-occurrence",
      "seed_track": "spotify:track:1M1HscO3JywTvswsVx1GcI",
      "recommended_track": "spotify:track:07KYRDFf8Q6sqj4PWCP9vh",
      "cooccurrence_count": 1013,
      "seed_total_connections": 211900,
      "rec_total_connections": 492237,
      "cooccurrence_strength": 0.004780556866446437,
      "explanation": "These tracks appeared together in 1,013 playlists. When 'spotify:track:1M1HscO3JywTvswsVx1GcI' is in a playlist, there's a 0.48% chance 'spotify:track:07KYRDFf8Q6sqj4PWCP9vh' is also there."
    },
    {
      "type": "co-occurrence",
      "seed_track": "spotify:track:3eze1OsZ1rqeXkKStNfTmi",
      "recommended_track": "spotify:track:0v9Wz8o0BT8DU38R4ddjeH",
      "cooccurrence_count": 5265,
      "seed_total_connections": 1082150,
      "rec_total_connections": 2651832,
      "cooccurrence_strength": 0.004865314420366862,
      "explanation": "These tracks appeared together in 5,265 playlists. When 'spotify:track:3eze1OsZ1rqeXkKStNfTmi' is in a playlist, there's a 0.49% chance 'spotify:track:0v9Wz8o0BT8DU38R4ddjeH' is also there."
    },
    {
      "type": "co-occurrence",
      "seed_track": "spotify:track:0EfsDEYaSjGYd66Pr881nq",
      "recommended_track": "spotify:track:5E30LdtzQTGqRvNd7l6kG5",
      "cooccurrence_count": 936,
      "seed_total_connections": 132593,
      "rec_total_connections": 195266,
      "cooccurrence_strength": 0.0070591961868273585,
      "explanation": "These tracks appeared together in 936 playlists. When 'spotify:track:0EfsDEYaSjGYd66Pr881nq' is in a playlist, there's a 0.71% chance 'spotify:track:5E30LdtzQTGqRvNd7l6kG5' is also there."
    }
  ],
  "recommendation_factors": {
    "mean_recommendation_entropy": 7.628754339269725,
    "median_recommendation_entropy": 7.735624261174664,
    "explanation": "Higher entropy means recommendations are more diverse/spread out. Lower entropy means recommendations are more concentrated on a few tracks."
  }
}
```
