import time
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from pathlib import Path
from dotenv import load_dotenv


AUDIO_FEATURE_COLS = [
    "energy", "tempo", "valence", "danceability", "acousticness",
    "instrumentalness", "liveness", "speechiness", "loudness",
    "mode", "key", "duration_ms", "time_signature",
]

BATCH_SIZE = 50
SEARCH_DELAY = 0.05
CHECKPOINT_INTERVAL = 500



#get spotify track ids to get features later
def search_spotify_ids(sp, songs_df, checkpoint_path):
    """
    For each (Artist, Track) in songs_df, search Spotify and record the
    best-match spotify track ID. Saves checkpoint CSV incrementally.

    Returns DataFrame with columns [track_id, spotify_id].
    """
    # Resume from checkpoint if it exists
    if checkpoint_path.exists():
        done = pd.read_csv(checkpoint_path)
        done_ids = set(done["track_id"].tolist())
        results = done.to_dict("records")
        print(f"  Resuming from checkpoint: {len(done_ids)} songs already searched")
    else:
        results = []
        done_ids = set()

    remaining = songs_df[~songs_df["track_id"].isin(done_ids)]
    total_remaining = len(remaining)
    print(f"  Remaining songs to search: {total_remaining}")

    for i, row in enumerate(remaining.itertuples()):
        track_id = row.track_id
        artist = row.Artist
        track = row.Track

        query = f"artist:{artist} track:{track}"
        spotify_id = None

        try:
            result = sp.search(q=query, type="track", limit=1)
            items = result["tracks"]["items"]
            if items:
                spotify_id = items[0]["id"]
        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 429:
                retry_after = int(e.headers.get("Retry-After", 5))
                print(f"  Rate limited. Sleeping {retry_after}s...")
                time.sleep(retry_after)
                try:
                    result = sp.search(q=query, type="track", limit=1)
                    items = result["tracks"]["items"]
                    if items:
                        spotify_id = items[0]["id"]
                except Exception:
                    pass
            else:
                print(f"  Spotify error for track_id={track_id}: {e}")
        except Exception as e:
            print(f"  Unexpected error for track_id={track_id}: {e}")

        results.append({"track_id": track_id, "spotify_id": spotify_id})

        # Periodic checkpoint
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            pd.DataFrame(results).to_csv(checkpoint_path, index=False)
            found = sum(1 for r in results if r["spotify_id"] is not None)
            print(f"  Checkpoint: {i+1}/{total_remaining} searched, {found} found so far")

        time.sleep(SEARCH_DELAY)

    # Final save
    search_df = pd.DataFrame(results)
    search_df.to_csv(checkpoint_path, index=False)
    return search_df


#grab audio features for each song id 
def fetch_audio_features_batch(sp, search_df):
    """
    Given search_df with columns [track_id, spotify_id], fetch audio features
    in batches of BATCH_SIZE.

    Returns DataFrame with track_id + 13 audio feature columns.
    """
    found = search_df[search_df["spotify_id"].notna()].copy()
    print(f"  Songs with Spotify match: {len(found)} / {len(search_df)}")

    all_features = []
    spotify_ids = found["spotify_id"].tolist()
    track_ids = found["track_id"].tolist()

    for start in range(0, len(spotify_ids), BATCH_SIZE):
        batch_spotify = spotify_ids[start:start + BATCH_SIZE]
        batch_track = track_ids[start:start + BATCH_SIZE]

        try:
            features_list = sp.audio_features(batch_spotify)
        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 429:
                retry_after = int(e.headers.get("Retry-After", 5))
                print(f"  Rate limited. Sleeping {retry_after}s...")
                time.sleep(retry_after)
                features_list = sp.audio_features(batch_spotify)
            elif e.http_status == 403:
                print(f"  ERROR: 403 Forbidden on audio_features endpoint.")
                print(f"  Spotify deprecated this endpoint for apps created after Nov 2024.")
                print(f"  Aborting feature fetch.")
                return pd.DataFrame(columns=["track_id"] + AUDIO_FEATURE_COLS)
            else:
                print(f"  Error on batch starting at {start}: {e}")
                features_list = [None] * len(batch_spotify)

        for tid, feat in zip(batch_track, features_list):
            if feat is not None:
                row = {"track_id": tid}
                for col in AUDIO_FEATURE_COLS:
                    row[col] = feat.get(col)
                all_features.append(row)

        processed = min(start + len(batch_spotify), len(spotify_ids))
        if (start // BATCH_SIZE) % 20 == 0:
            print(f"  Fetched features: {processed} / {len(spotify_ids)}")

        time.sleep(0.1)

    return pd.DataFrame(all_features)


def main():
    project_root = Path(__file__).parent.parent
    load_dotenv(project_root / ".env")

    songs_path = project_root / "data" / "processed" / "unique_song_interaction.csv"
    checkpoint_path = project_root / "data" / "processed" / "_spotify_search_checkpoint.csv"
    output_path = project_root / "data" / "processed" / "audio_features.csv"

    # Initialize Spotify client
    sp = spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(),
        requests_timeout=10,
        retries=3,
    )

    songs_df = pd.read_csv(songs_path)
    print(f"Loaded {len(songs_df)} unique songs")

    # Phase 1: Search
    print("\n" + "=" * 60)
    print("  PHASE 1: Searching Spotify for track IDs")
    print("=" * 60)
    search_df = search_spotify_ids(sp, songs_df, checkpoint_path)

    found_count = search_df["spotify_id"].notna().sum()
    miss_count = search_df["spotify_id"].isna().sum()
    print(f"\n  Search complete: {found_count} found, {miss_count} not found")
    print(f"  Hit rate: {found_count / len(search_df) * 100:.1f}%")

    # Phase 2: Batch fetch audio features
    print("\n" + "=" * 60)
    print("  PHASE 2: Fetching audio features in batches")
    print("=" * 60)
    features_df = fetch_audio_features_batch(sp, search_df)

    # Phase 3: Save
    print("\n" + "=" * 60)
    print("  PHASE 3: Saving audio features")
    print("=" * 60)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)
    print(f"  Saved {len(features_df)} songs with audio features to {output_path}")
    print(f"\n  Coverage: {len(features_df)} / {len(songs_df)} songs "
          f"({len(features_df) / len(songs_df) * 100:.1f}%)")
    print(f"  Features per song: {len(AUDIO_FEATURE_COLS)}")


if __name__ == "__main__":
    main()
