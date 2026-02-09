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


