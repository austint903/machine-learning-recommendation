import numpy as np
import pandas as pd
from pathlib import Path



#load data
def load_and_prepare_data(csv_path: str) -> tuple[pd.DataFrame, dict, dict]:
    """
    Load user-song interaction data and create integer ID mappings.

    Returns:
        df: DataFrame with columns [user_id, song_id, play_count]
        user_to_id: mapping from username -> integer id
        song_to_id: mapping from (artist, song) -> integer id
    """
    raw = pd.read_csv(csv_path)
    print(f"Loaded {len(raw)} interactions")

    # Create integer ID mappings
    unique_users = raw["user"].unique()
    unique_songs = raw[["artist", "song"]].drop_duplicates()

    user_to_id = {name: i for i, name in enumerate(unique_users)}
    song_to_id = {
        (row["artist"], row["song"]): i
        for i, row in unique_songs.reset_index(drop=True).iterrows()
    }

    # Build clean dataframe with integer IDs
    df = pd.DataFrame({
        "user_id": raw["user"].map(user_to_id),
        "song_id": raw.apply(lambda r: song_to_id[(r["artist"], r["song"])], axis=1),
        "play_count": raw["play_count"].values,
    })

    print(f"  Users:  {len(user_to_id)}")
    print(f"  Songs:  {len(song_to_id)}")
    print(f"  Density: {len(df) / (len(user_to_id) * len(song_to_id)) * 100:.2f}%")

    return df, user_to_id, song_to_id

