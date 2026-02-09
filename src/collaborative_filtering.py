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


def split_data(
    df: pd.DataFrame,
    train_frac: float = 0.75,
    val_frac: float = 0.125,
    test_frac: float = 0.125,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Randomly split interaction data into train / validation / test sets.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9, "Fractions must sum to 1"

    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train = shuffled.iloc[:train_end]
    val = shuffled.iloc[train_end:val_end]
    test = shuffled.iloc[val_end:]

    print(f"\nData split (seed={seed}):")
    print(f"  Train:      {len(train):>7} ({len(train)/n*100:.1f}%)")
    print(f"  Validation: {len(val):>7} ({len(val)/n*100:.1f}%)")
    print(f"  Test:       {len(test):>7} ({len(test)/n*100:.1f}%)")

    return train, val, test

