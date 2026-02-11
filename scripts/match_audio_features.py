
import pandas as pd
from pathlib import Path


AUDIO_FEATURE_COLS = [
    "energy", "tempo", "valence", "danceability", "acousticness",
    "instrumentalness", "liveness", "speechiness", "loudness",
    "mode", "key", "duration_ms", "time_signature",
]


def main():
    project_root = Path(__file__).parent.parent
    our_songs_path = project_root / "data" / "processed" / "unique_song_interaction.csv"
    kaggle_path = project_root / "data" / "raw" / "song_audio_features.csv"
    output_path = project_root / "data" / "processed" / "audio_features.csv"

    our_songs = pd.read_csv(our_songs_path)
    print(f"Our songs: {len(our_songs)}")

    kaggle = pd.read_csv(kaggle_path)
    print(f"Kaggle songs: {len(kaggle)}")

    kaggle["artist_lower"] = kaggle["artists"].str.split(";").str[0].str.strip().str.lower()
    kaggle["track_lower"] = kaggle["track_name"].str.strip().str.lower()
    kaggle["match_key"] = kaggle["artist_lower"] + "|||" + kaggle["track_lower"]

    kaggle_dedup = kaggle.drop_duplicates(subset="match_key", keep="first")
    kaggle_lookup = kaggle_dedup.set_index("match_key")
    print(f"Kaggle unique (artist, track): {len(kaggle_lookup)}")

    our_songs["artist_lower"] = our_songs["Artist"].str.strip().str.lower()
    our_songs["track_lower"] = our_songs["Track"].str.strip().str.lower()
    our_songs["match_key"] = our_songs["artist_lower"] + "|||" + our_songs["track_lower"]

    matched_rows = []
    for _, row in our_songs.iterrows():
        key = row["match_key"]
        if key in kaggle_lookup.index:
            kaggle_row = kaggle_lookup.loc[key]
            feature_row = {"track_id": row["track_id"]}
            for col in AUDIO_FEATURE_COLS:
                feature_row[col] = kaggle_row[col]
            matched_rows.append(feature_row)

    features_df = pd.DataFrame(matched_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)

    print(f"\nMatched: {len(features_df)} / {len(our_songs)} songs "
          f"({len(features_df) / len(our_songs) * 100:.1f}%)")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
