import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


AUDIO_FEATURE_COLS = [
    "energy", "tempo", "valence", "danceability", "acousticness",
    "instrumentalness", "liveness", "speechiness", "loudness",
    "mode", "key", "duration_ms", "time_signature",
]

def load_audio_features(features_path: str) -> tuple[pd.DataFrame, np.ndarray, StandardScaler]:

    features_df = pd.read_csv(features_path)
    print(f"Loaded audio features for {len(features_df)} songs")

    X = features_df[AUDIO_FEATURE_COLS].values

    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"  Warning: {nan_count} NaN values found, filling with column means")
        col_means = np.nanmean(X, axis=0)
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = col_means[j]

    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    print(f"  Feature matrix shape: {X_normalized.shape}")

    return features_df, X_normalized, scaler

class ContentBasedModel:
    def __init__(self, n_neighbors: int = 20, metric: str = "cosine"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.knn = None
        self.features_df = None
        self.X_normalized = None
        self.scaler = None
        self.track_id_to_idx = {}

    def fit(self, features_df: pd.DataFrame, X_normalized: np.ndarray, scaler: StandardScaler):
        self.features_df = features_df
        self.X_normalized = X_normalized
        self.scaler = scaler

        self.track_id_to_idx = {
            tid: idx for idx, tid in enumerate(features_df["track_id"].values)
        }

        self.knn = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            algorithm="brute",
        )
        self.knn.fit(X_normalized)

        print(f"  KNN fitted: {X_normalized.shape[0]} songs, "
              f"k={self.n_neighbors}, metric={self.metric}")

    def build_user_profile(
        self,
        user_track_ids: list[int],
        play_counts: list[int] | None = None,
    ) -> np.ndarray | None:
        valid_indices = []
        valid_weights = []

        for i, tid in enumerate(user_track_ids):
            if tid in self.track_id_to_idx:
                valid_indices.append(self.track_id_to_idx[tid])
                if play_counts is not None:
                    valid_weights.append(play_counts[i])

        if len(valid_indices) == 0:
            return None

        vectors = self.X_normalized[valid_indices]

        if play_counts is not None and valid_weights:
            weights = np.array(valid_weights, dtype=np.float64)
            weights = weights / weights.sum()
            profile = np.average(vectors, axis=0, weights=weights)
        else:
            profile = np.mean(vectors, axis=0)

        return profile.reshape(1, -1)

    def recommend(
        self,
        user_track_ids: list[int],
        play_counts: list[int] | None = None,
        n_recommendations: int = 10,
    ) -> pd.DataFrame | None:

        profile = self.build_user_profile(user_track_ids, play_counts)
        if profile is None:
            return None

        exclude_ids = set(user_track_ids)

        k = min(n_recommendations + len(exclude_ids) + 10, len(self.X_normalized))
        distances, indices = self.knn.kneighbors(profile, n_neighbors=k)

        recs = []

        for dist, idx in zip(distances[0], indices[0]):
            tid = int(self.features_df.iloc[idx]["track_id"])
            if tid not in exclude_ids:
                recs.append({"track_id": tid, "distance": float(dist)})
            if len(recs) >= n_recommendations:
                break

        recs_df = pd.DataFrame(recs)

        songs_path = Path(__file__).parent.parent / "data" / "processed" / "unique_song_interaction.csv"
        song_meta = pd.read_csv(songs_path)
        recs_df = recs_df.merge(song_meta[["track_id", "Artist", "Track"]], on="track_id", how="left")

        return recs_df[["track_id", "Artist", "Track", "distance"]]

def main():
    project_root = Path(__file__).parent.parent
    features_path = project_root / "data" / "processed" / "audio_features.csv"
    interactions_path = project_root / "data" / "processed" / "user_song_interaction.csv"
    songs_path = project_root / "data" / "processed" / "unique_song_interaction.csv"

    print("=" * 60)
    print("  STEP 1: Loading audio features")
    print("=" * 60)
    features_df, X_normalized, scaler = load_audio_features(str(features_path))

    print("\n" + "=" * 60)
    print("  STEP 2: Fitting content-based model")
    print("=" * 60)
    model = ContentBasedModel(n_neighbors=20, metric="cosine")
    model.fit(features_df, X_normalized, scaler)

    print("\n" + "=" * 60)
    print("  STEP 3: Loading user interactions")
    print("=" * 60)
    interactions = pd.read_csv(interactions_path)
    songs = pd.read_csv(songs_path)
    print(f"  Loaded {len(interactions)} interactions for {interactions['user'].nunique()} users")

    song_to_track_id = {
        (row["Artist"], row["Track"]): row["track_id"]
        for _, row in songs.iterrows()
    }
    interactions["track_id"] = interactions.apply(
        lambda r: song_to_track_id.get((r["artist"], r["song"])), axis=1
    )
    matched = interactions["track_id"].notna().sum()
    print(f"  Matched {matched}/{len(interactions)} interactions to track_ids")

    print("\n" + "=" * 60)
    print("  STEP 4: Generating recommendations")
    print("=" * 60)

    for user_name in sorted(interactions["user"].unique()):
        user_data = interactions[interactions["user"] == user_name]
        user_tracks = user_data[user_data["track_id"].notna()]
        user_track_ids = user_tracks["track_id"].astype(int).tolist()
        user_play_counts = user_tracks["play_count"].tolist()

        recs = model.recommend(
            user_track_ids=user_track_ids,
            play_counts=user_play_counts,
            n_recommendations=10,
        )

        print(f"\n  Recommendations for {user_name} "
              f"({len(user_track_ids)} songs in history):")
        if recs is not None and len(recs) > 0:
            for _, row in recs.iterrows():
                print(f"    {row['Artist']} - {row['Track']}  "
                      f"(distance: {row['distance']:.4f})")
        else:
            print("    No recommendations (no audio features for listened songs)")


if __name__ == "__main__":
    main()
