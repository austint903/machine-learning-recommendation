import numpy as np
import pandas as pd
from pathlib import Path

from model.collaborative_filtering import SVD
from model.content_based_filtering import ContentBasedModel, load_audio_features, AUDIO_FEATURE_COLS


class HybridRecommender:
    """
    Two-stage hybrid recommender:
      Stage 1 (KNN via ContentBasedModel): Generate candidates based on audio feature similarity
      Stage 2 (SVD): Rank candidates using collaborative filtering item biases
    """

    def __init__(self, features_path, songs_path, svd_model_path=None):
        features_df, X_normalized, scaler = load_audio_features(features_path)
        self.knn_model = ContentBasedModel(n_neighbors=200, metric="cosine")
        self.knn_model.fit(features_df, X_normalized, scaler)

        song_meta = pd.read_csv(songs_path)
        self._track_id_to_meta = {
            row["track_id"]: (row["Artist"], row["Track"])
            for _, row in song_meta.iterrows()
        }

        self.svd = None
        if svd_model_path and Path(svd_model_path).exists():
            self.svd = SVD.load(svd_model_path)
            print(f"  Hybrid mode: KNN candidates → SVD ranking")
        else:
            print(f"  KNN-only mode: no SVD model found, ranking by cosine distance")

    def recommend(self, user_feature_vectors, num_candidates=100, num_recommendations=10):
        user_normalized = self.knn_model.scaler.transform(np.array(user_feature_vectors))
        profile = np.mean(user_normalized, axis=0).reshape(1, -1)

        k = min(num_candidates, len(self.knn_model.X_normalized))
        distances, indices = self.knn_model.knn.kneighbors(profile, n_neighbors=k)

        candidates = []
        for dist, idx in zip(distances[0], indices[0]):
            track_id = int(self.knn_model.features_df.iloc[idx]["track_id"])
            candidates.append({
                "track_id": track_id,
                "knn_distance": float(dist),
            })

        if self.svd is not None:
            for c in candidates:
                song_id = self._map_track_to_song_id(c["track_id"])
                if song_id is not None:
                    c["svd_score"] = self.svd.global_mean + self.svd.b_i[song_id]
                else:
                    c["svd_score"] = self.svd.global_mean

            candidates.sort(key=lambda c: c["svd_score"], reverse=True)
        else:
            candidates.sort(key=lambda c: c["knn_distance"])

        results = []
        for i, c in enumerate(candidates[:num_recommendations]):
            artist, track = self._get_song_meta(c["track_id"])

            results.append({
                "rank": i + 1,
                "artist": artist,
                "track": track,
                "svd_score": c.get("svd_score"),
                "knn_distance": c["knn_distance"],
            })

        return results

    def _map_track_to_song_id(self, track_id):
        artist, track = self._get_song_meta(track_id)
        if artist is None:
            return None

        song_id = self.svd.song_to_id.get((artist, track))
        if song_id is None:
            song_id = self.svd.song_to_id.get(
                (artist.lower() if isinstance(artist, str) else artist,
                 track.lower() if isinstance(track, str) else track)
            )
        return song_id

    def _get_song_meta(self, track_id):
        """Look up (Artist, Track) for a track_id."""
        meta = self._track_id_to_meta.get(track_id)
        if meta is None:
            return None, None
        return meta


def main():
    project_root = Path(__file__).parent.parent
    features_path = project_root / "data" / "processed" / "audio_features.csv"
    songs_path = project_root / "data" / "processed" / "unique_song_interaction.csv"
    svd_model_path = project_root / "data" / "models" / "svd_model.npz"

    print("=" * 60)
    print("  Hybrid Recommender — Standalone Test")
    print("=" * 60)

    recommender = HybridRecommender(
        features_path=features_path,
        songs_path=songs_path,
        svd_model_path=svd_model_path,
    )

    sample_idx = 0
    sample_vector = recommender.knn_model.features_df[AUDIO_FEATURE_COLS].iloc[sample_idx].values
    sample_tid = int(recommender.knn_model.features_df.iloc[sample_idx]["track_id"])
    sample_meta = recommender._get_song_meta(sample_tid)

    print(f"\n  Test song: {sample_meta[0]} - {sample_meta[1]}")
    print(f"  Generating recommendations...\n")

    results = recommender.recommend(
        user_feature_vectors=[sample_vector],
        num_candidates=100,
        num_recommendations=10,
    )

    for r in results:
        line = f"  {r['rank']:>2}. {r['artist']} - {r['track']}"
        if r["svd_score"] is not None:
            line += f"  (SVD: {r['svd_score']:.4f}, KNN dist: {r['knn_distance']:.4f})"
        else:
            line += f"  (KNN dist: {r['knn_distance']:.4f})"
        print(line)


if __name__ == "__main__":
    main()
