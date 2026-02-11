import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE

from model.collaborative_filtering import (
    load_and_prepare_data, split_data, SVD, compute_mse,
)
from model.content_based_filtering import (
    ContentBasedModel, load_audio_features, AUDIO_FEATURE_COLS,
)
from model.hybrid_recommender import HybridRecommender


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    recs = recommended[:k]
    hits = len(set(recs) & relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    recs = recommended[:k]
    hits = len(set(recs) & relevant)
    return hits / len(relevant) if len(relevant) > 0 else 0.0


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    recs = recommended[:k]
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(recs)
        if item in relevant
    )
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def build_id_bridge(interaction_path, unique_songs_path):
    df, user_to_id, song_to_id = load_and_prepare_data(interaction_path)
    id_to_song = {v: k for k, v in song_to_id.items()}

    songs_df = pd.read_csv(unique_songs_path)
    track_id_to_meta = {
        row["track_id"]: (row["Artist"], row["Track"])
        for _, row in songs_df.iterrows()
    }
    meta_to_track_id = {v: k for k, v in track_id_to_meta.items()}

    return user_to_id, song_to_id, id_to_song, track_id_to_meta, meta_to_track_id


def get_user_items(split_df, id_to_song):
    result = {}
    for user_id in split_df["user_id"].unique():
        user_rows = split_df[split_df["user_id"] == user_id]
        items = set()
        for _, row in user_rows.iterrows():
            song = id_to_song.get(row["song_id"])
            if song:
                items.add(song)
        result[user_id] = items
    return result


def svd_top_k(model, user_id, train_items, id_to_song, k=10):
    all_items = np.arange(model.n_items)
    user_array = np.full(model.n_items, user_id)
    scores = model.predict(user_array, all_items)

    song_to_id = {v: k for k, v in id_to_song.items()}
    for item in train_items:
        sid = song_to_id.get(item)
        if sid is not None:
            scores[sid] = -np.inf

    top_indices = np.argsort(scores)[::-1][:k]
    return [id_to_song[idx] for idx in top_indices if idx in id_to_song]


def knn_top_k(knn_model, user_track_ids, play_counts, exclude_track_ids,
              track_id_to_meta, k=10):
    profile = knn_model.build_user_profile(user_track_ids, play_counts)
    if profile is None:
        return []

    n_neighbors = min(k + len(exclude_track_ids) + 50, len(knn_model.X_normalized))
    distances, indices = knn_model.knn.kneighbors(profile, n_neighbors=n_neighbors)

    results = []
    exclude = set(exclude_track_ids)
    for idx in indices[0]:
        tid = int(knn_model.features_df.iloc[idx]["track_id"])
        if tid in exclude:
            continue
        meta = track_id_to_meta.get(tid)
        if meta:
            results.append(meta)
        if len(results) >= k:
            break
    return results


def hybrid_top_k(recommender, user_vectors, k=10):
    results = recommender.recommend(
        user_feature_vectors=user_vectors,
        num_candidates=200,
        num_recommendations=k,
    )
    return [(r["artist"], r["track"]) for r in results]

def plot_training_curves(history, save_path="plots/svd_training_curves.png"):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_mse"]) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_mse"],
            label=f"Train MSE (final: {history['train_mse'][-1]:.4f})", marker="o", markersize=4)
    if history.get("val_mse"):
        ax.plot(epochs, history["val_mse"],
                label=f"Val MSE (final: {history['val_mse'][-1]:.4f})", marker="s", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("SVD Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_feature_distributions(features_df, save_path="plots/audio_feature_distributions.png"):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()

    for i, col in enumerate(AUDIO_FEATURE_COLS):
        sns.histplot(features_df[col], ax=axes[i], kde=True, bins=50, color="steelblue")
        axes[i].set_title(col, fontsize=11)
        axes[i].set_xlabel("")

    # Hide unused subplots
    for j in range(len(AUDIO_FEATURE_COLS), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Audio Feature Distributions", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_correlation_heatmap(features_df, save_path="plots/feature_correlation_heatmap.png"):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    corr = features_df[AUDIO_FEATURE_COLS].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, ax=ax, linewidths=0.5)
    ax.set_title("Audio Feature Correlation Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_tsne_embeddings(X_normalized, features_df,
                         save_path="plots/tsne_song_embeddings.png",
                         sample_size=5000, random_state=42):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(X_normalized)
    if n > sample_size:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n, sample_size, replace=False)
        X_sample = X_normalized[idx]
        mode_sample = features_df["mode"].values[idx]
    else:
        X_sample = X_normalized
        mode_sample = features_df["mode"].values

    print(f"  Running t-SNE on {len(X_sample)} songs...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=random_state, max_iter=1000)
    embedding = tsne.fit_transform(X_sample)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#e74c3c" if m == 0 else "#3498db" for m in mode_sample]
    ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, alpha=0.3, s=5)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=8, label="Minor"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db", markersize=8, label="Major"),
    ]
    ax.legend(handles=legend_elements, fontsize=11)
    ax.set_title(f"t-SNE of Song Audio Features ({len(X_sample)} songs)", fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_model_comparison(results, k_values=(5, 10, 20),
                          save_path="plots/model_comparison.png"):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = ["precision", "recall", "ndcg"]
    metric_labels = ["Precision@k", "Recall@k", "NDCG@k"]

    models = []
    model_colors = {"svd": "#3498db", "knn": "#2ecc71", "hybrid": "#e67e22"}
    for name in ["svd", "knn", "hybrid"]:
        if results.get(name) is not None:
            models.append(name)

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric, label in zip(axes, metrics, metric_labels):
        x = np.arange(len(k_values))
        width = 0.8 / len(models)

        for i, model_name in enumerate(models):
            values = [results[model_name].get(f"{metric}@{k}", 0) for k in k_values]
            offset = (i - len(models) / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=model_name.upper(),
                   color=model_colors[model_name], alpha=0.85)

        ax.set_xlabel("k")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in k_values])
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Model Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def run_full_evaluation(data_dir, k_values=(5, 10, 20), svd_epochs=20):
    """
    Run the complete evaluation pipeline.

    Returns dict with per-model metrics, SVD training history, and per-user breakdown.
    """
    data_dir = Path(data_dir)
    interaction_path = data_dir / "processed" / "user_song_interaction.csv"
    unique_songs_path = data_dir / "processed" / "unique_song_interaction.csv"
    features_path = data_dir / "processed" / "audio_features.csv"
    svd_model_path = data_dir / "models" / "svd_model.npz"

    print("=" * 60)
    print("  Loading data and building ID mappings")
    print("=" * 60)
    user_to_id, song_to_id, id_to_song, track_id_to_meta, meta_to_track_id = \
        build_id_bridge(str(interaction_path), str(unique_songs_path))

    df, _, _ = load_and_prepare_data(str(interaction_path))
    train, val, test = split_data(df, seed=42)

    train_items_per_user = get_user_items(train, id_to_song)
    test_items_per_user = get_user_items(test, id_to_song)

    print("\n" + "=" * 60)
    print("  Training SVD model")
    print("=" * 60)
    n_users = len(user_to_id)
    n_items = len(song_to_id)
    svd_model = SVD(n_users=n_users, n_items=n_items, n_factors=20, lr=0.005, reg=0.02)
    svd_model.song_to_id = song_to_id
    history = svd_model.fit(train, val=val, n_epochs=svd_epochs, verbose=True)

    knn_model = None
    hybrid_model = None
    features_df = None
    X_normalized = None

    if features_path.exists():
        print("\n" + "=" * 60)
        print("  Loading audio features and fitting KNN")
        print("=" * 60)
        features_df, X_normalized, scaler = load_audio_features(str(features_path))
        knn_model = ContentBasedModel(n_neighbors=200, metric="cosine")
        knn_model.fit(features_df, X_normalized, scaler)

        print("\n  Setting up Hybrid Recommender...")
        hybrid_model = HybridRecommender(
            features_path=features_path,
            songs_path=unique_songs_path,
            svd_model_path=svd_model_path if svd_model_path.exists() else None,
        )
    else:
        print("\n  audio_features.csv not found — skipping KNN and Hybrid evaluation")

    print("\n" + "=" * 60)
    print("  Computing ranking metrics")
    print("=" * 60)

    results = {
        "svd": {},
        "knn": None,
        "hybrid": None,
        "svd_history": history,
        "features_df": features_df,
        "X_normalized": X_normalized,
        "per_user": {},
    }

    if knn_model is not None:
        results["knn"] = {}
    if hybrid_model is not None:
        results["hybrid"] = {}

    id_to_user = {v: k for k, v in user_to_id.items()}

    for user_id in sorted(train_items_per_user.keys()):
        user_name = id_to_user.get(user_id, f"user_{user_id}")
        train_items = train_items_per_user.get(user_id, set())
        test_items = test_items_per_user.get(user_id, set())

        if len(test_items) == 0:
            continue

        results["per_user"][user_name] = {}

        max_k = max(k_values)
        svd_recs = svd_top_k(svd_model, user_id, train_items, id_to_song, k=max_k)
        results["per_user"][user_name]["svd"] = {}
        for k in k_values:
            results["per_user"][user_name]["svd"][f"precision@{k}"] = precision_at_k(svd_recs, test_items, k)
            results["per_user"][user_name]["svd"][f"recall@{k}"] = recall_at_k(svd_recs, test_items, k)
            results["per_user"][user_name]["svd"][f"ndcg@{k}"] = ndcg_at_k(svd_recs, test_items, k)

        if knn_model is not None:
            user_train_track_ids = []
            user_train_play_counts = []
            user_rows = train[train["user_id"] == user_id]
            for _, row in user_rows.iterrows():
                song_tuple = id_to_song.get(row["song_id"])
                if song_tuple:
                    tid = meta_to_track_id.get(song_tuple)
                    if tid is not None and tid in knn_model.track_id_to_idx:
                        user_train_track_ids.append(tid)
                        user_train_play_counts.append(row["play_count"])

            knn_recs = knn_top_k(
                knn_model, user_train_track_ids, user_train_play_counts,
                set(user_train_track_ids), track_id_to_meta, k=max_k,
            )

            results["per_user"][user_name]["knn"] = {}
            for k in k_values:
                results["per_user"][user_name]["knn"][f"precision@{k}"] = precision_at_k(knn_recs, test_items, k)
                results["per_user"][user_name]["knn"][f"recall@{k}"] = recall_at_k(knn_recs, test_items, k)
                results["per_user"][user_name]["knn"][f"ndcg@{k}"] = ndcg_at_k(knn_recs, test_items, k)

        if hybrid_model is not None and len(user_train_track_ids) > 0:
            user_vectors = []
            for tid in user_train_track_ids:
                if tid in knn_model.track_id_to_idx:
                    idx = knn_model.track_id_to_idx[tid]
                    raw_vec = knn_model.features_df[AUDIO_FEATURE_COLS].iloc[idx].values
                    user_vectors.append(raw_vec)

            if len(user_vectors) > 0:
                hybrid_recs = hybrid_top_k(hybrid_model, user_vectors, k=max_k)
                results["per_user"][user_name]["hybrid"] = {}
                for k in k_values:
                    results["per_user"][user_name]["hybrid"][f"precision@{k}"] = precision_at_k(hybrid_recs, test_items, k)
                    results["per_user"][user_name]["hybrid"][f"recall@{k}"] = recall_at_k(hybrid_recs, test_items, k)
                    results["per_user"][user_name]["hybrid"][f"ndcg@{k}"] = ndcg_at_k(hybrid_recs, test_items, k)

        print(f"  {user_name}: {len(test_items)} test items")

    for model_name in ["svd", "knn", "hybrid"]:
        if results[model_name] is None:
            continue
        metric_sums = {}
        count = 0
        for user_name, user_data in results["per_user"].items():
            if model_name in user_data:
                count += 1
                for metric_key, value in user_data[model_name].items():
                    metric_sums[metric_key] = metric_sums.get(metric_key, 0) + value
        if count > 0:
            for metric_key in metric_sums:
                results[model_name][metric_key] = metric_sums[metric_key] / count

    print("\n" + "=" * 60)
    print("  Evaluation Results (averaged across users)")
    print("=" * 60)

    for model_name in ["svd", "knn", "hybrid"]:
        if results[model_name] is None:
            continue
        print(f"\n  {model_name.upper()}:")
        for k in k_values:
            p = results[model_name].get(f"precision@{k}", 0)
            r = results[model_name].get(f"recall@{k}", 0)
            n = results[model_name].get(f"ndcg@{k}", 0)
            print(f"    @{k:>2}  Precision: {p:.4f}  Recall: {r:.4f}  NDCG: {n:.4f}")

    return results


def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"

    results = run_full_evaluation(data_dir)

    print("\n" + "=" * 60)
    print("  Generating visualizations")
    print("=" * 60)

    plots_dir = project_root / "plots"

    plot_training_curves(results["svd_history"], save_path=plots_dir / "svd_training_curves.png")

    if results["features_df"] is not None:
        plot_feature_distributions(results["features_df"], save_path=plots_dir / "audio_feature_distributions.png")
        plot_correlation_heatmap(results["features_df"], save_path=plots_dir / "feature_correlation_heatmap.png")
        plot_tsne_embeddings(
            results["X_normalized"], results["features_df"],
            save_path=plots_dir / "tsne_song_embeddings.png",
        )

    plot_model_comparison(results, save_path=plots_dir / "model_comparison.png")

    print("\n  Done!")


if __name__ == "__main__":
    main()
