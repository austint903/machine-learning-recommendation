import json

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

#split data
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

#svd
class SVD:
    """
    SVD matrix factorization for collaborative filtering, trained with SGD.

    Predicts: r_hat(u, i) = global_mean + b_u + b_i + p_u . q_i

    Where:
        - global_mean: average rating across all training interactions
        - b_u: learned bias for user u
        - b_i: learned bias for item i
        - p_u: latent factor vector for user u  (shape: k)
        - q_i: latent factor vector for item i  (shape: k)
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int = 20,
        lr: float = 0.005,
        reg: float = 0.02,
        seed: int = 42,
    ):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg

        rng = np.random.RandomState(seed)

        # Initialize latent factor matrices with small random values
        self.P = rng.normal(0, 0.1, (n_users, n_factors))   # user factors
        self.Q = rng.normal(0, 0.1, (n_items, n_factors))   # item factors

        # Initialize biases to zero
        self.b_u = np.zeros(n_users)   # user biases
        self.b_i = np.zeros(n_items)   # item biases

        self.global_mean = 0.0
        self.song_to_id = {}  # set externally after construction

    def save(self, path: str | Path):
        """Save model parameters and ID mappings to a .npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            P=self.P,
            Q=self.Q,
            b_u=self.b_u,
            b_i=self.b_i,
            global_mean=np.array([self.global_mean]),
        )
        # Save song_to_id mapping as JSON alongside the .npz
        mapping_path = path.with_suffix(".json")
        with open(mapping_path, "w") as f:
            # Convert (artist, song) tuple keys to "artist|||song" strings
            serializable = {f"{k[0]}|||{k[1]}": v for k, v in self.song_to_id.items()}
            json.dump(serializable, f)
        print(f"  Model saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "SVD":
        """Load a pre-trained SVD model from a .npz file."""
        path = Path(path)
        data = np.load(path)
        P = data["P"]
        Q = data["Q"]

        model = cls(n_users=P.shape[0], n_items=Q.shape[0], n_factors=P.shape[1])
        model.P = P
        model.Q = Q
        model.b_u = data["b_u"]
        model.b_i = data["b_i"]
        model.global_mean = float(data["global_mean"][0])

        # Load song_to_id mapping
        mapping_path = path.with_suffix(".json")
        with open(mapping_path) as f:
            raw = json.load(f)
            model.song_to_id = {
                tuple(k.split("|||", 1)): v for k, v in raw.items()
            }

        print(f"  Model loaded from {path}")
        return model

    def predict_one(self, u: int, i: int) -> float:
        """Predict rating for a single (user, item) pair."""
        return self.global_mean + self.b_u[u] + self.b_i[i] + self.P[u] @ self.Q[i]

    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """Predict ratings for arrays of (user, item) pairs."""
        preds = (
            self.global_mean
            + self.b_u[user_ids]
            + self.b_i[item_ids]
            + np.sum(self.P[user_ids] * self.Q[item_ids], axis=1)
        )
        return preds

    def fit(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame | None = None,
        n_epochs: int = 20,
        verbose: bool = True,
    ) -> dict:
        """
        Train the SVD model using SGD on the training set.

        Args:
            train: DataFrame with columns [user_id, song_id, play_count]
            val: optional validation DataFrame (same schema)
            n_epochs: number of passes over the training data
            verbose: print progress each epoch

        Returns:
            history dict with train_mse and val_mse per epoch
        """
        self.global_mean = train["play_count"].mean()

        users = train["user_id"].values
        items = train["song_id"].values
        ratings = train["play_count"].values.astype(np.float64)

        history = {"train_mse": [], "val_mse": []}

        for epoch in range(1, n_epochs + 1):
            # Shuffle training data each epoch
            perm = np.random.permutation(len(ratings))

            for idx in perm:
                u = users[idx]
                i = items[idx]
                r = ratings[idx]

                # Compute prediction and error
                pred = self.predict_one(u, i)
                err = r - pred

                # SGD updates
                self.b_u[u] += self.lr * (err - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (err - self.reg * self.b_i[i])

                p_u_old = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * p_u_old - self.reg * self.Q[i])

            # --- End-of-epoch metrics ---
            train_mse = compute_mse(self, train)
            history["train_mse"].append(train_mse)

            msg = f"  Epoch {epoch:>3}/{n_epochs}  |  Train MSE: {train_mse:.6f}"

            if val is not None:
                val_mse = compute_mse(self, val)
                history["val_mse"].append(val_mse)
                msg += f"  |  Val MSE: {val_mse:.6f}"

            if verbose:
                print(msg)

        return history

#evaluation metrics/error metrics
def compute_mse(model: SVD, data: pd.DataFrame) -> float:
    """
    Compute Mean Squared Error of the model on a given dataset.

    Reusable on any split (train, validation, or test).
    """
    users = data["user_id"].values
    items = data["song_id"].values
    actuals = data["play_count"].values.astype(np.float64)

    preds = model.predict(users, items)
    mse = np.mean((actuals - preds) ** 2)
    return float(mse)


def compute_rmse(model: SVD, data: pd.DataFrame) -> float:
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(compute_mse(model, data)))


def evaluate_model(model: SVD, data: pd.DataFrame, label: str = "Test") -> dict:
    """
    Full evaluation report on a dataset split.

    Returns dict with MSE, RMSE, and MAE.
    """
    users = data["user_id"].values
    items = data["song_id"].values
    actuals = data["play_count"].values.astype(np.float64)

    preds = model.predict(users, items)
    errors = actuals - preds

    metrics = {
        "mse": float(np.mean(errors ** 2)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mae": float(np.mean(np.abs(errors))),
    }

    print(f"\n{'='*50}")
    print(f"  {label} Set Evaluation")
    print(f"{'='*50}")
    print(f"  Samples:  {len(data)}")
    print(f"  MSE:      {metrics['mse']:.6f}")
    print(f"  RMSE:     {metrics['rmse']:.6f}")
    print(f"  MAE:      {metrics['mae']:.6f}")
    print(f"{'='*50}")

    return metrics


def main():
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "processed" / "user_song_interaction.csv"

    # --- Step 1: Load & prepare ---
    print("=" * 60)
    print("  STEP 1: Loading and preparing data")
    print("=" * 60)
    df, user_to_id, song_to_id = load_and_prepare_data(str(data_path))

    # --- Step 2: Split ---
    print("\n" + "=" * 60)
    print("  STEP 2: Splitting data")
    print("=" * 60)
    train, val, test = split_data(df, train_frac=0.75, val_frac=0.125, test_frac=0.125)

    # --- Step 3 & 4: Initialize and train SVD with SGD ---
    print("\n" + "=" * 60)
    print("  STEP 3: Training SVD with SGD")
    print("=" * 60)
    n_users = len(user_to_id)
    n_items = len(song_to_id)
    print(f"  Matrix shape: {n_users} users x {n_items} items")
    print(f"  Latent factors: 20")
    print(f"  Learning rate: 0.005, Regularization: 0.02")
    print()

    model = SVD(
        n_users=n_users,
        n_items=n_items,
        n_factors=20,
        lr=0.005,
        reg=0.02,
    )
    model.song_to_id = song_to_id

    history = model.fit(train, val=val, n_epochs=20, verbose=True)

    # --- Step 5: Final evaluation on test set ---
    print("\n" + "=" * 60)
    print("  STEP 4: Evaluating on held-out test set")
    print("=" * 60)
    test_metrics = evaluate_model(model, test, label="Test")
    train_metrics = evaluate_model(model, train, label="Train")
    val_metrics = evaluate_model(model, val, label="Validation")

    print("\n\nSummary:")
    print(f"  Train MSE:  {train_metrics['mse']:.6f}  |  RMSE: {train_metrics['rmse']:.6f}")
    print(f"  Val MSE:    {val_metrics['mse']:.6f}  |  RMSE: {val_metrics['rmse']:.6f}")
    print(f"  Test MSE:   {test_metrics['mse']:.6f}  |  RMSE: {test_metrics['rmse']:.6f}")

    # --- Step 5: Save model ---
    print("\n" + "=" * 60)
    print("  STEP 5: Saving trained model")
    print("=" * 60)
    save_path = project_root / "data" / "models" / "svd_model.npz"
    model.save(save_path)


if __name__ == "__main__":
    main()
