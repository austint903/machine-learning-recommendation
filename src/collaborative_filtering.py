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

