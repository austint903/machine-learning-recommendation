# Machine Learning Recommendation Machine

A hybrid music recommendation engine that suggests songs by combining two ML techniques: collaborative filtering (learning from listening patterns) and content-based filtering (analyzing what makes songs sound similar).

Collaborative filtering uses SVD matrix factorization trained with stochastic gradient descent on Last.fm listening data to discover hidden patterns in user preferences. Content-based filtering uses Spotify audio features (energy, tempo, danceability, etc.) with K-Nearest Neighbors to find songs that sound similar to what a user already likes.

## Requirements

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- Spotify API credentials (client ID and secret from [Spotify Developer Dashboard](https://developer.spotify.com/dashboard))

## Setup

```bash
# Create virtual environment
python3.12 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv sync
```

Add your Spotify API credentials to `.env`
Look at `.env.example` to see example keys

## Commands

### 1. Parse raw Last.fm data

```bash
python scripts/parse_csv_user_song.py

python scripts/parse_csv_unique_song.py
```

These produce:
- `user_song_interaction.csv`
- `unique_song_interaction.csv`

### 2. Fetch Spotify audio features

```bash
python scripts/fetch_spotify_features.py
```

Searches Spotify for each song and fetches 13 audio features (energy, tempo, danceability, etc.). Takes ~70 minutes. Checkpoints every 500 songs so it can resume if interrupted.

Produces: `audio_features.csv`

### 3. Train collaborative filtering model (SVD)

```bash
python model/collaborative_filtering.py
```

Trains SVD matrix factorization with stochastic gradient descent on user-song interactions. Splits data 75/12.5/12.5 train/val/test and prints mean squared error each epoch. Saves the trained model to `data/models/svd_model.npz` for use by the hybrid recommender.

### 4. Run content-based filtering model (KNN)

```bash
python model/content_based_filtering.py
```

Loads audio features, fits KNN with cosine similarity, and generates 10 recommendations per user based on their listening history.

### 5. Run the app (hybrid recommender)

```bash
python app/app.py
```

Launches a Gradio web UI at `http://127.0.0.1:7860` (can also be deployed publically). Search for songs on Spotify, select up to 5, and get recommendations using a two-stage hybrid pipeline:
1. **KNN candidate generation** — finds ~100 sonically similar songs from the dataset
2. **SVD ranking** — re-ranks candidates using collaborative filtering item biases (learned song popularity)
