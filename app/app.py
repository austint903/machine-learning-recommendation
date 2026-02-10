import sys
from pathlib import Path

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import gradio as gr
from model.hybrid_recommender import HybridRecommender, AUDIO_FEATURE_COLS

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")
sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(),
    requests_timeout=10,
    retries=3,
)

def search_spotify(query):
    if not query or not query.strip():
        return gr.update(choices=[], value=None)

    try:
        result = sp.search(q=query.strip(), type="track", limit=5)
        items = result["tracks"]["items"]
        choices = [
            f"{t['artists'][0]['name']} - {t['name']}"
            for t in items
        ]
        return gr.update(choices=choices, value=choices[0] if choices else None)
    except Exception:
        return gr.update(choices=[], value=None)


def get_recommendations(sel1, sel2, sel3, sel4, sel5, num_recommendations):

    selections = [s for s in [sel1, sel2, sel3, sel4, sel5] if s]
    if len(selections) == 0:
        return "Please search and select at least one song."

    songs = []
    for entry in selections:
        if " - " in entry:
            artist, track = entry.split(" - ", 1)
            songs.append((artist.strip(), track.strip()))

    user_vectors = []
    found_songs = []

    for artist, track in songs:
        query = f"artist:{artist} track:{track}"
        try:
            result = sp.search(q=query, type="track", limit=1)
            items = result["tracks"]["items"]
            if not items:
                continue

            spotify_id = items[0]["id"]
            features = sp.audio_features([spotify_id])

            if features and features[0] is not None:
                vec = [features[0].get(col, 0) for col in AUDIO_FEATURE_COLS]
                user_vectors.append(vec)
                found_songs.append(f"{items[0]['artists'][0]['name']} - {items[0]['name']}")
        except Exception:
            continue

    if len(user_vectors) == 0:
        return "Could not fetch audio features for your songs."

    features_path = project_root / "data" / "processed" / "audio_features.csv"
    songs_path = project_root / "data" / "processed" / "unique_song_interaction.csv"
    svd_model_path = project_root / "data" / "models" / "svd_model.npz"

    if not features_path.exists():
        return "audio_features.csv not found. Run: python scripts/fetch_spotify_features.py"

    recommender = HybridRecommender(
        features_path=features_path,
        songs_path=songs_path,
        svd_model_path=svd_model_path,
    )

    n_recs = int(num_recommendations)
    results = recommender.recommend(
        user_feature_vectors=user_vectors,
        num_candidates=100,
        num_recommendations=n_recs,
    )

    lines = [f"Based on: {', '.join(found_songs)}"]

    if recommender.svd is not None:
        lines.append("Mode: Hybrid (KNN candidates → SVD ranking)\n")
    else:
        lines.append("Mode: KNN-only (train SVD for hybrid ranking)\n")

    for r in results:
        line = f"{r['rank']}. {r['artist']} - {r['track']}"
        if r["svd_score"] is not None:
            line += f"  (SVD: {r['svd_score']:.4f}, KNN: {r['knn_distance']:.4f})"
        else:
            line += f"  (KNN distance: {r['knn_distance']:.4f})"
        lines.append(line)

    return "\n".join(lines)


with gr.Blocks(title="Music Recommendation System") as demo:
    gr.Markdown(
        """
        # Music Recommendation System

        Search for songs on Spotify, select them, and get recommendations
        using a two-stage hybrid model (KNN candidate generation → SVD ranking).
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Search & Select Songs")
            gr.Markdown("Type a search query and press Enter to find songs on Spotify.")

            search1 = gr.Textbox(label="Search Song 1", placeholder="e.g. Radiohead Creep")
            select1 = gr.Dropdown(label="Select Song 1", choices=[], interactive=True)

            search2 = gr.Textbox(label="Search Song 2", placeholder="e.g. Arctic Monkeys 505")
            select2 = gr.Dropdown(label="Select Song 2", choices=[], interactive=True)

            search3 = gr.Textbox(label="Search Song 3", placeholder="e.g. Tame Impala")
            select3 = gr.Dropdown(label="Select Song 3", choices=[], interactive=True)

            search4 = gr.Textbox(label="Search Song 4", placeholder="e.g. Daft Punk")
            select4 = gr.Dropdown(label="Select Song 4", choices=[], interactive=True)

            search5 = gr.Textbox(label="Search Song 5", placeholder="e.g. Frank Ocean")
            select5 = gr.Dropdown(label="Select Song 5", choices=[], interactive=True)

            num_recs = gr.Slider(
                minimum=5,
                maximum=20,
                step=5,
                value=10,
                label="Number of Recommendations",
            )

            submit_btn = gr.Button("Get Recommendations", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### Your Recommendations")
            output = gr.Textbox(
                label="Recommended Songs",
                lines=25,
                placeholder="Search for songs, select them, and click 'Get Recommendations'..."
            )

    search1.submit(fn=search_spotify, inputs=search1, outputs=select1)
    search2.submit(fn=search_spotify, inputs=search2, outputs=select2)
    search3.submit(fn=search_spotify, inputs=search3, outputs=select3)
    search4.submit(fn=search_spotify, inputs=search4, outputs=select4)
    search5.submit(fn=search_spotify, inputs=search5, outputs=select5)

    submit_btn.click(
        fn=get_recommendations,
        inputs=[select1, select2, select3, select4, select5, num_recs],
        outputs=output,
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )
