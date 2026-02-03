import gradio as gr

# Placeholder data - will be replaced with actual model predictions
SAMPLE_SONGS = [
    "Song A - Artist 1",
    "Song B - Artist 2",
    "Song C - Artist 3",
    "Song D - Artist 4",
    "Song E - Artist 5",
]


def get_recommendations(rated_songs, num_recommendations):
    """
    Generate music recommendations based on user ratings.

    Args:
        rated_songs: Dictionary of song ratings from user
        num_recommendations: Number of songs to recommend

    Returns:
        Formatted recommendation results
    """
    # TODO: Replace with actual hybrid recommendation model
    # This is a placeholder that will integrate:
    # 1. Collaborative filtering (SVD)
    # 2. Content-based filtering (audio features from Spotify)

    recommendations = []
    for i in range(int(num_recommendations)):
        recommendations.append(
            f"{i+1}. Recommended Song {i+1} - Artist Name\n"
            f"   Similarity Score: 0.{95-i*5}\n"
            f"   Reason: Similar tempo and energy to your rated songs\n"
        )

    return "\n".join(recommendations)


# Build Gradio Interface
with gr.Blocks(title="Music Recommendation System") as demo:
    gr.Markdown(
        """
        # 🎵 Hybrid Music Recommendation System

        This system combines **collaborative filtering** and **content-based filtering**
        to suggest songs you'll love based on your listening preferences.

        ### How it works:
        - Rate 5-10 songs below (1-5 stars)
        - Our hybrid model analyzes:
          - **Collaborative**: What similar users listened to
          - **Content-based**: Audio features (tempo, energy, danceability, valence)
        - Get personalized recommendations!
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Rate Some Songs")
            gr.Markdown("Rate at least 5 songs to get started:")

            # Song rating inputs
            ratings = []
            for song in SAMPLE_SONGS[:10]:
                rating = gr.Slider(
                    minimum=0,
                    maximum=5,
                    step=1,
                    value=0,
                    label=song,
                    info="0 = Not rated, 1-5 = Your rating"
                )
                ratings.append(rating)

            num_recs = gr.Slider(
                minimum=5,
                maximum=20,
                step=5,
                value=10,
                label="Number of Recommendations",
                info="How many songs to recommend"
            )

            submit_btn = gr.Button("Get Recommendations", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### Your Recommendations")
            output = gr.Textbox(
                label="Recommended Songs",
                lines=15,
                placeholder="Rate some songs and click 'Get Recommendations' to see results..."
            )

    gr.Markdown(
        """
        ---
        **Model Status:** 🔴 Using placeholder data (model not yet trained)

        **Next Steps:**
        - Train collaborative filtering model (SVD) on Last.fm dataset
        - Fetch audio features from Spotify API
        - Implement hybrid fusion strategy
        - Evaluate with Precision@K, Recall@K metrics
        """
    )

    # Connect the button to the function
    submit_btn.click(
        fn=get_recommendations,
        inputs=[gr.State({}), num_recs],  # ratings will be passed once model is ready
        outputs=output
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )
