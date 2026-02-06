import pandas as pd
from pathlib import Path


def parse_raw_data_unique_song():
    """
    Parse raw Last.fm data and create a unique songs dataset.
    Creates unique_song_interaction.csv with track_id, Artist, and Track columns.
    """
    project_root = Path(__file__).parent.parent
    raw_data_path = project_root / 'data' / 'raw' / 'Last.fm_data.csv'
    processed_data_path = project_root / 'data' / 'processed' / 'unique_song_interaction.csv'

    print(f"Loading raw data from {raw_data_path}...")
    df = pd.read_csv(raw_data_path)

    print(f"Loaded {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")

    # Get unique Artist-Track combinations
    print("\nExtracting unique songs...")
    unique_songs = df[['Artist', 'Track']].drop_duplicates().reset_index(drop=True)

    # Add track_id as the index
    unique_songs.insert(0, 'track_id', unique_songs.index)

    # Create processed directory if it doesn't exist
    processed_data_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    print(f"\nSaving unique songs to {processed_data_path}...")
    unique_songs.to_csv(processed_data_path, index=False)

    # Display summary statistics
    print(f"\nProcessing complete!")
    print(f"Total unique songs: {len(unique_songs)}")
    print(f"\nSample of the data:")
    print(unique_songs.head(10))


if __name__ == "__main__":
    parse_raw_data_unique_song()