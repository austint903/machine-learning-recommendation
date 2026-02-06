import pandas as pd
from pathlib import Path


def parse_raw_data_user_song():
    # Define paths
    project_root = Path(__file__).parent.parent
    raw_data_path = project_root / 'data' / 'raw' / 'Last.fm_data.csv'
    processed_data_path = project_root / 'data' / 'processed' / 'user_song_interaction.csv'

    # Load raw data
    print(f"Loading raw data from {raw_data_path}")
    df = pd.read_csv(raw_data_path)

    # Display initial data info
    print(f"Loaded {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")

    # Aggregate play counts by Username, Artist, and Track
    print("\nAggregating play counts...")
    aggregated_df = df.groupby(['Username', 'Artist', 'Track']).size().reset_index(name='play_count')

    # Rename columns for clarity
    aggregated_df.columns = ['user', 'artist', 'song', 'play_count']

    # Sort by user and play count (descending)
    aggregated_df = aggregated_df.sort_values(['user', 'play_count'], ascending=[True, False])

    # Create processed directory if it doesn't exist
    processed_data_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    print(f"\nSaving aggregated data to {processed_data_path}...")
    aggregated_df.to_csv(processed_data_path, index=False)

    # Display summary statistics
    print(f"\nAggregation complete!")
    print(f"Total unique users: {aggregated_df['user'].nunique()}")
    print(f"Total unique artists: {aggregated_df['artist'].nunique()}")
    print(f"Total unique songs: {aggregated_df['song'].nunique()}")
    print(f"Total user-song interactions: {len(aggregated_df)}")
    print(f"\nSample of the aggregated data:")
    print(aggregated_df.head(10))


if __name__ == "__main__":
    parse_raw_data_user_song()


