import pandas as pd
import os

def calculate_average_wind():
    """
    Calculate the average wind speed from current_data.csv
    """
    csv_file = "current_data.csv"
    
    # Check if the file exists
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        print("Please run a weather query first to generate the data.")
        return None
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} records from {csv_file}")
        
        # Display basic info about the dataset
        print(f"\nDataset columns: {list(df.columns)}")
        
        # Check if wind_speed column exists
        if 'wind_speed' not in df.columns:
            print("Error: 'wind_speed' column not found in the dataset!")
            print("Available columns:", df.columns.tolist())
            return None
        
        # Calculate average wind speed
        avg_wind = df['wind_speed'].mean()
        min_wind = df['wind_speed'].min()
        max_wind = df['wind_speed'].max()
        
        # Display results
        print(f"\n=== Wind Speed Analysis ===")
        print(f"Average Wind Speed: {avg_wind:.2f} m/s")
        print(f"Minimum Wind Speed: {min_wind:.2f} m/s")
        print(f"Maximum Wind Speed: {max_wind:.2f} m/s")
        
        # Additional statistics
        std_wind = df['wind_speed'].std()
        print(f"Standard Deviation: {std_wind:.2f} m/s")
        
        # Show first few records
        print(f"\nFirst 5 records:")
        if 'timestamp' in df.columns:
            print(df[['timestamp', 'wind_speed']].head())
        else:
            print(df[['wind_speed']].head())
        
        return avg_wind
        
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

if __name__ == "__main__":
    print("Wind Speed Analysis Tool")
    print("=" * 30)
    
    average_wind = calculate_average_wind()
    
    if average_wind is not None:
        print(f"\n✅ Analysis complete! Average wind speed: {average_wind:.2f} m/s")
    else:
        print("\n❌ Analysis failed. Please check the error messages above.")