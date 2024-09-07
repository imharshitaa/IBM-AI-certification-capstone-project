
import pandas as pd

def load_data(file_path):
    """Load traffic data from a CSV file."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Clean and preprocess the traffic data."""
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Handle missing values (if any)
    df = df.fillna({
        'Speed (km/h)': df['Speed (km/h)'].mean(),
        'Vehicle Count': df['Vehicle Count'].median(),
        'Weather Conditions': 'Clear',
        'Incident Reports': 'None'
    })
    
    # Convert Timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Additional feature engineering if needed
    df['is_rush_hour'] = df['Hour of Day'].apply(lambda x: 1 if 7 <= x <= 9 or 16 <= x <= 18 else 0)
    
    return df

def save_processed_data(df, output_path):
    """Save the processed data to a new CSV file."""
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Example usage
    file_path = "traffic_data.csv"
    output_path = "processed_traffic_data.csv"
    
    # Load the data
    traffic_data = load_data(file_path)
    
    # Clean and preprocess the data
    processed_data = clean_data(traffic_data)
    
    # Save the processed data
    save_processed_data(processed_data, output_path)
    print(f"Processed data saved to {output_path}")
