import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Load the processed data from CSV."""
    return pd.read_csv(file_path)

def plot_traffic_flow_distribution(df):
    """Plot the distribution of traffic flow categories."""
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Traffic Flow')
    plt.title('Distribution of Traffic Flow')
    plt.xlabel('Traffic Flow Category')
    plt.ylabel('Count')
    plt.show()

def plot_speed_vs_traffic_flow(df):
    """Plot a scatter plot of Speed vs Traffic Flow."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Speed (km/h)', y='Vehicle Count', hue='Traffic Flow', palette='coolwarm')
    plt.title('Speed vs Vehicle Count by Traffic Flow')
    plt.xlabel('Speed (km/h)')
    plt.ylabel('Vehicle Count')
    plt.show()

def plot_predictions_vs_actual(df):
    """Compare predicted vs actual traffic flow."""
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Predicted Traffic Flow', hue='Traffic Flow', palette='Set2')
    plt.title('Predicted vs Actual Traffic Flow')
    plt.xlabel('Predicted Traffic Flow')
    plt.ylabel('Count')
    plt.legend(title='Actual Traffic Flow')
    plt.show()

if __name__ == "__main__":
    # Example usage
    data_path = "processed_traffic_data.csv"
    
    # Load the data
    traffic_data = load_data(data_path)
    
    # Visualize the distribution of traffic flow categories
    plot_traffic_flow_distribution(traffic_data)
    
    # Visualize speed vs vehicle count colored by traffic flow
    plot_speed_vs_traffic_flow(traffic_data)
    
    # If predictions are already made, you can visualize predicted vs actual traffic flow
    # Assuming the DataFrame has a 'Predicted Traffic Flow' column with predictions
    # traffic_data['Predicted Traffic Flow'] = ['Light', 'Medium', 'Heavy', ...]  # Example predicted data
    # plot_predictions_vs_actual(traffic_data)
