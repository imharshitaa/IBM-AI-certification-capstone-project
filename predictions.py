import pandas as pd
import joblib

def load_model(model_path, scaler_path):
    """Load the trained model and scaler from disk."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def preprocess_input_data(df, scaler):
    """Preprocess the input data for prediction."""
    # Select the same features used during training
    X = df[['Speed (km/h)', 'Vehicle Count', 'is_rush_hour', 'Hour of Day']]
    
    # Convert categorical features (if any) to numeric (e.g., one-hot encoding)
    X = pd.get_dummies(X, drop_first=True)
    
    # Scale the input data using the scaler
    X_scaled = scaler.transform(X)
    
    return X_scaled

def make_predictions(model, X_scaled):
    """Make predictions using the trained model."""
    predictions = model.predict(X_scaled)
    
    # Convert numeric predictions back to categorical labels
    flow_map_reverse = {0: 'Light', 1: 'Medium', 2: 'Heavy'}
    predicted_flow = [flow_map_reverse[round(pred)] for pred in predictions]
    
    return predicted_flow

if __name__ == "__main__":
    # Example usage
    model_path = "traffic_flow_model.pkl"
    scaler_path = "scaler.pkl"
    
    # Load the model and scaler
    model, scaler = load_model(model_path, scaler_path)
    
    # Example: new input data (you can replace this with real-time data)
    new_data = pd.DataFrame({
        'Speed (km/h)': [35, 60],
        'Vehicle Count': [400, 1000],
        'is_rush_hour': [1, 0],
        'Hour of Day': [8, 14]
    })
    
    # Preprocess the new data
    X_scaled = preprocess_input_data(new_data, scaler)
    
    # Make predictions
    predictions = make_predictions(model, X_scaled)
    
    # Display the results
    new_data['Predicted Traffic Flow'] = predictions
    print(new_data)
