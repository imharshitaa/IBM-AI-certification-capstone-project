import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def load_processed_data(file_path):
    """Load the processed traffic data for model training."""
    return pd.read_csv(file_path)

def preprocess_features(df):
    """Prepare the features and target variable for model training."""
    # Features: Use Speed, Vehicle Count, is_rush_hour, and Hour of Day
    X = df[['Speed (km/h)', 'Vehicle Count', 'is_rush_hour', 'Hour of Day']]
    
    # Convert categorical features (if any) to numeric (e.g., one-hot encoding)
    X = pd.get_dummies(X, drop_first=True)
    
    # Target: Predicting Traffic Flow as a numeric value (encoded as Light=0, Medium=1, Heavy=2)
    traffic_flow_map = {'Light': 0, 'Medium': 1, 'Heavy': 2}
    y = df['Traffic Flow'].map(traffic_flow_map)
    
    return X, y

def train_model(X, y):
    """Train a regression model to predict traffic flow."""
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a simple linear regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"Model trained with Mean Squared Error: {mse}")
    
    return model, scaler

def save_model(model, scaler, model_path, scaler_path):
    """Save the trained model and scaler to disk."""
    import joblib
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

if __name__ == "__main__":
    # Example usage
    file_path = "traffic_data.csv"
    model_path = "traffic_flow_model.pkl"
    scaler_path = "scaler.pkl"
    
    # Load and preprocess the data
    processed_data = load_processed_data(file_path)
    X, y = preprocess_features(processed_data)
    
    # Train the model
    model, scaler = train_model(X, y)
    
    # Save the model and scaler
    save_model(model, scaler, model_path, scaler_path)
