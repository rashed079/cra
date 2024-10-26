# src/data_processing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path='data/raw_data.csv'):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    # Drop rows with missing values or encode categorical features
    df.dropna(inplace=True)
    
    # Feature Engineering
    df['loan_to_income_ratio'] = df['loan_amount'] / df['income']
    
    # Split data
    X = df.drop('default', axis=1)
    y = df['default']
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
