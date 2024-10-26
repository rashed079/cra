# main.py
from src.data_processing import load_data, preprocess_data
from src.logistic_regression import train_logistic_regression, evaluate_model
from src.deep_learning_model import train_deep_learning_model
from src.matrix_factorization import matrix_factorization

# Load and preprocess data
df = load_data()
X_train, X_test, y_train, y_test = preprocess_data(df)

# Logistic Regression
logistic_model = train_logistic_regression(X_train, y_train)
logistic_accuracy = evaluate_model(logistic_model, X_test, y_test)
print("Logistic Regression Accuracy:", logistic_accuracy)

# Matrix Factorization
W, H = matrix_factorization(X_train)

# Deep Learning Model
deep_learning_model = train_deep_learning_model(X_train, y_train)
print("Deep Learning Model training complete.")