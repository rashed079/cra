# src/logistic_regression.py
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

if __name__ == "__main__":
    from data_processing import preprocess_data, load_data
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    model = train_logistic_regression(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    print("Logistic Regression Accuracy:", accuracy)