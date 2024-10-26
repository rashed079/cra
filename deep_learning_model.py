# src/deep_learning_model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import accuracy_score

def create_deep_learning_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])
    return model

def train_deep_learning_model(X_train, y_train):
    model = create_deep_learning_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    return model

if __name__ == "__main__":
    from data_processing import preprocess_data, load_data
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    model = train_deep_learning_model(X_train, y_train)
    predictions = model.predict(X_test)
    predictions = [1 if p > 0.5 else 0 for p in predictions]
    accuracy = accuracy_score(y_test, predictions)
    print("Deep Learning Model Accuracy:", accuracy)