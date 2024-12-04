def predict_and_trade(model, scaler, X_test, y_test):
    """Predicts prices and executes a simple trading strategy.

    Args:
        model: Trained LSTM model.
        scaler: Scaler used for data preprocessing.
        X_test: Test input data.
        y_test: Actual test target values.
    """

    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    for real_price, predicted_price in zip(y_test, predicted_prices):
        if predicted_price > real_price:
            print("Buy")
        elif predicted_price < real_price:
            print("Sell")

if __name__ is "__main__":
    print(None)