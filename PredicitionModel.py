import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

ticker_input = input("Enter The Stock Ticker in YFINANCE format : ")

def prediction_model(TCK: str):
    ticker = yf.Ticker(TCK)
    stock = ticker.history(start="2000-01-01", end="2024-12-31")

    del stock['Stock Splits'], stock['Dividends']

    def compute_rsi(series, window=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    stock['Future'] = stock['Close'].shift(-1)
    stock['MA_5'] = stock['Close'].rolling(5).mean()
    stock['MA_20'] = stock['Close'].rolling(20).mean()

    stock['Return'] = stock['Close'].pct_change()
    stock['Momentum_5'] = stock['Close'] - stock['Close'].shift(5)

    stock['Volatility_10'] = stock['Return'].rolling(10).std()
    stock['RSI_14'] = compute_rsi(stock['Close'], window=14)

    stock['Price_vs_MA5'] = stock['Close'] - stock['MA_5']
    stock['MA_Crossover'] = (stock['MA_5'] > stock['MA_20']).astype(int)

    stock['Body_Size'] = abs(stock['Close'] - stock['Open'])
    stock['Upper_Shadow'] = stock['High'] - stock[['Close', 'Open']].max(axis=1)
    stock['Lower_Shadow'] = stock[['Close', 'Open']].min(axis=1) - stock['Low']
    stock['Bullish_Candle'] = (stock['Close'] > stock['Open']).astype(int)

    stock['ROC_5'] = (stock['Close'] - stock['Close'].shift(5)) / stock['Close'].shift(5)
    stock['Volume_Change'] = stock['Volume'].pct_change()

    donchian_window = 20
    stock['DC_Upper'] = stock['High'].rolling(window=donchian_window).max()
    stock['DC_Lower'] = stock['Low'].rolling(window=donchian_window).min()
    stock['DC_Mid'] = (stock['DC_Upper'] + stock['DC_Lower']) / 2
    stock['Close_vs_DC_Upper'] = stock['Close'] - stock['DC_Upper']
    stock['Close_vs_DC_Lower'] = stock['Close'] - stock['DC_Lower']

    features = [
        'Close', 'Volume',
        'Return', 'MA_5', 'MA_20', 'Momentum_5',
        'Price_vs_MA5',
        'Body_Size', 'Upper_Shadow', 'Lower_Shadow',
        'ROC_5', 'Volatility_10', 'RSI_14',
        'Volume_Change',
        'Close_vs_DC_Upper', 'Close_vs_DC_Lower'
    ]

    stock['Target'] = (stock['Future'] > stock['Close']).astype(int)

    stock = stock.dropna(subset=['Future'])
    stock.dropna(inplace=True)

    X = stock[features]
    Y = stock['Target']

    split_idx = int(len(stock) * 0.8)

    X_train = X.iloc[:split_idx]
    Y_train = Y.iloc[:split_idx]

    X_test = X.iloc[split_idx:]
    Y_test = Y.iloc[split_idx:]

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_split=10,
        max_depth=8,
        random_state=1,
        class_weight='balanced'
    )

    model.fit(X_train, Y_train)

    Y_proba = model.predict_proba(X_test)[:, 1]
    Y_pred = model.predict(X_test)

    print(f"\n--- Model Results for {ticker} ---")

    cm = confusion_matrix(Y_test, Y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

    importances = model.feature_importances_
    features_names = X_train.columns
    indices = importances.argsort()[::-1]

    plt.figure(figsize=(8,6))
    sns.barplot(x=importances[indices], y=features_names[indices])
    plt.title('Feature Importances')
    plt.show()

    Y_train_proba = model.predict_proba(X_train)[:, 1]
    fpr, tpr, _ = roc_curve(Y_train, Y_train_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Training Data')
    plt.legend(loc='lower right')
    plt.show()

    corr_matrix = X.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=0.5)
    plt.title("Correlation Matrix of Stock Features")
    plt.tight_layout()
    plt.show()

    latest_features = X.iloc[-1:]
    prediction = model.predict(latest_features)
    probability = model.predict_proba(latest_features)[0, 1]

    print(f"Prediction for next day: {'Up' if prediction[0] == 1 else 'Down'} with probability {probability:.2f}")

    print(classification_report(Y_test, Y_pred))
    print(Y_train.value_counts(normalize=True))

print(prediction_model(ticker_input))
