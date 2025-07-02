# 📈 Stock-Prediction-Model  
🔍 Next-Day Price Movement Prediction Using Machine Learning

This repository contains a Python-based stock prediction model that forecasts **next-day price direction** (up or down) for publicly traded stocks using technical indicators and a Random Forest Classifier. It pulls historical data directly from Yahoo Finance and performs analysis with visual outputs to support decision-making.

---

## 📝 How to Use

1. **Download or clone the repository**:

2. **Install dependencies**:

pip install yfinance pandas numpy matplotlib seaborn scikit-learn

3. **Run the script**:

python stock_prediction.py

4. **Enter your input** when prompted:

Enter The Stock Ticker in YFINANCE format : AAPL

The script will:

* Collect stock history from Yahoo Finance (2000–2024)
* Compute technical indicators
* Train a Random Forest classifier
* Predict the price movement for the next trading day
* Display charts including:

  * Confusion matrix
  * ROC curve
  * Feature importance bar chart
  * Correlation heatmap

---

## 📊 Features

* Predicts **up or down movement** for the next trading day
* 15+ technical indicators as features, including:

  * RSI, volatility, momentum, rate of change
  * Candlestick patterns (shadows, body size, bullish signal)
  * Moving averages and crossovers
  * Donchian channels
* Uses **Random Forest Classifier** with class balancing
* Shows **ROC AUC**, **confusion matrix**, and **feature importance**
* Designed to help with **trading insights and educational analysis**

---
## ✅ Requirements

* Python 3.7 or newer
* Libraries:

  * yfinance
  * pandas
  * numpy
  * matplotlib
  * seaborn
  * scikit-learn

---

## 📌 Disclaimer

This tool is provided **for educational and informational purposes only**. It does not constitute financial advice or investment recommendation. Use at your own risk and consult a professional advisor before making financial decisions.

---

## 📧 Contact

Have questions, ideas, or want to contribute?
Open an issue or contact the repository owner at: KeshavPanchal1127

