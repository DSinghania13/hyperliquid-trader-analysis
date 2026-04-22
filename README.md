# 📈 Hyperliquid Trader Behavior vs Market Sentiment
**Primetrade.ai - Data Science Intern Assignment**

🚀 **[Click Here to view the Live Interactive Dashboard!](https://hyperliquid-trader-analysis.streamlit.app/)**

This repository contains an analysis of how Bitcoin's Fear & Greed index influences trader behavior and performance on the Hyperliquid platform. It includes data wrangling, segmentation, and advanced machine learning models (K-Means and Random Forest) to predict next-day profitability.

---

## 📊 Output Charts & Visualizations

### 1. Performance Overview by Sentiment
![Sentiment Performance](https://github.com/user-attachments/assets/7571601b-0e53-4932-a23a-36dddaeedfd5)

### 2. Behavioral Shifts (Trade Frequency & Risk)
![Sentiment Behavior](https://github.com/user-attachments/assets/7fe2d75a-6f1e-419d-a30c-1ebca883b6ff)

### 3. Trader Archetypes (Win-Rate Segments)
![Segment Behavior](https://github.com/user-attachments/assets/28b82696-7955-411e-8d09-1c40acc94ff9)

---

## 📝 Executive Write-Up

### Methodology
1. **Data Alignment & Cleaning:** Ingested Historical Trader Data and Fear/Greed Index. Converted Hyperliquid `Timestamp IST` to standard dates and merged the datasets on a daily level. Both datasets were verified to have zero missing values and zero duplicate rows.
2. **Feature Engineering:** Aggregated execution-level data into daily metrics per account. Engineered features include Daily PnL, Win Rate, Trade Frequency, Average Trade Size (used as a proxy for leverage/risk exposure), and Long/Short Ratio.
3. **Segmentation & Machine Learning:** Grouped accounts into basic behavioral segments using median thresholds. Furthermore, developed a **K-Means Clustering** algorithm to automatically categorize users into three advanced archetypes (Scalpers, Whales, Casual). Finally, trained a **Random Forest Classifier** to predict next-day profitability.

### Key Insights
1. **Performance Asymmetry:** While win rates remain remarkably stable across market sentiments (~61%), the *Average Daily PnL* is significantly higher during 'Fear' days ($5,185) compared to 'Greed' days ($4,144). 'Fear' markets are driven by massive blow-out days for specific top traders.
2. **Behavioral Shifts:** Traders adapt aggressively to sentiment. During 'Fear' days, average trade frequency jumps to 105 trades per account (vs. 77 in Greed), and average trade size spikes by ~43%. 
3. **Archetype Divergence:** High-Win-Rate traders thrive in 'Fear' markets, drastically increasing their trade size and capturing volatility. Conversely, Low-Win-Rate traders struggle during 'Fear' markets but perform much better in trending 'Greed' environments.

### Strategy Recommendations
Based on the data, here are two proposed platform rules of thumb:

**1. Contextual Risk Constraint (The "Fear" Rule)**
* *Rule:* During 'Fear' market days, algorithmically enforce a reduction in maximum allowable position size limits for the *Low-Win-Rate / Infrequent* trader segment. Conversely, dynamically increase position limits for *High-Win-Rate* accounts.
* *Why:* Protects inconsistent traders from liquidation during high-volatility events, while maximizing volume and fees from traders who demonstrably capitalize on Fear swings.

**2. Regime-Based Liquidity Incentives (The "Greed" Rule)**
* *Rule:* When the sentiment index shifts to 'Greed', overall trading frequency naturally drops. The platform should deploy targeted UI alerts or temporarily increase maker-rebates specifically for the *Frequent* trader segment.
* *Why:* Incentivizes active users to provide limit orders, ensuring the order book remains sufficiently deep even when the broader market slows down.

---

## 💻 Setup & How to Run Locally

If you prefer to run the analysis and Streamlit app locally rather than using the live link above, follow these steps:

### 1. Clone the repository:
```bash
git clone https://github.com/DSinghania13/hyperliquid-trader-analysis.git
cd hyperliquid-trader-analysis
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
python3.10 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

### 3. Install dependencies:
Ensure you have Python installed, then run:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib
```

### 4. Run the Jupyter Notebook (Optional):
Open notebook.ipynb to view the step-by-step data cleaning, exploratory data analysis, and model training.

### 5. Launch the Interactive Dashboard:
Run the Streamlit app to explore the live K-Means clustering and Random Forest predictor:
```bash
streamlit run app.py
```

