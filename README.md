# Hyperliquid Trader Behavior vs Market Sentiment

**[Click Here to view the Live Interactive Dashboard!](https://hyperliquid-trader-analysis.streamlit.app/)**

This repository contains an analysis of how Bitcoin's Fear & Greed index influences trader behavior and performance on the Hyperliquid platform. It includes data wrangling, segmentation, and advanced machine learning models (K-Means and Random Forest) to predict next-day profitability.

---

## Output Charts & Visualizations

### 1. Performance Overview by Sentiment
![Sentiment Performance](https://github.com/user-attachments/assets/7571601b-0e53-4932-a23a-36dddaeedfd5)

### 2. Behavioral Shifts (Trade Frequency & Risk)
![Sentiment Behavior](https://github.com/user-attachments/assets/7fe2d75a-6f1e-419d-a30c-1ebca883b6ff)

### 3. Trader Archetypes (Win-Rate Segments)
![Segment Behavior](https://github.com/user-attachments/assets/28b82696-7955-411e-8d09-1c40acc94ff9)

---

## Executive Write-Up

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

## Setup & How to Run Locally

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
pip install -r requirements.txt
```

### 4. Run the Jupyter Notebook (Optional):
Open notebook.ipynb to view the step-by-step data cleaning, exploratory data analysis, and model training.

### 5. Launch the Interactive Dashboard:
Run the Streamlit app to explore the live K-Means clustering and Random Forest predictor:
```bash
streamlit run app.py
```

---

## How to Test the Machine Learning Models
Once the dashboard is running (either locally or via the live link), navigate to the interactive tabs and try plugging in these test profiles to see the models in action!

#### K-Means Archetype Test Data (Tab 2)

| Test Case | Avg Trade Size ($) | Lifetime Trades | Win Rate | Expected Archetype | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 1,500 | 50 | 0.45 | **Retail / Casual** | Low volume, low trade count. |
| **2** | 25,000 | 200 | 0.60 | **Whale / Swing Trader** | Massive trade size, moderate frequency. |
| **3** | 500 | 8,000 | 0.75 | **High-Frequency Scalper** | Tiny sizes but thousands of trades. |
| **4** | 5,000 | 120 | 0.50 | **Retail / Casual** | Average numbers across the board. |
| **5** | 45,000 | 50 | 0.55 | **Whale / Swing Trader** | Extremely large sizes, very few trades. |
| **6** | 1,200 | 15,000 | 0.68 | **High-Frequency Scalper** | Massive trade volume. |
| **7** | 200 | 15 | 0.30 | **Retail / Casual** | Brand new or very unsuccessful retail trader. |
| **8** | 2,000 | 6,500 | 0.80 | **High-Frequency Scalper** | Highly active and highly successful. |
| **9** | 18,000 | 500 | 0.45 | **Whale / Swing Trader** | Just crossing the threshold for a Whale account. |
| **10**| 8,000 | 400 | 0.52 | **Retail / Casual** | Upper end of retail, but not quite a Whale or Scalper. |

#### Random Forest Profitability Test Data (Tab 3)

| Test Case | Today's PnL ($) | Trades Today | Avg Trade Size | Win Rate | Sentiment | Expected Prediction |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 5,000 | 25 | 1,500 | 0.85 | Greed | **Profitable** (Great momentum) |
| **2** | -2,500 | 10 | 5,000 | 0.30 | Fear | **Not Profitable** (Poor current performance) |
| **3** | 150 | 5 | 200 | 0.60 | Neutral | **Profitable** (Slow, but consistent) |
| **4** | 12,000 | 150 | 2,500 | 0.75 | Fear | **Profitable** (Thriving in volatility) |
| **5** | -500 | 50 | 1,000 | 0.45 | Greed | **Not Profitable** (Overtrading, losing money) |
| **6** | 35,000 | 5 | 25,000 | 1.00 | Greed | **Profitable** (Massive win rate/size) |
| **7** | -12,000 | 80 | 5,000 | 0.20 | Fear | **Not Profitable** (Heavy losses in a fearful market) |
| **8** | 800 | 15 | 800 | 0.55 | Neutral | **Profitable** (Slight edge, average day) |
| **9** | 0 | 2 | 10,000 | 0.00 | Fear | **Not Profitable** (No wins today) |
| **10**| 4,500 | 200 | 500 | 0.65 | Greed | **Profitable** (High-frequency scalping successfully) |

