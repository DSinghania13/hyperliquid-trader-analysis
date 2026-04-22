import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Hyperliquid Trader Analysis", layout="wide")

st.title("Hyperliquid Trader Behavior vs Market Sentiment")
st.markdown(
    "A lightweight interactive dashboard to explore Bitcoin's Fear/Greed index, trader archetypes, and predictive modeling.")


@st.cache_data
def load_data():
    return pd.read_csv('data/daily_metrics.csv')


@st.cache_resource
def load_models():
    kmeans = joblib.load('models/kmeans_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    rf = joblib.load('models/rf_model.pkl')
    return kmeans, scaler, rf


try:
    df = load_data()
    kmeans, scaler, rf = load_models()

    df_ml_dummy = pd.get_dummies(df.copy(), columns=['sentiment'], drop_first=True)
    rf_feature_cols = ['daily_pnl', 'num_trades', 'avg_trade_size', 'win_rate'] + [c for c in df_ml_dummy.columns if
                                                                                   'sentiment_' in c]

    # --- TABS SETUP ---
    tab1, tab2, tab3 = st.tabs(
        ["Market Sentiment", "Trader Archetypes (Interactive)", "Profitability Predictor (Interactive)"])

    # ==========================================
    # TAB 1: YOUR ORIGINAL DASHBOARD
    # ==========================================
    with tab1:
        st.subheader("Filter the Data")
        selected_sentiment = st.multiselect(
            "Select Market Sentiment",
            options=df['sentiment'].unique(),
            default=df['sentiment'].unique()
        )

        filtered_df = df[df['sentiment'].isin(selected_sentiment)]

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Trades Selected", f"{filtered_df['num_trades'].sum():,}")
        col2.metric("Average Win Rate", f"{filtered_df['win_rate'].mean() * 100:.1f}%")
        col3.metric("Avg Trade Size", f"${filtered_df['avg_trade_size'].mean():,.0f}")

        st.divider()

        st.subheader("Performance Breakdown")
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        sns.barplot(data=filtered_df, x='sentiment', y='daily_pnl', ax=ax[0], palette='viridis')
        ax[0].set_title('Average Daily PnL per Sentiment')
        ax[0].set_ylabel('PnL (USD)')

        sns.barplot(data=filtered_df, x='sentiment', y='avg_trade_size', ax=ax[1], palette='magma')
        ax[1].set_title('Average Trade Size per Sentiment')
        ax[1].set_ylabel('Trade Size (USD)')

        st.pyplot(fig)

        with st.expander("Peek at the raw data"):
            st.dataframe(filtered_df.head(100))

    # ==========================================
    # TAB 2: INTERACTIVE K-MEANS CLUSTERING
    # ==========================================
    with tab2:
        st.subheader("K-Means Clustering: Test a Trader Profile")
        st.markdown(
            "Use the sliders below to create a hypothetical trader. The pre-trained K-Means model will classify them into an archetype.")

        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            input_trade_size = st.number_input("Average Trade Size ($)", min_value=10, max_value=50000, value=5000)
        with col_c2:
            input_total_trades = st.number_input("Total Lifetime Trades", min_value=1, max_value=20000, value=100)
        with col_c3:
            input_win_rate = st.slider("Average Win Rate", min_value=0.0, max_value=1.0, value=0.55)

        input_scaled = scaler.transform([[input_trade_size, input_total_trades, input_win_rate]])
        predicted_cluster = kmeans.predict(input_scaled)[0]


        def get_archetype_name(cluster_id, trades, size):
            if trades > 5000: return "High-Frequency Scalper"
            if size > 15000: return "Whale / Swing Trader"
            return "Retail / Casual Trader"


        archetype_result = get_archetype_name(predicted_cluster, input_total_trades, input_trade_size)

        st.success(f"### Predicted Archetype: **{archetype_result}**")
        st.divider()

        st.markdown("### Historical Archetype Distribution")
        trader_profiles = df.groupby('Account').agg(
            avg_trade_size=('avg_trade_size', 'mean'),
            total_trades=('num_trades', 'sum'),
            avg_win_rate=('win_rate', 'mean')
        ).fillna(0).reset_index()

        X_hist_scaled = scaler.transform(trader_profiles[['avg_trade_size', 'total_trades', 'avg_win_rate']])
        trader_profiles['Cluster'] = kmeans.predict(X_hist_scaled)


        def name_historical_cluster(row):
            if row['total_trades'] > 5000: return "High-Frequency Scalper"
            if row['avg_trade_size'] > 15000: return "Whale / Swing Trader"
            return "Retail / Casual Trader"


        trader_profiles['Archetype'] = trader_profiles.apply(name_historical_cluster, axis=1)

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=trader_profiles, x='total_trades', y='avg_trade_size', hue='Archetype', palette='Set1',
                        s=100)
        plt.title("Historical Trade Size vs Trade Frequency")
        st.pyplot(fig2)

    # ==========================================
    # TAB 3: INTERACTIVE RANDOM FOREST PREDICTOR
    # ==========================================
    with tab3:
        st.subheader("Predict Tomorrow's Profitability")
        st.markdown(
            "Input today's trading stats and the market sentiment to see if the Random Forest model predicts a profitable day tomorrow.")

        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            rf_pnl = st.number_input("Today's PnL ($)", min_value=-50000, max_value=50000, value=100)
            rf_trades = st.number_input("Trades Made Today", min_value=1, max_value=500, value=10)
        with col_p2:
            rf_trade_size = st.number_input("Average Trade Size Today ($)", min_value=10, max_value=50000, value=1000)
            rf_win_rate = st.slider("Today's Win Rate", min_value=0.0, max_value=1.0, value=0.60)
        with col_p3:
            rf_sentiment = st.selectbox("Current Market Sentiment", options=["Greed", "Fear", "Neutral"])

        input_data = {
            'daily_pnl': rf_pnl,
            'num_trades': rf_trades,
            'avg_trade_size': rf_trade_size,
            'win_rate': rf_win_rate
        }

        for col in rf_feature_cols:
            if 'sentiment_' in col:
                input_data[col] = 1 if col == f'sentiment_{rf_sentiment}' else 0

        input_df = pd.DataFrame([input_data])[rf_feature_cols]

        if st.button("Predict Tomorrow's Result"):
            prediction = rf.predict(input_df)[0]
            if prediction == 1:
                st.success("### Model Prediction: **PROFITABLE Tomorrow**")
            else:
                st.error("### Model Prediction: **NOT PROFITABLE Tomorrow**")

        st.divider()
        st.markdown("### What drives these predictions?")
        importances = pd.Series(rf.feature_importances_, index=rf_feature_cols).sort_values(ascending=False)
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        sns.barplot(x=importances.values, y=importances.index, palette='Blues_r')
        plt.title("Feature Importances")
        st.pyplot(fig3)

except FileNotFoundError:
    st.error(
        "Could not find the necessary files. Make sure 'daily_metrics.csv' is in the 'data' folder, and your `.pkl` files are in the 'models' folder.")