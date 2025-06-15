# Récupération de données en direct via Binance (1 minute interval)
def fetch_binance_data(symbol="BTCUSDT", interval="1m", limit=60):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume"
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
    return df

def simulate_realtime_binance(model, rsi_window):
    prediction_log = []
    for _ in range(20):
        df_live = fetch_binance_data()
        df_ind = compute_indicators(df_live.copy(), rsi_window)
        if not df_ind.empty:
            X = df_ind.drop(columns=["target"])
            pred = model.predict(X.iloc[[-1]])[0]
            price = df_ind['close'].iloc[-1]
            prediction_log.append((df_ind.index[-1], price, pred))
        time.sleep(1)
    return pd.DataFrame(prediction_log, columns=["timestamp", "close", "prediction"])

# ----- Onglet Temps Réel (Live avec Binance) -----
tabs.append("Temps Réel")
with st.tabs(tabs)[-1]:
    st.subheader("📡 Prédiction Temps Réel (Binance API)")
    if os.path.exists("model_xgboost.pkl"):
        model = joblib.load("model_xgboost.pkl")
        rsi_realtime = st.slider("Fenêtre RSI (temps réel)", 5, 30, 14)

        if st.button("Démarrer le live Binance"):
            result_df = simulate_realtime_binance(model, rsi_realtime)
            st.success("Données Binance traitées.")

            st.line_chart(result_df.set_index("timestamp")["close"])
            st.bar_chart(result_df.set_index("timestamp")["prediction"])

            if st.button("Exporter les prédictions temps réel"):
                result_df.to_csv("predictions_realtime.csv", index=False)
                st.success("Exporté sous predictions_realtime.csv")
    else:
        st.warning("Aucun modèle trouvé pour la simulation temps réel.")
