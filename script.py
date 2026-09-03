import os
import datetime
import yfinance as yf
import pandas as pd
import xgboost as xgb

def run_prediction():
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Ejecutando predicción SOXL...")
    try:
        # 1. Descargar datos
        soxl = yf.download("SOXL", period="120d", interval="5m", group_by='ticker')["SOXL"]
        nasdaq = yf.download("^IXIC", period="120d", interval="5m", group_by='ticker')["^IXIC"]
        sox_index = yf.download("^SOX", period="120d", interval="5m", group_by='ticker')["^SOX"]
        vix = yf.download("^VIX", period="120d", interval="5m", group_by='ticker')["^VIX"]

        # 2. Features
        soxl['return_1h'] = soxl['Close'].pct_change(12)
        soxl['vol_ratio'] = soxl['Volume'].rolling(12).sum() / soxl['Volume'].rolling(78).mean()
        nasdaq['trend_1h'] = nasdaq['Close'].pct_change(12)
        sox_index['trend_1h'] = sox_index['Close'].pct_change(12)
        vix['level'] = vix['Close']

        # 3. Dataset diario
        daily = soxl.resample('1D').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'})
        daily['return_1h'] = soxl['return_1h'].resample('1D').last()
        daily['vol_ratio'] = soxl['vol_ratio'].resample('1D').last()
        daily['nasdaq_trend'] = nasdaq['trend_1h'].resample('1D').last()
        daily['sox_trend'] = sox_index['trend_1h'].resample('1D').last()
        daily['vix_level'] = vix['level'].resample('1D').last()

        # 4. Features y targets
        X = daily[['Open','return_1h','vol_ratio','nasdaq_trend','sox_trend','vix_level']].dropna()
        y = daily[['Low','High','Close']].loc[X.index]

        if X.empty:
            print("⚠️ Sin datos suficientes.")
            return

        # 5. Modelos
        models = {}
        preds = {}
        for target in ['Low','High','Close']:
            dtrain = xgb.DMatrix(X, label=y[target])
            params = {'objective':'reg:squarederror', 'max_depth':5, 'eta':0.1}
            model = xgb.train(params, dtrain, num_boost_round=200)
            models[target] = model

        # 6. Predicción
        dlast = xgb.DMatrix(X.tail(1))
        for target in ['Low','High','Close']:
            preds[target] = float(models[target].predict(dlast))

        print("Predicción:", preds)

        # 7. Guardar en CSV
        csv_filename = "soxl_predictions.csv"
        today = datetime.date.today().strftime("%Y-%m-%d")
        pred_df = pd.DataFrame([preds], index=[today])
        
        file_exists = os.path.exists(csv_filename)
        pred_df.to_csv(csv_filename, mode='a', header=not file_exists)
        print(f"💾 Guardado en {csv_filename}")

    except Exception as e:
        print(f"❌ Error: {e}")

# Ejecución única directa para la nube
if __name__ == "__main__":
    run_prediction()
