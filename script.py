import os
import datetime
import yfinance as yf
import pandas as pd
import xgboost as xgb

def run_prediction():
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Iniciando proceso de predicción SOXL...")
    try:
        # 1. Descargar datos (Periodo correcto de 60 días para intervalos de 5 minutos)
        print("📥 Descargando datos desde Yahoo Finance...")
        soxl = yf.download("SOXL", period="60d", interval="5m", group_by='ticker')["SOXL"]
        nasdaq = yf.download("^IXIC", period="60d", interval="5m", group_by='ticker')["^IXIC"]
        sox_index = yf.download("^SOX", period="60d", interval="5m", group_by='ticker')["^SOX"]
        vix = yf.download("^VIX", period="60d", interval="5m", group_by='ticker')["^VIX"]

        if soxl.empty or nasdaq.empty:
            print("❌ Error: No se pudieron obtener datos de Yahoo Finance.")
            return

        # 2. Construcción de Features
        soxl['return_1h'] = soxl['Close'].pct_change(12)
        soxl['vol_ratio'] = soxl['Volume'].rolling(12).sum() / soxl['Volume'].rolling(78).mean()
        nasdaq['trend_1h'] = nasdaq['Close'].pct_change(12)
        sox_index['trend_1h'] = sox_index['Close'].pct_change(12)
        vix['level'] = vix['Close']

        # 3. Dataset diario resampleado
        daily = soxl.resample('1D').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'})
        daily['return_1h'] = soxl['return_1h'].resample('1D').last()
        daily['vol_ratio'] = soxl['vol_ratio'].resample('1D').last()
        daily['nasdaq_trend'] = nasdaq['trend_1h'].resample('1D').last()
        daily['sox_trend'] = sox_index['trend_1h'].resample('1D').last()
        daily['vix_level'] = vix['level'].resample('1D').last()

        # Limpiar filas vacías para el entrenamiento
        X_train = daily[['Open','return_1h','vol_ratio','nasdaq_trend','sox_trend','vix_level']].dropna()
        y_train = daily[['Low','High','Close']].loc[X_train.index]

        if X_train.empty:
            print("⚠️ Matriz de entrenamiento vacía. Verificando estructura de datos...")
            return

        # 5. Entrenar modelos XGBoost
        print("🧠 Entrenando modelos de Inteligencia Artificial...")
        models = {}
        preds = {}
        for target in ['Low','High','Close']:
            dtrain = xgb.DMatrix(X_train, label=y_train[target])
            params = {'objective':'reg:squarederror', 'max_depth':5, 'eta':0.1}
            model = xgb.train(params, dtrain, num_boost_round=100)
            models[target] = model

        # 6. Tomar el último registro disponible con datos completos para la predicción
        dlast = xgb.DMatrix(X_train.tail(1))
        for target in ['Low','High','Close']:
            preds[target] = round(float(models[target].predict(dlast)[0]), 2)

        print("🔮 Predicción obtenida con éxito:", preds)

        # 7. Guardar en CSV de manera persistente
        csv_filename = "soxl_predictions.csv"
        # Usamos la fecha del último dato real procesado
        last_date = X_train.index[-1].strftime("%Y-%m-%d")
        pred_df = pd.DataFrame([preds], index=[last_date])
        
        file_exists = os.path.exists(csv_filename)
        pred_df.to_csv(csv_filename, mode='a', header=not file_exists)
        print(f"💾 ¡Archivo guardado exitosamente como {csv_filename}!")

    except Exception as e:
        print(f"❌ Ocurrió un fallo general en el script: {e}")

if __name__ == "__main__":
    run_prediction()
