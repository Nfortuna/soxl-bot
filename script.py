import os
import datetime
import yfinance as yf
import pandas as pd
import xgboost as xgb

def run_prediction():
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Iniciando proceso de predicción SOXL...")
    csv_filename = "soxl_predictions.csv"
    today = datetime.date.today().strftime("%Y-%m-%d")
    
    # Valores base de contingencia (Mercado cerrado o fin de semana)
    preds = {"Low": 105.44, "High": 116.29, "Close": 113.22}
    es_real = False
    
    try:
        print("📥 Descargando datos desde Yahoo Finance...")
        soxl = yf.download("SOXL", period="60d", interval="5m", group_by='ticker')
        
        if not soxl.empty:
            df_soxl = soxl["SOXL"] if isinstance(soxl.columns, pd.MultiIndex) else soxl
            
            if len(df_soxl) > 50:
                print("📊 Datos de mercado encontrados. Calculando indicadores...")
                df_soxl['return_1h'] = df_soxl['Close'].pct_change(12)
                df_soxl['vol_ratio'] = df_soxl['Volume'].rolling(12).sum() / df_soxl['Volume'].rolling(78).mean()

                daily = df_soxl.resample('1D').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
                
                if len(daily) > 5:
                    X_train = daily[['Open']].dropna()
                    y_train = daily[['Low','High','Close']].loc[X_train.index]
                    
                    models = {}
                    for target in ['Low','High','Close']:
                        dtrain = xgb.DMatrix(X_train, label=y_train[target])
                        model = xgb.train({'objective':'reg:squarederror', 'max_depth':3}, dtrain, num_boost_round=20)
                        models[target] = model
                    
                    dlast = xgb.DMatrix(X_train.tail(1))
                    for target in ['Low','High','Close']:
                        preds[target] = round(float(models[target].predict(dlast)), 2)
                    es_real = True
    except Exception as e:
        print(f"⚠️ Nota de contingencia bursátil: {e}")
        
    print(f"🔮 Predicción final: {preds}")
    
    # Guardar en el historial CSV
    pred_df = pd.DataFrame([preds], index=[today])
    file_exists = os.path.exists(csv_filename)
    pred_df.to_csv(csv_filename, mode='a', header=not file_exists)
    print(f"💾 Archivo escrito con éxito: {csv_filename}")   
    # --- CREAR REPORTE PARA GITHUB ACTIONS ---
    tipo_data = "Datos de Mercado Reales" if es_real else "Valores Base de Contingencia"
    with open("telegram_msg.txt", "w", encoding="utf-8") as f:
        f.write(
            f"🤖 Predicción SOXL Activa\n"
            f"📅 Fecha: {today}\n"
            f"🔹 Estado: {tipo_data}\n\n"
            f"📈 High estimado: {preds['High']}\n"
            f"📉 Low estimado: {preds['Low']}\n"
            f"🏁 Close estimado: {preds['Close']}\n\n"
            f"💾 Historial actualizado en GitHub."
        )
    print("📝 Archivo de texto 'telegram_msg.txt' generado para el bot nativo.")

if __name__ == "__main__":
    run_prediction()
