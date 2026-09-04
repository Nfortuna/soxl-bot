import os
import datetime
import yfinance as yf
import pandas as pd
import xgboost as xgb

def run_prediction():
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Iniciando predicción SOXL con el Índice Completo (30 Activos)...")
    csv_filename = "soxl_predictions.csv"
    today = datetime.date.today().strftime("%Y-%m-%d")
    
    # Valores de contingencia base incluyendo las nuevas métricas
    preds = {"Low": 105.44, "High": 116.29, "Close": 113.22, "Real": 112.50, "Close Real": 112.90}
    es_real = False
    
    tickers_indice = [
        "NVDA", "MU", "AMD", "AVGO", "INTC", "AMAT", "TSM", "MRVL", "LRCX", "KLAC", "QCOM", "ASML",
        "TXN", "ADI", "MCHP", "NXPI", "ON", "MPWR", "CRUS", "DIOD", "LSCC", "RMBS", "SLAB", "WOLF",
        "TER", "COHR", "ENTG", "FORM", "ONTO", "MKSI"
    ]
    
    # Ponderaciones aproximadas normalizadas
    pesos = {
        "NVDA": 0.12, "MU": 0.12, "AMD": 0.12, "AVGO": 0.11, 
        "INTC": 0.06, "AMAT": 0.06, "TSM": 0.06, "MRVL": 0.05, 
        "LRCX": 0.05, "KLAC": 0.05, "QCOM": 0.04, "ASML": 0.02
    }
    for t in tickers_indice:
        if t not in pesos:
            pesos[t] = 0.0077

    try:
        print("📥 Descargando historial de los 30 activos simultáneamente...")
        datos = yf.download(tickers_indice, period="60d", interval="5m", group_by='ticker')
        soxl_data = yf.download("SOXL", period="60d", interval="5m", group_by='ticker')
        
        if not datos.empty and not soxl_data.empty:
            df_soxl = soxl_data["SOXL"] if isinstance(soxl_data.columns, pd.MultiIndex) else soxl_data
            
            print("🧮 Ensamblando la tendencia integrada de los 30 componentes...")
            retornos_componentes = []
            retornos_actuales_ponderados = []
            
            for ticker in tickers_indice:
                if ticker in datos.columns.levels if isinstance(datos.columns, pd.MultiIndex) else ticker in datos.columns:
                    df_t = datos[ticker] if isinstance(datos.columns, pd.MultiIndex) else datos
                    retorno_1h = df_t['Close'].pct_change(12)
                    retornos_componentes.append(retorno_1h)
                    
                    if not df_t.empty:
                        ultima_variacion_accion = float(df_t['Close'].pct_change(12).iloc[-1])
                        retornos_actuales_ponderados.append(ultima_variacion_accion * pesos[ticker])
            
            df_retornos_historicos = pd.concat(retornos_componentes, axis=1).mean(axis=1)
            df_soxl['index_trend_1h'] = df_retornos_historicos
            df_soxl['vol_ratio'] = df_soxl['Volume'].rolling(12).sum() / df_soxl['Volume'].rolling(78).mean()
            
            daily = df_soxl.resample('1D').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()
            daily['index_trend'] = df_soxl['index_trend_1h'].resample('1D').last()
            daily['vol_ratio'] = df_soxl['vol_ratio'].resample('1D').last()
            
            X_train = daily[['Open', 'index_trend', 'vol_ratio']].dropna()
            y_train = daily[['Low','High','Close']].loc[X_train.index]
            
            if len(X_train) > 5:
                print("🧠 Entrenando los modelos XGBoost...")
                models = {}
                for target in ['Low','High','Close']:
                    dtrain = xgb.DMatrix(X_train, label=y_train[target])
                    model = xgb.train({'objective':'reg:squarederror', 'max_depth':4, 'eta':0.1}, dtrain, num_boost_round=30)
                    models[target] = model
                
                dlast = xgb.DMatrix(X_train.tail(1))
                for target in ['Low','High','Close']:
                    preds[target] = round(float(models[target].predict(dlast)), 2)
                
                # --- CÁLCULO DE MÉTRICAS REALES Y PROYECTADAS ---
                precio_apertura_soxl = float(df_soxl['Open'].resample('1D').first().iloc[-1])
                precio_actual_soxl = float(df_soxl['Close'].iloc[-1])
                variacion_mercado_30 = sum(retornos_actuales_ponderados)
                
                # 1. Real: Precio real actual de mercado en el momento de la corrida
                preds["Real"] = round(precio_actual_soxl, 2)
                
                # 2. Close Real: Proyección matemática de cierre basada en la inercia 3X de los componentes
                preds["Close Real"] = round(precio_apertura_soxl * (1 + (variacion_mercado_30 * 3)), 2)
                es_real = True

    except Exception as e:
        print(f"⚠️ Nota de contingencia (Procesando con pesos base): {e}")
        
    print(f"🔮 Entregable Final: {preds}")
    
    # Guardar en el archivo CSV histórico de GitHub
    pred_df = pd.DataFrame([preds], index=[today])
    file_exists = os.path.exists(csv_filename)
    pred_df.to_csv(csv_filename, mode='a', header=not file_exists)
    
    # Formatear reporte para el mensaje de Telegram
    tipo_data = "Datos del Índice Completo (30 componentes)" if es_real else "Valores Base de Contingencia"
    with open("telegram_msg.txt", "w", encoding="utf-8") as f:
        f.write(
            f"🤖 Predicción SOXL Activa\n"
            f"📅 Fecha: {today}\n"
            f"🔹 Estado: {tipo_data}\n\n"
            f"📈 High estimado: {preds['High']}\n"
            f"📉 Low estimado: {preds['Low']}\n"
            f"🏁 Close estimado (IA): {preds['Close']}\n"
            f"📊 Real actual (SOXL): {preds['Real']}\n"
            f"🎯 Close Real (Proyección 30 emp.): {preds['Close Real']}\n\n"
            f"💾 Historial de 30 activos actualizado en GitHub."
        )

if __name__ == "__main__":
    run_prediction()
