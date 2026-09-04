import os
import datetime
import yfinance as yf
import pandas as pd
import xgboost as xgb

def run_prediction():
    modo = os.getenv("HORARIO_EJECUCION", "CIERRE")
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Iniciando predicción SOXL con Indicadores Avanzados (Modo: {modo})...")
    
    csv_filename = "soxl_predictions.csv"
    
    # 1. IDENTIFICACIÓN DINÁMICA DE LA FECHA ACTUAL DE MERCADO
    # Descargamos una sola vela rápida de SOXL para ver cuál es el día real activo en Wall Street ahora mismo
    try:
        test_df = yf.download("SOXL", period="1d", interval="5m")
        today = test_df.index[-1].strftime('%Y-%m-%d')
    except Exception:
        today = datetime.date.today().strftime("%Y-%m-%d")
        
    print(f"📅 Fecha operativa identificada para el análisis: {today}")
    
    # Valores base de contingencia (Cambiados a números únicos para detectar fallos drásticos de red)
    preds = {"Low": 1.0, "High": 1.0, "Close": 1.0, "Real": 1.0, "Close Real": 1.0, "Tendencia": "Estable"}
    es_real = False
    
    tickers_indice = [
        "NVDA", "MU", "AMD", "AVGO", "INTC", "AMAT", "TSM", "MRVL", "LRCX", "KLAC", "QCOM", "ASML",
        "TXN", "ADI", "MCHP", "NXPI", "ON", "MPWR", "CRUS", "DIOD", "LSCC", "RMBS", "SLAB", "WOLF",
        "TER", "COHR", "ENTG", "FORM", "ONTO", "MKSI"
    ]
    
    pesos = {
        "NVDA": 0.12, "MU": 0.12, "AMD": 0.12, "AVGO": 0.11, "INTC": 0.06, 
        "AMAT": 0.06, "TSM": 0.06, "MRVL": 0.05, "LRCX": 0.05, "KLAC": 0.05, "QCOM": 0.04, "ASML": 0.02
    }
    for t in tickers_indice:
        if t not in pesos: pesos[t] = 0.0077

    try:
        print("📥 Descargando paquete de datos consolidado desde Yahoo Finance...")
        incluir_premarket = True if modo == "APERTURA" else False
        
        datos = yf.download(tickers_indice, period="45d", interval="5m", group_by='ticker', prepost=incluir_premarket)
        soxl_data = yf.download("SOXL", period="45d", interval="5m", group_by='ticker', prepost=incluir_premarket)
        nasdaq_data = yf.download("^IXIC", period="45d", interval="5m", group_by='ticker', prepost=incluir_premarket)
        vix_data = yf.download("^VIX", period="45d", interval="5m", group_by='ticker', prepost=incluir_premarket)
        
        if not datos.empty and not soxl_data.empty:
            df_soxl = soxl_data["SOXL"] if isinstance(soxl_data.columns, pd.MultiIndex) else soxl_data
            df_nasdaq = nasdaq_data["^IXIC"] if isinstance(nasdaq_data.columns, pd.MultiIndex) else nasdaq_data
            df_vix = vix_data["^VIX"] if isinstance(vix_data.columns, pd.MultiIndex) else vix_data
            
            # Localizar de forma segura las filas del día operativo quitando la zona horaria del índice
            df_soxl.index = df_soxl.index.tz_localize(None)
            hoy_soxl = df_soxl[df_soxl.index.strftime('%Y-%m-%d') == today]
            
            # Si sigue vacío por desfase horaria extrema, forzar el último bloque de datos del dataframe
            if hoy_soxl.empty:
                today = df_soxl.index[-1].strftime('%Y-%m-%d')
                hoy_soxl = df_soxl[df_soxl.index.strftime('%Y-%m-%d') == today]
            
            print(f"📊 Extrayendo métricas intradía para la fecha: {today} (Registros hoy: {len(hoy_soxl)})")
            
            variacion_ponderada_actual = 0.0
            proyeccion_ponderada_cierre = 0.0
            retornos_componentes = []
            
            for ticker in tickers_indice:
                # Comprobar la existencia del ticker en la estructura descargada de yfinance
                if isinstance(datos.columns, pd.MultiIndex) and ticker in datos.columns.levels[0]:
                    df_t = datos[ticker].copy()
                else:
                    df_t = datos.copy()
                    
                df_t.index = df_t.index.tz_localize(None)
                retornos_componentes.append(df_t['Close'].pct_change(12))
                
                hoy_t = df_t[df_t.index.strftime('%Y-%m-%d') == today]
                
                if len(hoy_t) >= 2:
                    precio_apertura_t = float(hoy_t['Open'].iloc[0])
                    precio_actual_t = float(hoy_t['Close'].iloc[-1])
                    
                    var_actual_t = (precio_actual_t - precio_apertura_t) / precio_apertura_t
                    variacion_ponderada_actual += (var_actual_t * pesos[ticker])
                    
                    cambio_reciente = (precio_actual_t - float(hoy_t['Close'].iloc[-2])) / float(hoy_t['Close'].iloc[-2])
                    proyeccion_cierre_t = precio_actual_t * (1 + cambio_reciente * 6)
                    var_proyectada_t = (proyeccion_cierre_t - precio_apertura_t) / precio_apertura_t
                    proyeccion_ponderada_cierre += (var_proyectada_t * pesos[ticker])
            
            # --- CONSTRUCCIÓN SINTÉTICA DE LA IA ---
            df_retornos_historicos = pd.concat(retornos_componentes, axis=1).mean(axis=1)
            df_soxl['index_trend_1h'] = df_retornos_historicos
            df_soxl['vol_ratio'] = df_soxl['Volume'].rolling(12).sum() / df_soxl['Volume'].rolling(78).mean()
            
            df_nasdaq.index = df_nasdaq.index.tz_localize(None)
            df_vix.index = df_vix.index.tz_localize(None)
            df_soxl['nasdaq_trend'] = df_nasdaq['Close'].pct_change(12)
            df_soxl['vix_level'] = df_vix['Close']
            
            daily = df_soxl.resample('1D').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
            daily['index_trend'] = df_soxl['index_trend_1h'].resample('1D').last()
            daily['vol_ratio'] = df_soxl['vol_ratio'].resample('1D').last()
            daily['nasdaq_trend'] = df_soxl['nasdaq_trend'].resample('1D').last()
            daily['vix_level'] = df_soxl['vix_level'].resample('1D').last()
            
            X_train = daily[['Open', 'Volume', 'index_trend', 'vol_ratio', 'nasdaq_trend', 'vix_level']].dropna()
            y_train = daily[['Low','High','Close']].loc[X_train.index]
            
            if len(X_train) > 5:
                models = {}
                for target in ['Low','High','Close']:
                    dtrain = xgb.DMatrix(X_train, label=y_train[target])
                    model = xgb.train({'objective':'reg:squarederror', 'max_depth':4}, dtrain, num_boost_round=25)
                    models[target] = model
                
                dlast = xgb.DMatrix(X_train.tail(1))
                for target in ['Low','High','Close']:
                    preds[target] = round(float(models[target].predict(dlast)), 2)
                
                if len(hoy_soxl) >= 1:
                    precio_apertura_soxl = float(hoy_soxl['Open'].iloc[0])
                    preds["Real"] = round(precio_apertura_soxl * (1 + (variacion_ponderada_actual * 3)), 2)
                    preds["Close Real"] = round(precio_apertura_soxl * (1 + (proyeccion_ponderada_cierre * 3)), 2)
                    preds["Tendencia"] = "Alza" if preds["Close"] >= precio_apertura_soxl else "Baja"
                    es_real = True

    except Exception as e:
        print(f"❌ Excepción crítica capturada en consola: {e}")
        
    print(f"🔮 Valores Calculados del Modelo: {preds}")
    
    # Inserción de la nueva fila de datos reales en el CSV histórico
    id_registro = f"{today}_{modo}"
    pred_df = pd.DataFrame([preds], index=[id_registro])
    file_exists = os.path.exists(csv_filename)
    pred_df.to_csv(csv_filename, mode='a', header=not file_exists)
    
    encabezado = "☀️ *REPORTE PRE-MERCADO SOXL*" if modo == "APERTURA" else "📉 *REPORTE PRE-CIERRE SOXL*"
    icon_tendencia = "🟢" if preds["Tendencia"] == "Alza" else "🔴"
    tipo_data = "Cálculo Real Basado en 30 Empresas" if es_real else "⚠️ Valores de Contingencia Activados por Error"
    
    with open("telegram_msg.txt", "w", encoding="utf-8") as f:
        f.write(
            f"{encabezado}\n"
            f"📅 Fecha de Análisis: {today}\n"
            f"🔹 Estado: {tipo_data}\n"
            f"{icon_tendencia} *Tendencia:* {preds['Tendencia']}\n\n"
            f"📈 High estimado: {preds['High']}\n"
            f"📉 Low estimado: {preds['Low']}\n"
            f"🏁 Close estimado (IA): {preds['Close']}\n"
            f"📊 Real (30 Empresas): {preds['Real']}\n"
            f"🎯 Close Real (Proyección 30): {preds['Close Real']}\n\n"
            f"💾 Base de datos unificada en GitHub."
        )

if __name__ == "__main__":
    run_prediction()
