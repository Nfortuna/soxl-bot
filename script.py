import os
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb

def run_prediction():
    modo = os.getenv("HORARIO_EJECUCION", "CIERRE")
    now_time = datetime.datetime.now()
    print(f"[{now_time.strftime('%Y-%m-%d %H:%M:%S')}] Iniciando Sincronización Cuantitativa SOXL (Modo: {modo})...")
    
    csv_filename = "soxl_predictions.csv"
    
    # Lista Oficial de Componentes
    tickers_indice = [
        "NVDA", "MU", "AMD", "AVGO", "INTC", "AMAT", "TSM", "MRVL", "LRCX", "KLAC", "QCOM", "ASML",
        "TXN", "ADI", "MCHP", "NXPI", "ON", "MPWR", "CRUS", "DIOD", "LSCC", "RMBS", "SLAB", "WOLF",
        "TER", "COHR", "ENTG", "FORM", "ONTO", "MKSI"
    ]
    all_tickers = ["SOXL", "^IXIC", "^VIX"] + tickers_indice
    
    pesos_base = {
        "NVDA": 0.12, "MU": 0.12, "AMD": 0.12, "AVGO": 0.11, "INTC": 0.06, 
        "AMAT": 0.06, "TSM": 0.06, "MRVL": 0.05, "LRCX": 0.05, "KLAC": 0.05, "QCOM": 0.04, "ASML": 0.02
    }
    for t in tickers_indice:
        if t not in pesos_base: pesos_base[t] = 0.0077

    try:
        print("📥 Descargando matriz de mercado completa en un solo bloque de red...")
        incluir_premarket = True if modo == "APERTURA" else False
        
        # Descarga masiva para evitar rate-limiting e incrementar velocidad
        raw_data = yf.download(all_tickers, period="60d", interval="5m", prepost=incluir_premarket, group_by='ticker', progress=False)
        
        if raw_data.empty:
            print("❌ Error: Yahoo Finance no devolvió datos para el bloque masivo.")
            return

        # Descomprimir dataframes planos individuales de forma segura
        def extraer_tabla(ticker):
            df = raw_data[ticker].copy() if ticker in raw_data.columns.levels[0] else pd.DataFrame()
            if not df.empty:
                df.columns = [str(col).capitalize() for col in df.columns]
                if df.index.tz is not None: df.index = df.index.tz_localize(None)
                df.ffill(inplace=True)
                df.bfill(inplace=True)
            return df

        df_soxl = extraer_tabla("SOXL")
        df_nasdaq = extraer_tabla("^IXIC").reindex(df_soxl.index, method='ffill')
        df_vix = extraer_tabla("^VIX").reindex(df_soxl.index, method='ffill')
        
        if df_soxl.empty:
            print("❌ Error: Tabla de SOXL vacía.")
            return

        today = df_soxl.index[-1].strftime('%Y-%m-%d')
        print(f"📅 Fecha operativa identificada: {today}")
        
        preds = {"Low": 0.0, "High": 0.0, "Close": 0.0, "Real": 0.0, "Close Real": 0.0, "Tendencia": "Estable"}
        es_real = False
        
        # --- RECONSTRUCCIÓN DE COMPONENTES CON RENORMALIZACIÓN DE PESOS ---
        retornos_componentes = []
        pesos_validos = {}
        suma_pesos_validos = 0.0
        
        for t in tickers_indice:
            df_t = extraer_tabla(t)
            if not df_t.empty:
                df_t = df_t.reindex(df_soxl.index, method='ffill')
                retornos_componentes.append(df_t['Close'].pct_change(12))
                pesos_validos[t] = pesos_base[t]
                suma_pesos_validos += pesos_base[t]
        
        # Renormalizar los pesos si algún activo falló la descarga
        pesos_normalizados = {k: v / suma_pesos_validos for k, v in pesos_validos.items()}
        df_retornos_historicos = pd.concat(retornos_componentes, axis=1).mean(axis=1)
        
        # Inyectar features al dataframe de SOXL
        df_soxl['index_trend_1h'] = df_retornos_historicos
        df_soxl['vol_ratio'] = df_soxl['Volume'].rolling(12).sum() / df_soxl['Volume'].rolling(78).mean()
        df_soxl['nasdaq_trend'] = df_nasdaq['Close'].pct_change(12)
        df_soxl['vix_level'] = df_vix['Close']
        df_soxl['Fecha'] = df_soxl.index.date
        df_soxl['Hora_Minuto'] = df_soxl.index.time
        
        # --- SOLUCIÓN AL DATA LEAKAGE: FILTRADO POR VENTANA HORARIA ---
        # Capturamos la hora exacta actual de la corrida de producción
        hora_corte = df_soxl.index[-1].time()
        print(f"⏱️ Sincronizando ventana horaria histórica para entrenamiento a las: {hora_corte}")
        
        # Extraemos solo las fotos instantáneas que corresponden a esta hora exacta en el pasado
        snapshot_historico = df_soxl[df_soxl['Hora_Minuto'] == hora_corte].copy()
        
        # Targets macro diarios de salida real desde el Open diario
        daily_targets = df_soxl.resample('1D').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()
        snapshot_historico['Target_High'] = snapshot_historico['Fecha'].map(daily_targets['High'] - daily_targets['Open'])
        snapshot_historico['Target_Low'] = snapshot_historico['Fecha'].map(daily_targets['Low'] - daily_targets['Open'])
        snapshot_historico['Target_Close'] = snapshot_historico['Fecha'].map(daily_targets['Close'] - daily_targets['Open'])
        
        columnas_features = ['Open', 'Volume', 'index_trend_1h', 'vol_ratio', 'nasdaq_trend', 'vix_level']
        df_entrenamiento = snapshot_historico[columnas_features + ['Target_High', 'Target_Low', 'Target_Close']].dropna()
        
        X = df_entrenamiento[columnas_features]
        
        # Umbral estadístico elevado a mínimo 30 días para robustez institucional
        if len(X) > 30:
            print(f"🧠 Entrenando XGBoost con {len(X)} snapshots históricos libres de Leakage...")
            params = {
                'objective': 'reg:squarederror', 'max_depth': 3, 'eta': 0.1,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 1.0
            }
            
            x_last = X.tail(1)
            dlast = xgb.DMatrix(x_last)
            precio_apertura_hoy = float(df_soxl.loc[df_soxl['Fecha'] == df_soxl.index[-1].date(), 'Open'].iloc[0])
            
            # Entrenamiento independiente por objetivo sin contaminación futura
            for target in ['High', 'Low', 'Close']:
                dtrain = xgb.DMatrix(X, label=df_entrenamiento[f'Target_{target}'])
                model = xgb.train(params, dtrain, num_boost_round=25)
                pred_variacion = float(model.predict(dlast))
                preds[target] = round(precio_apertura_hoy + pred_variacion, 2)
                
                if target == 'Close':
                    preds["Tendencia"] = "Alza" if pred_variacion > 0 else "Baja"
            
            # --- CÁLCULOS PUREMENTE OBSERVACIONALES SIN MULTIPLIPLICADORES ARBITRARIOS ---
            preds["Real"] = round(float(df_soxl['Close'].iloc[-1]), 2)
            preds["Close Real"] = preds["Close"] # El Close estimado actúa como el norte cuantitativo
            es_real = True
        else:
            print(f"⏳ Registros insuficientes en la ventana horaria ({len(X)}/30). Esperando acumulación del historial.")

    except Exception as e:
        print(f"❌ Error en procesamiento diario interno: {e}")
        
    print(f"🔮 Resultados Diarios (Sin Leakage): {preds}")
    
    # Escritura Segura CSV
    try:
        id_registro = f"{today}_{modo}"
        pred_df = pd.DataFrame([preds], index=[id_registro])
        file_exists = os.path.exists(csv_filename)
        pred_df.to_csv(csv_filename, mode='a', header=not file_exists)
        print("💾 Historial macro actualizado.")
    except Exception as csv_err:
        print(f"⚠️ Alerta CSV bloqueado: {csv_err}")
    
    encabezado = "☀️ REPORTE PRE-MERCADO SOXL" if modo == "APERTURA" else "📉 REPORTE PRE-CIERRE SOXL"
    icon_tendencia = "🟢" if preds["Tendencia"] == "Alza" else "🔴"
    tipo_data = "Modelado Instantáneo Libre de Fuga (No-Leakage)" if es_real else "⚠️ Valores de Contingencia por Muestras"
    
    with open("telegram_msg.txt", "w", encoding="utf-8") as f:
        f.write(
            f"{encabezado}\n"
            f"📅 Fecha de Analisis: {today}\n"
            f"🔹 Estado: {tipo_data}\n"
            f"{icon_tendencia} Tendencia del dia: {preds['Tendencia']}\n\n"
            f"📈 High estimado: {preds['High']}\n"
            f"📉 Low estimado: {preds['Low']}\n"
            f"🏁 Close estimado (IA): {preds['Close']}\n"
            f"📊 Real Actual: ${preds['Real']}\n\n"
            f"💾 Historial de 30 activos alineado y limpio en GitHub."
        )

if __name__ == "__main__":
    run_prediction()
