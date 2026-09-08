import os
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb

def descargar_activo_seguro(ticker, period="60d", interval="5m", prepost=False):
    """Descarga datos intradía y normaliza forzosamente las columnas para evitar MultiIndex."""
    try:
        df = yf.download(ticker, period=period, interval=interval, prepost=prepost, progress=False)
        if df.empty:
            print(f"⚠️ Alerta: Yahoo Finance devolvió datos vacíos para {ticker}")
            return pd.DataFrame()
        
        # Normalización estructural de Pandas
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.columns = [str(col).capitalize() for col in df.columns]
        columnas_necesarias = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        if not all(col in df.columns for col in columnas_necesarias):
            df = df.rename(columns=lambda x: str(x).split('_')[-1].capitalize())
            
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        return df[columnas_necesarias]
    except Exception as e:
        print(f"❌ Error crítico al descargar o procesar el activo {ticker}: {e}")
        return pd.DataFrame()

def run_prediction():
    modo = os.getenv("HORARIO_EJECUCION", "CIERRE")
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Iniciando predicción SOXL Diaria Avanzada (Modo: {modo})...")
    
    csv_filename = "soxl_predictions.csv"
    
    # Identificar el día operativo dinámico real en Wall Street
    try:
        test_df = yf.download("SOXL", period="1d", interval="5m")
        today = test_df.index[-1].strftime('%Y-%m-%d')
    except Exception:
        today = datetime.date.today().strftime("%Y-%m-%d")
        
    print(f"📅 Fecha operativa identificada para el análisis diario: {today}")
    
    preds = {"Low": 0.0, "High": 0.0, "Close": 0.0, "Real": 0.0, "Close Real": 0.0, "Tendencia": "Estable"}
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
        print("📥 Descargando paquete de datos unificado...")
        incluir_premarket = True if modo == "APERTURA" else False
        
        df_soxl = descargar_activo_seguro("SOXL", period="60d", interval="5m", prepost=incluir_premarket)
        df_nasdaq = descargar_activo_seguro("^IXIC", period="60d", interval="5m", prepost=incluir_premarket)
        df_vix = descargar_activo_seguro("^VIX", period="60d", interval="5m", prepost=incluir_premarket)
        
        if not df_soxl.empty and not df_nasdaq.empty and not df_vix.empty:
            hoy_soxl = df_soxl[df_soxl.index.strftime('%Y-%m-%d') == today]
            if hoy_soxl.empty:
                today = df_soxl.index[-1].strftime('%Y-%m-%d')
                hoy_soxl = df_soxl[df_soxl.index.strftime('%Y-%m-%d') == today]
            
            variacion_ponderada_actual = 0.0
            proyeccion_ponderada_cierre = 0.0
            retornos_componentes = []
            
            for ticker in tickers_indice:
                df_t = descargar_activo_seguro(ticker, period="60d", interval="5m", prepost=incluir_premarket)
                if not df_t.empty:
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
            
            # --- MODELADO XGBOOST ---
            df_retornos_historicos = pd.concat(retornos_componentes, axis=1).mean(axis=1)
            df_soxl['index_trend_1h'] = df_retornos_historicos
            df_soxl['vol_ratio'] = df_soxl['Volume'].rolling(12).sum() / df_soxl['Volume'].rolling(78).mean()
            
            df_nasdaq = df_nasdaq.reindex(df_soxl.index, method='ffill')
            df_vix = df_vix.reindex(df_soxl.index, method='ffill')
            df_soxl['nasdaq_trend'] = df_nasdaq['Close'].pct_change(12)
            df_soxl['vix_level'] = df_vix['Close']
            
            daily = df_soxl.resample('1D').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
            daily['index_trend'] = df_soxl['index_trend_1h'].resample('1D').last()
            daily['vol_ratio'] = df_soxl['vol_ratio'].resample('1D').last()
            daily['nasdaq_trend'] = df_soxl['nasdaq_trend'].resample('1D').last()
            daily['vix_level'] = df_vix['vix_level'].resample('1D').last() if 'vix_level' in df_soxl.columns else df_vix['Close'].resample('1D').last()
            
            columnas_features = ['Open', 'Volume', 'index_trend', 'vol_ratio', 'nasdaq_trend', 'vix_level']
            df_limpio = daily[columnas_features + ['Close', 'High', 'Low']].dropna()
            
            X = df_limpio[columnas_features]
            
            if len(X) > 5:
                for target in ['Low','High','Close']:
                    y_target = df_limpio[target]
                    dtrain = xgb.DMatrix(X, label=y_target)
                    model = xgb.train({'objective':'reg:squarederror', 'max_depth':4}, dtrain, num_boost_round=25)
                    preds[target] = round(float(model.predict(xgb.DMatrix(X.tail(1)))), 2)
                
                if len(hoy_soxl) >= 1:
                    precio_apertura_soxl = float(hoy_soxl['Open'].iloc[0])
                    preds["Real"] = round(precio_apertura_soxl * (1 + (variacion_ponderada_actual * 3)), 2)
                    preds["Close Real"] = round(precio_apertura_soxl * (1 + (proyeccion_ponderada_cierre * 3)), 2)
                    preds["Tendencia"] = "Alza" if preds["Close"] >= precio_apertura_soxl else "Baja"
                    es_real = True

    except Exception as e:
        print(f"❌ Error en procesamiento diario interno: {e}")
        
    print(f"🔮 Resultados Diarios: {preds}")
    
    # Inserción en el CSV
    id_registro = f"{today}_{modo}"
    pred_df = pd.DataFrame([preds], index=[id_registro])
    file_exists = os.path.exists(csv_filename)
    pred_df.to_csv(csv_filename, mode='a', header=not file_exists)
    
    encabezado = "☀️ REPORTE PRE-MERCADO SOXL" if modo == "APERTURA" else "📉 REPORTE PRE-CIERRE SOXL"
    icon_tendencia = "🟢" if preds["Tendencia"] == "Alza" else "🔴"
    tipo_data = "Calculo Puro Subyacente (30 Empresas)" if es_real else "⚠️ Valores de Contingencia por Fallo"
    
    with open("telegram_msg.txt", "w", encoding="utf-8") as f:
        f.write(
            f"{encabezado}\n"
            f"📅 Fecha de Analisis: {today}\n"
            f"🔹 Estado: {tipo_data}\n"
            f"{icon_tendencia} Tendencia del dia: {preds['Tendencia']}\n\n"
            f"📈 High estimado: {preds['High']}\n"
            f"📉 Low estimado: {preds['Low']}\n"
            f"🏁 Close estimado (IA): {preds['Close']}\n"
            f"📊 Real (30 Empresas): {preds['Real']}\n"
            f"🎯 Close Real (Proyeccion 30): {preds['Close Real']}\n\n"
            f"💾 Historial de 30 activos actualizado en GitHub."
        )

if __name__ == "__main__":
    run_prediction()
