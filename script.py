import os
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb

def descargar_activo_seguro(ticker, period="60d", interval="5m", prepost=False):
    """Downloads intraday bars and flattens column structures safely."""
    try:
        df = yf.download(ticker, period=period, interval=interval, prepost=prepost, progress=False)
        if df.empty:
            print(f"⚠️ Warning: Empty table received for {ticker}")
            return pd.DataFrame()
        
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
        print(f"❌ Critical download error for {ticker}: {e}")
        return pd.DataFrame()

def run_prediction():
    modo = os.getenv("HORARIO_EJECUCION", "CIERRE")
    now_time = datetime.datetime.now()
    print(f"[{now_time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Sincronizacion Cuantitativa SOXL (Modo: {modo})...")
    
    csv_filename = "soxl_predictions.csv"
    today_str = now_time.strftime('%Y-%m-%d')
    
    # CRITICAL FIX: Safe global baseline initialization outside of the try block
    preds = {"Low": 0.0, "High": 0.0, "Close": 0.0, "Real": 0.0, "Tendencia": "Estable"}
    es_real = False
    
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
        print("📥 Downloading unified market block via single corporate API call...")
        incluir_premarket = True if modo == "APERTURA" else False
        raw_data = yf.download(all_tickers, period="60d", interval="5m", prepost=incluir_premarket, group_by='ticker', progress=False)
        
        if raw_data.empty:
            print("❌ Error: Yahoo Finance mass download blanked out.")
            return

        # CRITICAL FIX: Flawless cross-sectional extraction handler for structural MultiIndex tables
        def extraer_tabla(ticker):
            if isinstance(raw_data.columns, pd.MultiIndex):
                if ticker in raw_data.columns.levels[0]:
                    df = raw_data.xs(ticker, axis=1, level=0).copy()
                else:
                    return pd.DataFrame()
            else:
                df = raw_data[ticker].copy() if ticker in raw_data.columns else pd.DataFrame()
                
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
            print("❌ Error: SOXL mapping block failed.")
            return

        hoy_date = df_soxl.index[-1].date()
        today_str = hoy_date.strftime('%Y-%m-%d')
        print(f"📅 Operational target date: {today_str}")
        
        # --- TRUE RENORMALIZED MATRIX WEIGHTING FIX ---
        retornos_componentes = []
        tickers_descargados = []
        suma_pesos_validos = 0.0
        
        for t in tickers_indice:
            df_t = extraer_tabla(t)
            if not df_t.empty:
                df_t = df_t.reindex(df_soxl.index, method='ffill')
                retornos_componentes.append(df_t['Close'].pct_change(12))
                tickers_descargados.append(t)
                suma_pesos_validos += pesos_base[t]
        
        pesos_normalizados = {k: pesos_base[k] / suma_pesos_validos for k in tickers_descargados}
        pesos_array = [pesos_normalizados[t] for t in tickers_descargados]
        
        df_concat = pd.concat(retornos_componentes, axis=1)
        df_retornos_historicos = df_concat.mul(pesos_array, axis=1).sum(axis=1)
        
        df_soxl['index_trend_1h'] = df_retornos_historicos
        df_soxl['vol_ratio'] = df_soxl['Volume'].rolling(12).sum() / df_soxl['Volume'].rolling(78).mean()
        df_soxl['nasdaq_trend'] = df_nasdaq['Close'].pct_change(12)
        df_soxl['vix_level'] = df_vix['Close']
        df_soxl['Fecha'] = df_soxl.index.date
        df_soxl['Hora_Minuto'] = df_soxl.index.time
        
        hora_corte = df_soxl.index[-1].time()
        print(f"⏱️ Filtering macro history slices strictly at: {hora_corte}")
        snapshot_historico = df_soxl[df_soxl['Hora_Minuto'] == hora_corte].copy()
        
        fila_hoy = snapshot_historico[snapshot_historico['Fecha'] == hoy_date]
        columnas_features = ['Open', 'Volume', 'index_trend_1h', 'vol_ratio', 'nasdaq_trend', 'vix_level']
        
        daily_targets = df_soxl.resample('1D').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()
        snapshot_historico['Target_High'] = snapshot_historico['Fecha'].map(daily_targets['High'] - daily_targets['Open'])
        snapshot_historico['Target_Low'] = snapshot_historico['Fecha'].map(daily_targets['Low'] - daily_targets['Open'])
        snapshot_historico['Target_Close'] = snapshot_historico['Fecha'].map(daily_targets['Close'] - daily_targets['Open'])
        
        df_entrenamiento = snapshot_historico[snapshot_historico['Fecha'] != hoy_date]
        df_entrenamiento = df_entrenamiento[columnas_features + ['Target_High', 'Target_Low', 'Target_Close']].dropna()
        
        X = df_entrenamiento[columnas_features]
        x_last = fila_hoy[columnas_features].tail(1)
        
        if len(X) >= 20 and not x_last.empty:
            print(f"🧠 Training out-of-sample models via {len(X)} historic snapshots...")
            params = {
                'objective': 'reg:squarederror', 'max_depth': 3, 'eta': 0.1,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 1.0
            }
            dlast = xgb.DMatrix(x_last)
            precio_apertura_hoy = float(fila_hoy['Open'].iloc[-1])
            
            for target in ['High', 'Low', 'Close']:
                dtrain = xgb.DMatrix(X, label=df_entrenamiento[f'Target_{target}'])
                model = xgb.train(params, dtrain, num_boost_round=25)
                pred_variacion = float(model.predict(dlast))
                preds[target] = round(precio_apertura_hoy + pred_variacion, 2)
                
                if target == 'Close':
                    preds["Tendencia"] = "Alza" if pred_variacion > 0 else "Baja"
            
            preds["Real"] = round(float(df_soxl['Close'].iloc[-1]), 2)
            es_real = True
        else:
            print(f"⏳ Insufficient historical alignment rows available ({len(X)}/20). Awaiting deeper log parsing.")

    except Exception as e:
        print(f"❌ Internal calculation anomaly caught: {e}")
        
    print(f"🔮 Clean Micro-Target Output Profile: {preds}")
    
    try:
        id_registro = f"{today_str}_{modo}"
        pred_df = pd.DataFrame([preds], index=[id_registro])
        file_exists = os.path.exists(csv_filename)
        pred_df.to_csv(csv_filename, mode='a', header=not file_exists)
        print("💾 Macro log parsed securely to data array.")
    except Exception as csv_err:
        print(f"⚠️ Write execution block: {csv_err}")
    
    encabezado = "☀️ REPORTE PRE-MERCADO SOXL" if modo == "APERTURA" else "📉 REPORTE PRE-CIERRE SOXL"
    icon_tendencia = "🟢" if preds["Tendencia"] == "Alza" else "🔴"
    tipo_data = "Modelado Cuantitativo Out-of-Sample Saneado" if es_real else "⚠️ Valores de Contingencia por Muestras"
    
    with open("telegram_msg.txt", "w", encoding="utf-8") as f:
        f.write(
            f"{encabezado}\n"
            f"📅 Fecha de Analisis: {today_str}\n"
            f"🔹 Estado: {tipo_data}\n"
            f"{icon_tendencia} Tendencia del dia: {preds['Tendencia']}\n\n"
            f"📈 High estimado: {preds['High']}\n"
            f"📉 Low estimado: {preds['Low']}\n"
            f"🏁 Close estimado (IA): {preds['Close']}\n"
            f"📊 Real Actual: ${preds['Real']}\n\n"
            f"💾 Historial de 30 activos alineado, renormalizado y limpio."
        )

if __name__ == "__main__":
    run_prediction()
