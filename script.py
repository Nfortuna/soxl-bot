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
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting advanced macro-model (Mode: {modo})...")
    
    csv_filename = "soxl_predictions.csv"
    
    try:
        test_df = yf.download("SOXL", period="1d", interval="5m")
        today = test_df.index[-1].strftime('%Y-%m-%d')
    except Exception:
        today = datetime.date.today().strftime("%Y-%m-%d")
        
    print(f"📅 Operational target date: {today}")
    
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
        print("📥 Requesting unified daily master block metrics...")
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
            
            # --- CONSOLIDATING THE FEATURE ENVIRONMENT ---
            df_retornos_historicos = pd.concat(retornos_componentes, axis=1).mean(axis=1)
            df_soxl['index_trend_1h'] = df_retornos_historicos
            df_soxl['vol_ratio'] = df_soxl['Volume'].rolling(12).sum() / df_soxl['Volume'].rolling(78).mean()
            
            df_nasdaq = df_nasdaq.reindex(df_soxl.index, method='ffill')
            df_vix = df_vix.reindex(df_soxl.index, method='ffill')
            df_soxl['nasdaq_trend'] = df_nasdaq['Close'].pct_change(12)
            df_soxl['vix_level'] = df_vix['Close']
            
            # Macro Day-Resampling Aggregation
            daily = df_soxl.resample('1D').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
            daily['index_trend'] = df_soxl['index_trend_1h'].resample('1D').last()
            daily['vol_ratio'] = df_soxl['vol_ratio'].resample('1D').last()
            daily['nasdaq_trend'] = df_soxl['nasdaq_trend'].resample('1D').last()
            daily['vix_level'] = df_soxl['vix_level'].resample('1D').last()
            
            # --- TARGET UPGRADE: TARGET IS NET VARIATION FROM THE OPEN ($ CHANGE) ---
            daily['Target_High'] = daily['High'] - daily['Open']
            daily['Target_Low'] = daily['Low'] - daily['Open']
            daily['Target_Close'] = daily['Close'] - daily['Open']
            
            columnas_features = ['Open', 'Volume', 'index_trend', 'vol_ratio', 'nasdaq_trend', 'vix_level']
            df_limpio = daily[columnas_features + ['Target_High', 'Target_Low', 'Target_Close']].dropna()
            
            X = df_limpio[columnas_features]
            
            if len(X) > 5:
                # Institutional Parameters matching the updated scalper setup
                params = {
                    'objective': 'reg:squarederror',
                    'max_depth': 4,
                    'eta': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_lambda': 1.0
                }
                
                # Predict macro net movement offsets independently
                dtrain_high = xgb.DMatrix(X, label=df_limpio['Target_High'])
                model_high = xgb.train(params, dtrain_high, num_boost_round=25)
                
                dtrain_low = xgb.DMatrix(X, label=df_limpio['Target_Low'])
                model_low = xgb.train(params, dtrain_low, num_boost_round=25)
                
                dtrain_close = xgb.DMatrix(X, label=df_limpio['Target_Close'])
                model_close = xgb.train(params, dtrain_close, num_boost_round=25)
                
                # Fetch last valid profile row
                x_last = X.tail(1)
                dlast = xgb.DMatrix(x_last)
                precio_apertura_hoy = float(hoy_soxl['Open'].iloc[0]) if len(hoy_soxl) >= 1 else float(x_last['Open'].iloc[0])
                
                # Reconstruction: Open Price + Predicted Net Variance Component
                preds["High"] = round(precio_apertura_hoy + float(model_high.predict(dlast)[0]), 2)
                preds["Low"] = round(precio_apertura_hoy + float(model_low.predict(dlast)[0]), 2)
                
                net_close_variation = float(model_close.predict(dlast)[0])
                preds["Close"] = round(precio_apertura_hoy + net_close_variation, 2)
                
                if len(hoy_soxl) >= 1:
                    preds["Real"] = round(precio_apertura_hoy * (1 + (variacion_ponderada_actual * 3)), 2)
                    preds["Close Real"] = round(precio_apertura_hoy * (1 + (proyeccion_ponderada_cierre * 3)), 2)
                    preds["Tendencia"] = "Alza" if net_close_variation > 0 else "Baja"
                    es_real = True

    except Exception as e:
        print(f"❌ Daily computation runtime break: {e}")
        
    print(f"🔮 Final Computed Output Structure: {preds}")
    
    # Secure isolated filesystem transaction
    try:
        id_registro = f"{today}_{modo}"
        pred_df = pd.DataFrame([preds], index=[id_registro])
        file_exists = os.path.exists(csv_filename)
        pred_df.to_csv(csv_filename, mode='a', header=not file_exists)
        print("💾 File line pushed smoothly to historical macro database.")
    except Exception as io_err:
        print(f"⚠️ Lock out alert: CSV file open inside secondary app layer: {io_err}")
    
    encabezado = "☀️ REPORTE PRE-MERCADO SOXL" if modo == "APERTURA" else "📉 REPORTE PRE-CIERRE SOXL"
    icon_tendencia = "🟢" if preds["Tendencia"] == "Alza" else "🔴"
    tipo_data = "Calculo Puro Subyacente (30 Empresas)" if es_real else "⚠️ Valores de Contingencia por Fallo"
    
    # Fragmento acortado en líneas para evitar desbordes del cuadro de copia
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
            f"💾 Historial unificado actualizado en GitHub."
        )

if __name__ == "__main__":
    run_prediction()
