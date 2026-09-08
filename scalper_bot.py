import os
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb

def calcular_vwap_diario(df):
    """Calcula el VWAP de forma vectorizada reiniciándolo desde cero cada mañana."""
    df_copy = df.copy()
    df_copy['Fecha'] = df_copy.index.date
    vp = df_copy['Close'] * df_copy['Volume']
    df_copy['Cum_VP'] = vp.groupby(df_copy['Fecha']).cumsum()
    df_copy['Cum_Vol'] = df_copy['Volume'].groupby(df_copy['Fecha']).cumsum()
    return df_copy['Cum_VP'] / df_copy['Cum_Vol'].replace(0, 1)

def calcular_atr(df, period=14):
    """Calcula el Average True Range (ATR) para medir expansión de volatilidad."""
    high_low = df['High'] - df['Low']
    high_close_prev = np.abs(df['High'] - df['Close'].shift(1))
    low_close_prev = np.abs(df['Low'] - df['Close'].shift(1))
    df_tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1)
    true_range = df_tr.max(axis=1)
    return true_range.rolling(period).mean()

def descargar_activo_seguro(ticker, period="7d", prepost=False):
    """Descarga datos de forma aislada y normaliza estrictamente las columnas."""
    try:
        df = yf.download(ticker, period=period, interval="1m", prepost=prepost, progress=False)
        if df.empty:
            print(f"⚠️ Alerta: Yahoo Finance devolvió datos vacíos para {ticker}")
            return pd.DataFrame()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.columns = [str(col).capitalize() for col in df.columns]
        columnas_necesarias = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        if not all(col in df.columns for col in columnas_necesarias):
            print(f"❌ Error: {ticker} no contiene la estructura estándar de precios.")
            return pd.DataFrame()
            
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        return df[columnas_necesarias]
    except Exception as e:
        print(f"❌ Error crítico al descargar o procesar el activo {ticker}: {e}")
        return pd.DataFrame()

def run_scalper():
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Iniciando Pro-Scalper SOXL de Grado Cuantitativo Avanzado...")
    
    csv_filename = "scalper_predictions.csv"
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    hora_key = datetime.datetime.now().strftime("%H:%M")
    id_registro = f"{today_str}_{hora_key}"
    
    preds = {
        "Actual": 0.0, "Proyectado_Cambio_10m": 0.0, "Tendencia": "Estable", 
        "Sesgo_VWAP": "Neutro", "Impulso": "Neutro", "Senal_Alerta": "Normal",
        "Nivel_Nasdaq": 0.0, "Nivel_VIX": 0.0
    }
    es_real = False
    
    df_soxl = descargar_activo_seguro("SOXL", period="7d")
    df_qqq = descargar_activo_seguro("QQQ", period="7d")
    df_nvda = descargar_activo_seguro("NVDA", period="7d")
    df_aapl = descargar_activo_seguro("AAPL", period="7d")
    df_msft = descargar_activo_seguro("MSFT", period="7d")
    df_nasdaq_raw = descargar_activo_seguro("^IXIC", period="7d") # Ampliado a 7d para entrenamiento conjunto
    df_vix_raw = descargar_activo_seguro("^VIX", period="7d")     # Ampliado a 7d para entrenamiento conjunto
    
    if not df_soxl.empty and not df_qqq.empty and not df_nvda.empty and not df_nasdaq_raw.empty and not df_vix_raw.empty:
        try:
            # Sincronización cruzada de la micro-infraestructura completa
            df_qqq = df_qqq.reindex(df_soxl.index, method='ffill')
            df_nvda = df_nvda.reindex(df_soxl.index, method='ffill')
            df_aapl = df_aapl.reindex(df_soxl.index, method='ffill')
            df_msft = df_msft.reindex(df_soxl.index, method='ffill')
            df_nasdaq = df_nasdaq_raw.reindex(df_soxl.index, method='ffill')
            df_vix = df_vix_raw.reindex(df_soxl.index, method='ffill')
            
            # --- CÁLCULO DE INDICADORES BASE ---
            df_soxl['VWAP'] = calcular_vwap_diario(df_soxl)
            df_soxl['ATR'] = calcular_atr(df_soxl, period=14)
            df_soxl['EMA_9'] = df_soxl['Close'].ewm(span=9, adjust=False).mean()
            df_soxl['EMA_21'] = df_soxl['Close'].ewm(span=21, adjust=False).mean()
            
            ema_12 = df_soxl['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = df_soxl['Close'].ewm(span=26, adjust=False).mean()
            df_soxl['MACD_Line'] = ema_12 - ema_26
            df_soxl['MACD_Signal'] = df_soxl['MACD_Line'].ewm(span=9, adjust=False).mean()
            df_soxl['MACD_Hist'] = df_soxl['MACD_Line'] - df_soxl['MACD_Signal']
            
            # Tendencias de 1 minuto de activos cruzados
            df_soxl['qqq_trend_1m'] = df_qqq['Close'].pct_change(1)
            df_soxl['nvda_trend_1m'] = df_nvda['Close'].pct_change(1)
            df_soxl['aapl_trend_1m'] = df_aapl['Close'].pct_change(1)
            df_soxl['msft_trend_1m'] = df_msft['Close'].pct_change(1)
            
            # --- IMPLEMENTACIÓN DE MEJORAS CUANTITATIVAS ---
            # 1. Features Relativas Normalizadas por Volatilidad ATR
            df_soxl['dist_vwap'] = (df_soxl['Close'] - df_soxl['VWAP']) / df_soxl['ATR'].replace(0, 0.01)
            df_soxl['dist_ema9_ema21'] = (df_soxl['EMA_9'] - df_soxl['EMA_21']) / df_soxl['ATR'].replace(0, 0.01)
            
            # 2. Features Macro Reales dentro del entrenamiento
            df_soxl['nasdaq_trend_1h'] = df_nasdaq['Close'].pct_change(12)
            df_soxl['vix_level'] = df_vix['Close']
            
            # 3. Feature Horaria Intradía (Minutos desde la apertura de Wall Street)
            df_soxl['minutos_desde_apertura'] = (df_soxl.index - df_soxl.index.normalize()).total_seconds() / 60.0 - 570.0
            
            # 4. Target de Variación Neta (Cambio absoluto en $ en lugar de precio final copia)
            df_soxl['Target_10m'] = df_soxl['Close'].shift(-10) - df_soxl['Close']
            
            # Matriz definitiva des-correlacionada y de alta varianza informativa
            columnas_features = [
                'Volume', 'dist_vwap', 'dist_ema9_ema21', 'MACD_Hist', 'vix_level', 
                'minutos_desde_apertura', 'qqq_trend_1m', 'nvda_trend_1m', 'aapl_trend_1m', 'msft_trend_1m'
            ]
            
            df_limpio = df_soxl[columnas_features + ['Target_10m']].dropna()
            X = df_limpio[columnas_features]
            y = df_limpio['Target_10m']
            
            if len(X) > 50:
                # 5. Parámetros Avanzados de Regularización L2 y Sub-muestreo para mitigar Overfitting
                params = {
                    'objective': 'reg:squarederror', 
                    'max_depth': 3, 
                    'eta': 0.1,
                    'subsample': 0.8, 
                    'colsample_bytree': 0.8, 
                    'reg_lambda': 1.0
                }
                dtrain = xgb.DMatrix(X, label=y)
                model = xgb.train(params, dtrain, num_boost_round=30)
                
                ultimo_bloque = df_soxl[columnas_features].tail(1)
                dlast = xgb.DMatrix(ultimo_bloque)
                
                preds["Actual"] = round(float(df_soxl['Close'].iloc[-1]), 2)
                cambio_estimado = float(model.predict(dlast))
                preds["Proyectado_Cambio_10m"] = round(cambio_estimado, 3)
                preds["Tendencia"] = "Alza" if cambio_estimado > 0 else "Baja"
                
                ultimo_precio = preds["Actual"]
                ultimo_vwap = round(float(df_soxl['VWAP'].iloc[-1]), 2)
                ultima_ema9 = float(df_soxl['EMA_9'].iloc[-1])
                ultima_ema21 = float(df_soxl['EMA_21'].iloc[-1])
                ultimo_macd_hist = float(df_soxl['MACD_Hist'].iloc[-1])
                ultimo_atr = float(df_soxl['ATR'].iloc[-1])
                
                preds["Sesgo_VWAP"] = "COMPRADORES" if ultimo_precio > ultimo_vwap else "VENDEDORES"
                
                if ultimo_precio > ultima_ema9 and ultima_ema9 > ultima_ema21 and ultimo_macd_hist > 0:
                    preds["Impulso"] = "ALCISTA"
                elif ultimo_precio < ultima_ema9 and ultima_ema9 < ultima_ema21 and ultimo_macd_hist < 0:
                    preds["Impulso"] = "BAJISTA"
                else:
                    preds["Impulso"] = "COMPRESION"
                    
                ret_qqq = float(df_qqq['Close'].pct_change(1).iloc[-1])
                ret_nvda = float(df_nvda['Close'].pct_change(1).iloc[-1])
                ret_aapl = float(df_aapl['Close'].pct_change(1).iloc[-1])
                ret_msft = float(df_msft['Close'].pct_change(1).iloc[-1])
                
                coincidencia_alcista = ret_qqq > 0 and ret_nvda > 0 and ret_aapl > 0 and ret_msft > 0
                coincidencia_bajista = ret_qq < 0 and ret_nvda < 0 and ret_aapl < 0 and ret_msft < 0
                rango_vela_actual = np.abs(float(df_soxl['Close'].iloc[-1]) - float(df_soxl['Open'].iloc[-1]))
                
                if coincidencia_alcista and rango_vela_actual > (ultimo_atr * 1.2):
                    preds["Senal_Alerta"] = "RUPTURA_ALCISTA"
                elif coincidencia_bajista and rango_vela_actual > (ultimo_atr * 1.2):
                    preds["Senal_Alerta"] = "RUPTURA_BAJISTA"
                else:
                    preds["Senal_Alerta"] = "NORMAL"
                
                preds["Nivel_Nasdaq"] = round(float(df_nasdaq_raw['Close'].iloc[-1]), 2)
                preds["Nivel_VIX"] = round(float(df_vix_raw['Close'].iloc[-1]), 2)
                es_real = True
        except Exception as e:
            print(f"❌ Error en procesamiento técnico interno: {e}")
            
    # --- BLINDAJE DE ESCRITURA EN HISTÓRICO CSV (MANEJO DE EXCEPCIONES AISLADO) ---
    try:
        pred_df = pd.DataFrame([preds], index=[id_registro])
