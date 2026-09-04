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
    
    # Vectorización rápida y eficiente sugerida
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

def descargar_activo_seguro(ticker, period="10d", prepost=False):
    """Descarga datos de forma aislada y reporta fallos específicos por activo."""
    try:
        df = yf.download(ticker, period=period, interval="1m", prepost=prepost)
        if df.empty:
            print(f"⚠️ Alerta: Yahoo Finance devolvió datos vacíos para {ticker}")
            return pd.DataFrame()
        
        # Corrección MultiIndex estructural si yfinance los incluye de forma nativa
        if isinstance(df.columns, pd.MultiIndex):
            df = df[ticker]
            
        # Protección segura de zona horaria sugerida
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        return df
    except Exception as e:
        print(f"❌ Error crítico al descargar o procesar el activo {ticker}: {e}")
        return pd.DataFrame()

def run_scalper():
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Iniciando Pro-Scalper SOXL Refactorizado...")
    
    preds = {
        "Actual": 0.0, "Proyectado_10m": 0.0, "Tendencia": "Estable", 
        "Sesgo_VWAP": "Neutro", "Impulso": "Neutro", "Senal_Alerta": "Normal"
    }
    es_real = False
    
    # Descargas individuales limpias y seguras con logs específicos
    df_soxl = descargar_activo_seguro("SOXL", period="10d")
    df_qqq = descargar_activo_seguro("QQQ", period="10d")
    df_nvda = descargar_activo_seguro("NVDA", period="10d")
    df_aapl = descargar_activo_seguro("AAPL", period="10d")
    df_msft = descargar_activo_seguro("MSFT", period="10d")
    
    # Validar que tengamos el bloque completo para cruzar correlaciones
    if not df_soxl.empty and not df_qqq.empty and not df_nvda.empty and not df_aapl.empty and not df_msft.empty:
        try:
            # Sincronizar índices de tiempo cruzado mediante un reindexado limpio al de SOXL
            df_qqq = df_qqq.reindex(df_soxl.index, method='ffill')
            df_nvda = df_nvda.reindex(df_soxl.index, method='ffill')
            df_aapl = df_aapl.reindex(df_soxl.index, method='ffill')
            df_msft = df_msft.reindex(df_soxl.index, method='ffill')
            
            # --- CÁLCULO DE INDICADORES ---
            df_soxl['VWAP'] = calcular_vwap_diario(df_soxl)
            df_soxl['ATR'] = calcular_atr(df_soxl, period=14)
            df_soxl['EMA_9'] = df_soxl['Close'].ewm(span=9, adjust=False).mean()
            df_soxl['EMA_21'] = df_soxl['Close'].ewm(span=21, adjust=False).mean()
            
            ema_12 = df_soxl['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = df_soxl['Close'].ewm(span=26, adjust=False).mean()
            df_soxl['MACD_Line'] = ema_12 - ema_26
            df_soxl['MACD_Signal'] = df_soxl['MACD_Line'].ewm(span=9, adjust=False).mean()
            df_soxl['MACD_Hist'] = df_soxl['MACD_Line'] - df_soxl['MACD_Signal']
            
            # Flujos del Sistema de Alerta Temprana (Big Tech)
            df_soxl['qqq_trend_1m'] = df_qqq['Close'].pct_change(1)
            df_soxl['nvda_trend_1m'] = df_nvda['Close'].pct_change(1)
            df_soxl['aapl_trend_1m'] = df_aapl['Close'].pct_change(1)
            df_soxl['msft_trend_1m'] = df_msft['Close'].pct_change(1)
            
            df_soxl['Target_10m'] = df_soxl['Close'].shift(-10)
            
            # Matriz multivariable optimizada reduciendo features redundantes
            columnas_features = [
                'Close', 'Volume', 'VWAP', 'ATR', 'EMA_9', 'EMA_21', 'MACD_Hist',
                'qqq_trend_1m', 'nvda_trend_1m', 'aapl_trend_1m', 'msft_trend_1m'
            ]
            X = df_soxl[columnas_features].dropna()
            y = df_soxl['Target_10m'].loc[X.index]
            
            if len(X) > 50:
                # Mitigación de Overfitting: max_depth reducido a 3 para evitar memorizar ruido de corto plazo
                dtrain = xgb.DMatrix(X, label=y)
                model = xgb.train({'objective':'reg:squarederror', 'max_depth':3, 'eta':0.1}, dtrain, num_boost_round=30)
                
                ultimo_bloque = df_soxl[columnas_features].tail(1)
                dlast = xgb.DMatrix(ultimo_bloque)
                
                preds["Actual"] = round(float(ultimo_bloque['Close'].iloc[0]), 2)
                preds["Proyectado_10m"] = round(float(model.predict(dlast)[0]), 2)
                preds["Tendencia"] = "Alza" if preds["Proyectado_10m"] > preds["Actual"] else "Baja"
                
                ultimo_precio = preds["Actual"]
                ultimo_vwap = round(float(ultimo_bloque['VWAP'].iloc[0]), 2)
                ultima_ema9 = float(ultimo_bloque['EMA_9'].iloc[0])
                ultima_ema21 = float(ultimo_bloque['EMA_21'].iloc[0])
                ultimo_macd_hist = float(ultimo_bloque['MACD_Hist'].iloc[0])
                ultimo_atr = float(ultimo_bloque['ATR'].iloc[0])
                
                preds["Sesgo_VWAP"] = "COMPRADORES (Alcista)" if ultimo_precio > ultimo_vwap else "VENDEDORES (Bajista)"
                
                if ultimo_precio > ultima_ema9 and ultima_ema9 > ultima_ema21 and ultimo_macd_hist > 0:
                    preds["Impulso"] = "FUERTE IMPULSO ALCISTA 🔥"
                elif ultimo_precio < ultima_ema9 and ultima_ema9 < ultima_ema21 and ultimo_macd_hist < 0:
                    preds["Impulso"] = "FUERTE PRESIÓN BAJISTA 🩸"
                else:
                    preds["Impulso"] = "Compresión de Rango Transicional ⏳"
                    
                # Alerta matricial de confirmación rápida por volumen/ATR
                ret_qqq = float(df_qqq['Close'].pct_change(1).iloc[-1])
                ret_nvda = float(df_nvda['Close'].pct_change(1).iloc[-1])
                ret_aapl = float(df_aapl['Close'].pct_change(1).iloc[-1])
                ret_msft = float(df_msft['Close'].pct_change(1).iloc[-1])
                
                coincidencia_alcista = ret_qqq > 0 and ret_nvda > 0 and ret_aapl > 0 and ret_msft > 0
                coincidencia_bajista = ret_qqq < 0 and ret_nvda < 0 and ret_aapl < 0 and ret_msft < 0
                rango_vela_actual = np.abs(float(df_soxl['Close'].iloc[-1]) - float(df_soxl['Open'].iloc[-1]))
                
                if coincidencia_alcista and rango_vela_actual > (ultimo_atr * 1.2):
                    preds["Senal_Alerta"] = "🚀 RUPTURA ALCISTA CONFIRMADA POR RANGOS ATR"
                elif coincidencia_bajista and rango_vela_actual > (ultimo_atr * 1.2):
                    preds["Senal_Alerta"] = "🚨 RUPTURA BAJISTA CON EXPANSIÓN ATR"
                else:
                    preds["Senal_Alerta"] = "Flujo Normal (Filtro ATR Activo)"
                    
                es_real = True
        except Exception as e:
            print(f"❌ Error en el bucle de procesamiento técnico: {e}")
            
    # Formateo estructurado del mensaje en Telegram
    hora_actual = datetime.datetime.now().strftime("%I:%M %p")
    tipo_data = "Indicadores Estratégicos Consolidados" if es_real else "⚠️ Valores de Contingencia por Fallo"
    icon_tendencia = "🟢" if preds["Tendencia"] == "Alza" else "🔴"
    
    with open("telegram_scalper_msg.txt", "w", encoding="utf-8") as f:
        f.write(
            f"⚡ *PRO-SCALPER REFACTORIZADO* ({hora_actual} EST)\n"
            f"🔹 Estado: {tipo_data}\n"
            f"{icon_tendencia} *Señal Probabilística (10m):* {preds['Tendencia']}\n\n"
            f"💵 *Precio Actual SOXL:* ${preds['Actual']}\n"
            f"🎯 *Proyección IA Estimada:* ${preds['Proyectado_10m']}\n\n"
            f"📊 *Estrategia Intradía (VWAP Diario):*\n"
            f"▪️ Control del Día: {preds['Sesgo_VWAP']}\n"
            f"▪️ Impulso Reciente: {preds['Impulso']}\n\n"
            f"🚨 *Filtro de Rupturas (ATR 14):*\n"
            f"▪️ {preds['Senal_Alerta']}\n"
        )
    print("📝 Archivo 'telegram_scalper_msg.txt' guardado exitosamente con la arquitectura limpia.")

if __name__ == "__main__":
    run_scalper()
