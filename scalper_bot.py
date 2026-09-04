import os
import datetime
import yfinance as yf
import pandas as pd
import xgboost as xgb

def calcular_vwap(df):
    vp = df['Close'] * df['Volume']
    cum_vp = vp.cumsum()
    cum_vol = df['Volume'].cumsum()
    return cum_vp / cum_vol.replace(0, 1)

def run_scalper():
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Iniciando Alerta Temprana Multivariable de Scalping SOXL...")
    today = datetime.date.today().strftime("%Y-%m-%d")
    
    preds = {"Actual": 0.0, "Proyectado_10m": 0.0, "Tendencia": "Estable", "Sesgo_VWAP": "Neutro", "Impulso": "Neutro", "Senal_Alerta": "Normal"}
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
        print("📥 Capturando ráfagas de 1 minuto de SOXL, QQQ, NVDA, AAPL y MSFT...")
        datos = yf.download(tickers_indice, period="2d", interval="1m")
        soxl_data = yf.download("SOXL", period="2d", interval="1m")
        qqq_data = yf.download("QQQ", period="2d", interval="1m")
        nvda_data = yf.download("NVDA", period="2d", interval="1m")
        aapl_data = yf.download("AAPL", period="2d", interval="1m")
        msft_data = yf.download("MSFT", period="2d", interval="1m")
        
        if not soxl_data.empty and not qqq_data.empty and not nvda_data.empty:
            df_soxl = soxl_data["SOXL"] if isinstance(soxl_data.columns, pd.MultiIndex) else soxl_data
            df_qqq = qqq_data["QQQ"] if isinstance(qqq_data.columns, pd.MultiIndex) else qqq_data
            df_nvda = nvda_data["NVDA"] if isinstance(nvda_data.columns, pd.MultiIndex) else nvda_data
            df_aapl = aapl_data["AAPL"] if isinstance(aapl_data.columns, pd.MultiIndex) else aapl_data
            df_msft = msft_data["MSFT"] if isinstance(msft_data.columns, pd.MultiIndex) else msft_data
            
            for df in [df_soxl, df_qqq, df_nvda, df_aapl, df_msft]:
                df.index = df.index.tz_localize(None)
            
            # --- 1. INDICADORES ---
            df_soxl['VWAP'] = calcular_vwap(df_soxl)
            df_soxl['EMA_9'] = df_soxl['Close'].ewm(span=9, adjust=False).mean()
            df_soxl['EMA_21'] = df_soxl['Close'].ewm(span=21, adjust=False).mean()
            
            ema_12 = df_soxl['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = df_soxl['Close'].ewm(span=26, adjust=False).mean()
            df_soxl['MACD_Line'] = ema_12 - ema_26
            df_soxl['MACD_Signal'] = df_soxl['MACD_Line'].ewm(span=9, adjust=False).mean()
            
            # --- 2. CORRELACIÓN CRUZADA ---
            df_soxl['qqq_trend_1m'] = df_qqq['Close'].pct_change(1)
            df_soxl['nvda_trend_1m'] = df_nvda['Close'].pct_change(1)
            df_soxl['aapl_trend_1m'] = df_aapl['Close'].pct_change(1)
            df_soxl['msft_trend_1m'] = df_msft['Close'].pct_change(1)
            
            df_soxl['Target_10m'] = df_soxl['Close'].shift(-10)
            
            columnas_features = [
                'Close', 'Volume', 'VWAP', 'EMA_9', 'EMA_21', 'MACD_Line', 'MACD_Signal',
                'qqq_trend_1m', 'nvda_trend_1m', 'aapl_trend_1m', 'msft_trend_1m'
            ]
            X = df_soxl[columnas_features].dropna()
            y = df_soxl['Target_10m'].loc[X.index]
            
            if len(X) > 10:
                dtrain = xgb.DMatrix(X, label=y)
                model = xgb.train({'objective':'reg:squarederror', 'max_depth':5, 'eta':0.1}, dtrain, num_boost_round=35)
                
                ultimo_bloque = df_soxl[columnas_features].tail(1)
                dlast = xgb.DMatrix(ultimo_bloque)
                
                preds["Actual"] = round(float(ultimo_bloque['Close'].iloc[0]), 2)
                preds["Proyectado_10m"] = round(float(model.predict(dlast)[0]), 2)
                preds["Tendencia"] = "Alza" if preds["Proyectado_10m"] > preds["Actual"] else "Baja"
                
                # --- 3. AUDITORÍA EN VIVO ---
                ultimo_precio = preds["Actual"]
                ultimo_vwap = round(float(ultimo_bloque['VWAP'].iloc[0]), 2)
                ultima_ema9 = float(ultimo_bloque['EMA_9'].iloc[0])
                ultima_ema21 = float(ultimo_bloque['EMA_21'].iloc[0])
                ultimo_macd = float(ultimo_bloque['MACD_Line'].iloc[0])
                ultima_signal = float(ultimo_bloque['MACD_Signal'].iloc[0])
                
                preds["Sesgo_VWAP"] = "COMPRADORES (Alcista)" if ultimo_precio > ultimo_vwap else "VENDEDORES (Bajista)"
                
                if ultimo_precio > ultima_ema9 and ultima_ema9 > ultima_ema21 and ultimo_macd > ultima_signal:
                    preds["Impulso"] = "FUERTE IMPULSO ALCISTA 🔥"
                elif ultimo_precio < ultima_ema9 and ultima_ema9 < ultima_ema21 and ultimo_macd < ultima_signal:
                    preds["Impulso"] = "FUERTE PRESIÓN BAJISTA 🩸"
                else:
                    preds["Impulso"] = "Compresión / Rango Transicional ⏳"
                    
                ultimo_retorno_qqq = float(df_qqq['Close'].pct_change(1).iloc[-1])
                ultimo_retorno_nvda = float(df_nvda['Close'].pct_change(1).iloc[-1])
                
                if ultimo_retorno_qqq > 0.0005 and ultimo_retorno_nvda > 0.001:
                    preds["Senal_Alerta"] = "⚡ ALERTA DE RUPTURA ALCISTA (Arrastre NVDA/QQQ)"
                elif ultimo_retorno_qqq < -0.0005 and ultimo_retorno_nvda < -0.001:
                    preds["Senal_Alerta"] = "⚠️ ALERTA DE DESPLOME INMINENTE (Fuga Institucional)"
                else:
                    preds["Senal_Alerta"] = "Flujo de Arbitraje Normal"
                    
                es_real = True

    except Exception as e:
        print(f"❌ Error en scalper: {e}")
        
    hora_actual = datetime.datetime.now().strftime("%I:%M %p")
    tipo_data = "Indicadores Estratégicos Sincronizados" if es_real else "⚠️ Valores de Contingencia"
    icon_tendencia = "🟢" if preds["Tendencia"] == "Alza" else "🔴"
    
    with open("telegram_scalper_msg.txt", "w", encoding="utf-8") as f:
        f.write(
            f"⚡ *SCALPER MULTIVARIABLE SOXL* ({hora_actual} EST)\n"
            f"🔹 Estado: {tipo_data}\n"
            f"{icon_tendencia} *Micro-Tendencia (10m):* {preds['Tendencia']}\n\n"
            f"💵 *Precio Actual:* ${preds['Actual']}\n"
            f"🎯 *Proyección 10 Minutos:* ${preds['Proyectado_10m']}\n\n"
            f"📊 *Estrategia Institucional:*\n"
            f"▪️ Control: {preds['Sesgo_VWAP']}\n"
            f"▪️ Impulso: {preds['Impulso']}\n\n"
            f"🚨 *Alerta Temprana (Correlación):*\n"
            f"▪️ {preds['Senal_Alerta']}\n"
        )

if __name__ == "__main__":
    run_scalper()
